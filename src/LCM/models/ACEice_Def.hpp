//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "ACEice.hpp"
#include "Albany_Utils.hpp"
#include "MiniNonlinearSolver.h"

namespace LCM {

template <typename EvalT, typename Traits>
ACEiceMiniKernel<EvalT, Traits>::ACEiceMiniKernel(
    ConstitutiveModel<EvalT, Traits>&    model,
    Teuchos::ParameterList*              p,
    Teuchos::RCP<Albany::Layouts> const& dl)
    : BaseKernel(model)
{
  // Baseline constants
  sat_mod_ = p->get<RealType>("Saturation Modulus", 0.0);
  sat_exp_ = p->get<RealType>("Saturation Exponent", 0.0);

  ice_density_          = p->get<RealType>("ACE Ice Density", 0.0);
  water_density_        = p->get<RealType>("ACE Water Density", 0.0);
  ice_thermal_cond_     = p->get<RealType>("ACE Ice Thermal Conductivity", 0.0);
  water_thermal_cond_   = p->get<RealType>("ACE Water Thermal Conductivity", 0.0);
  ice_heat_capacity_    = p->get<RealType>("ACE Ice Heat Capacity", 0.0);
  water_heat_capacity_  = p->get<RealType>("ACE Water Heat Capacity", 0.0);

  ice_saturation_init_  = p->get<RealType>("ACE Ice Initial Saturation", 0.0);
  ice_saturation_max_   = p->get<RealType>("ACE Ice Maximum Saturation", 0.0);
  water_saturation_min_ = p->get<RealType>("ACE Water Minimum Saturation", 0.0);
  porosity_             = p->get<RealType>("ACE Porosity", 0.0);

  // retrieve appropriate field name strings
  std::string const cauchy_string       = field_name_map_["Cauchy_Stress"];
  std::string const Fp_string           = field_name_map_["Fp"];
  std::string const eqps_string         = field_name_map_["eqps"];
  std::string const yieldSurface_string = field_name_map_["Yield_Surface"];
  std::string const source_string       = field_name_map_["Mechanical_Source"];
  std::string const F_string            = field_name_map_["F"];
  std::string const J_string            = field_name_map_["J"];

  // define the dependent fields
  setDependentField(F_string, dl->qp_tensor);
  setDependentField(J_string, dl->qp_scalar);
  setDependentField("Elastic Modulus", dl->qp_scalar);
  setDependentField("Hardening Modulus", dl->qp_scalar);
  setDependentField("Poissons Ratio", dl->qp_scalar);
  setDependentField("Yield Strength", dl->qp_scalar);
  setDependentField("Delta Time", dl->workset_scalar);

  // Computed incrementally
  setEvaluatedField("ACE Ice Saturation", dl->qp_scalar);

  // For output/convenience
  setEvaluatedField("ACE Density", dl->qp_scalar);
  setEvaluatedField("ACE Heat Capacity", dl->qp_scalar);
  setEvaluatedField("ACE Thermal Conductivity", dl->qp_scalar);
  setEvaluatedField("ACE Water Saturation", dl->qp_scalar);

  // define the evaluated fields
  setEvaluatedField(cauchy_string, dl->qp_tensor);
  setEvaluatedField(Fp_string, dl->qp_tensor);
  setEvaluatedField(eqps_string, dl->qp_scalar);
  setEvaluatedField(yieldSurface_string, dl->qp_scalar);
  if (have_temperature_ == true) {
    setEvaluatedField(source_string, dl->qp_scalar);
  }

  // define the state variables

  // stress
  addStateVariable(
      cauchy_string,
      dl->qp_tensor,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output Cauchy Stress", false));

  // Fp
  addStateVariable(
      Fp_string,
      dl->qp_tensor,
      "identity",
      0.0,
      true,
      p->get<bool>("Output Fp", false));

  // eqps
  addStateVariable(
      eqps_string,
      dl->qp_scalar,
      "scalar",
      0.0,
      true,
      p->get<bool>("Output eqps", false));

  // yield surface
  addStateVariable(
      yieldSurface_string,
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output Yield Surface", false));

  // Ice saturation
  addStateVariable(
      "ACE Ice Saturation",
      dl->qp_scalar,
      "scalar",
      1.0,
      false,
      p->get<bool>("Output ACE Ice Saturation", false));

  // Density
  addStateVariable(
      "ACE Density",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output ACE Density", false));

  // Heat Capacity
  addStateVariable(
      "ACE Heat Capacity",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output ACE Heat Capacity", false));

  // ACE Thermal Conductivity
  addStateVariable(
      "ACE Thermal Conductivity",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("Output ACE Thermal Conductivity", false));

  // ACE Water Saturation
  addStateVariable(
      "ACE Water Saturation",
      dl->qp_scalar,
      "scalar",
      0.0,
      false,
      p->get<bool>("ACE Water Saturation", false));

  // mechanical source
  if (have_temperature_ == true) {
    addStateVariable(
        source_string,
        dl->qp_scalar,
        "scalar",
        0.0,
        false,
        p->get<bool>("Output Mechanical Source", false));
  }
}

template <typename EvalT, typename Traits>
void
ACEiceMiniKernel<EvalT, Traits>::init(
    Workset&                 workset,
    FieldMap<const ScalarT>& input_fields,
    FieldMap<ScalarT>&       output_fields)
{
  std::string cauchy_string       = field_name_map_["Cauchy_Stress"];
  std::string Fp_string           = field_name_map_["Fp"];
  std::string eqps_string         = field_name_map_["eqps"];
  std::string yieldSurface_string = field_name_map_["Yield_Surface"];
  std::string source_string       = field_name_map_["Mechanical_Source"];
  std::string F_string            = field_name_map_["F"];
  std::string J_string            = field_name_map_["J"];

  def_grad_ = *input_fields[F_string];
  J_        = *input_fields[J_string];

  elastic_modulus_   = *input_fields["Elastic Modulus"];
  hardening_modulus_ = *input_fields["Hardening Modulus"];
  poissons_ratio_    = *input_fields["Poissons Ratio"];
  yield_strength_    = *input_fields["Yield Strength"];

  delta_time_ = *input_fields["Delta Time"];

  stress_     = *output_fields[cauchy_string];
  Fp_         = *output_fields[Fp_string];
  eqps_       = *output_fields[eqps_string];
  yield_surf_ = *output_fields[yieldSurface_string];

  ice_saturation_   = *output_fields["ACE Ice Saturation"];
  density_          = *output_fields["ACE Density"];
  heat_capacity_    = *output_fields["ACE Heat Capacity"];
  thermal_cond_     = *output_fields["ACE Thermal Conductivity"];
  water_saturation_ = *output_fields["ACE Water Saturation"];

  if (have_temperature_ == true) {
    source_ = *output_fields[source_string];
  }

  // get State Variables
  Fp_old_              = (*workset.stateArrayPtr)[Fp_string + "_old"];
  eqps_old_            = (*workset.stateArrayPtr)[eqps_string + "_old"];
  T_old_               = (*workset.stateArrayPtr)["Temperature_old"];
  ice_saturation_old_  = (*workset.stateArrayPtr)["ACE Ice Saturation_old"];
}

namespace {

static RealType const SQ23{std::sqrt(2.0 / 3.0)};

}  // anonymous namespace

//
// ACE ice nonlinear system
//
template <typename EvalT, minitensor::Index M = 1>
class IceNLS : public minitensor::
                   Function_Base<IceNLS<EvalT, M>, typename EvalT::ScalarT, M>
{
  using S = typename EvalT::ScalarT;

 public:
  IceNLS(
      RealType sat_mod,
      RealType sat_exp,
      RealType eqps_old,
      S const& K,
      S const& smag,
      S const& mubar,
      S const& Y)
      : sat_mod_(sat_mod), sat_exp_(sat_exp), eqps_old_(eqps_old), K_(K),
        smag_(smag), mubar_(mubar), Y_(Y)
  {
  }

  constexpr static char const* const NAME{"ACE ice NLS"};

  using Base =
      minitensor::Function_Base<IceNLS<EvalT, M>, typename EvalT::ScalarT, M>;

  // Default value.
  template <typename T, minitensor::Index N>
  T
  value(minitensor::Vector<T, N> const& x)
  {
    return Base::value(*this, x);
  }

  // Explicit gradient.
  template <typename T, minitensor::Index N>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const& x)
  {
    // Firewalls.
    minitensor::Index const dimension = x.get_dimension();

    ALBANY_EXPECT(dimension == Base::DIMENSION);

    // Variables that potentially have Albany::Traits sensitivity
    // information need to be handled by the peel functor so that
    // proper conversions take place.
    T const K     = peel<EvalT, T, N>()(K_);
    T const smag  = peel<EvalT, T, N>()(smag_);
    T const mubar = peel<EvalT, T, N>()(mubar_);
    T const Y     = peel<EvalT, T, N>()(Y_);

    // This is the actual computation of the gradient.
    minitensor::Vector<T, N> r(dimension);

    T const& X     = x(0);
    T const  alpha = eqps_old_ + SQ23 * X;
    T const  H     = K * alpha + sat_mod_ * (1.0 - std::exp(-sat_exp_ * alpha));
    T const  R     = smag - (2.0 * mubar * X + SQ23 * (Y + H));

    r(0) = R;

    return r;
  }

  // Default AD hessian.
  template <typename T, minitensor::Index N>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const& x)
  {
    return Base::hessian(*this, x);
  }

  // Constants.
  RealType const sat_mod_{0.0};
  RealType const sat_exp_{0.0};
  RealType const eqps_old_{0.0};

  // Inputs
  S const& K_;
  S const& smag_;
  S const& mubar_;
  S const& Y_;
};

template <typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION void
ACEiceMiniKernel<EvalT, Traits>::operator()(int cell, int pt) const
{
  constexpr minitensor::Index MAX_DIM{3};

  using Tensor = minitensor::Tensor<ScalarT, MAX_DIM>;

  Tensor       F(num_dims_);
  Tensor const I(minitensor::eye<ScalarT, MAX_DIM>(num_dims_));
  Tensor       sigma(num_dims_);

  ScalarT const E     = elastic_modulus_(cell, pt);
  ScalarT const nu    = poissons_ratio_(cell, pt);
  ScalarT const kappa = E / (3.0 * (1.0 - 2.0 * nu));
  ScalarT const mu    = E / (2.0 * (1.0 + nu));
  ScalarT const K     = hardening_modulus_(cell, pt);
  ScalarT const Y     = yield_strength_(cell, pt);
  ScalarT const J1    = J_(cell, pt);
  ScalarT const Jm23  = 1.0 / std::cbrt(J1 * J1);

  // fill local tensors
  F.fill(def_grad_, cell, pt, 0, 0);

  // Mechanical deformation gradient
  auto Fm = Tensor(F);
  if (have_temperature_) {
    ScalarT dtemp           = temperature_(cell, pt) - ref_temperature_;
    ScalarT thermal_stretch = std::exp(expansion_coeff_ * dtemp);
    Fm /= thermal_stretch;
  }

  Tensor Fpn(num_dims_);

  for (int i{0}; i < num_dims_; ++i) {
    for (int j{0}; j < num_dims_; ++j) {
      Fpn(i, j) = ScalarT(Fp_old_(cell, pt, i, j));
    }
  }
  
  // Calculate melting temperature
  ScalarT sal = 0.10;  // note: this should come from chemical part of model
  ScalarT sal15 = std::sqrt(sal * sal * sal);
  ScalarT pressure = 101325.0; // [Pa] 
  // pressure = (1.0/3.0)*minitensor::trace(stress_(cell, pt)); 
  ScalarT Tmelt =
      (-0.057 * sal) + (0.00170523 * sal15) - (0.0002154996 * sal * sal) - 
      ((0.000753/10000.0) * pressure);
      
  // Calculate temperature change (not sure where temperature_ comes from, but 
  // it seems to be getting used in here already for the source term, and I
  // assume its the current temperature value)
  ScalarT dTemp = temperature_(cell, pt) - T_old_(cell, pt);
  
  // Calculate the freezing curve function df/dTemp
  ScalarT T_range = 1.0;
  ScalarT T_low   = Tmelt - (T_range/2.0);
  ScalarT T_high  = Tmelt + (T_range/2.0);
  ScalarT i_sat_evaluated;
  
  // completely frozen
  if (temperature_(cell, pt) <= T_low) {
    i_sat_evaluated = 1.0;
  }
  // completely melted
  if (temperature_(cell, pt) >= T_high) {
    i_sat_evaluated = 0.0;
  }
  // in phase change
  if ((temperature_(cell, pt) > T_low) && (temperature_(cell, pt) < T_high)) {
    i_sat_evaluated = -1.0*(temperature_(cell, pt)/T_range) + T_high;
  }
  
  ScalarT dfdT = (i_sat_evaluated - ice_saturation_old_(cell, pt))/dTemp;
  
  // Update the ice saturation
  ice_saturation_(cell, pt) += dfdT * dTemp;
  ice_saturation_(cell, pt) = std::max(0.0, ice_saturation_(cell, pt));
  ice_saturation_(cell, pt) = std::min(ice_saturation_max_, 
                                       ice_saturation_(cell, pt));
  
  // Update the water saturation
  water_saturation_(cell, pt) = 1.0 - ice_saturation_(cell, pt);
  water_saturation_(cell, pt) = std::max(water_saturation_min_,
                                         water_saturation_(cell, pt));
  water_saturation_(cell, pt) = std::min(1.0,water_saturation_(cell, pt));
  
  // The following calculations need porosity: density, heat capacity, thermal
  // conductivity. However, we need to calculate a porosity field at the 
  // beginning of the simulation based on lithostatic pressure and surface
  // porosity value. Porosity doesn't change in time.
  ScalarT porosity_ = 0.65;
  
  // Update the effective material density
  density_(cell, pt) = 
      porosity_*(ice_density_*ice_saturation_(cell, pt) + 
                 water_density_*water_saturation_(cell, pt));
      
  // Update the effective material heat capacity
  heat_capacity_(cell, pt) = 
      porosity_*(ice_heat_capacity_*ice_saturation_(cell, pt) + 
                 water_heat_capacity_*water_saturation_(cell, pt));
     
  // Update the effective material thermal conductivity
  thermal_cond_(cell, pt) = 
      pow(ice_thermal_cond_,(ice_saturation_(cell, pt)*porosity_)) * 
      pow(water_thermal_cond_,(water_saturation_(cell, pt)*porosity_));
  
  // Swap for old variables
  // these cause compiler errors!!
  //ice_saturation_old_(cell, pt) = ice_saturation_(cell, pt);
  //T_old_(cell, pt) = temperature_(cell, pt);

  // compute trial state
  Tensor const  Fpinv = minitensor::inverse(Fpn);
  Tensor const  Cpinv = Fpinv * minitensor::transpose(Fpinv);
  Tensor const  be    = Jm23 * Fm * Cpinv * minitensor::transpose(Fm);
  Tensor        s     = mu * minitensor::dev(be);
  ScalarT const mubar = minitensor::trace(be) * mu / (num_dims_);

  // check yield condition
  ScalarT const smag = minitensor::norm(s);
  ScalarT const f =
      smag -
      SQ23 * (Y + K * eqps_old_(cell, pt) +
              sat_mod_ * (1.0 - std::exp(-sat_exp_ * eqps_old_(cell, pt))));

  RealType constexpr yield_tolerance = 1.0e-12;

  if (f > yield_tolerance) {
    // Use minimization equivalent to return mapping
    using ValueT = typename Sacado::ValueType<ScalarT>::type;
    using NLS    = IceNLS<EvalT>;

    constexpr minitensor::Index nls_dim{NLS::DIMENSION};

    using MIN  = minitensor::Minimizer<ValueT, nls_dim>;
    using STEP = minitensor::NewtonStep<NLS, ValueT, nls_dim>;

    MIN  minimizer;
    STEP step;
    NLS  j2nls(sat_mod_, sat_exp_, eqps_old_(cell, pt), K, smag, mubar, Y);

    minitensor::Vector<ScalarT, nls_dim> x;

    x(0) = 0.0;

    LCM::MiniSolver<MIN, STEP, NLS, EvalT, nls_dim> mini_solver(
        minimizer, step, j2nls, x);

    ScalarT const alpha = eqps_old_(cell, pt) + SQ23 * x(0);
    ScalarT const H     = K * alpha + sat_mod_ * (1.0 - exp(-sat_exp_ * alpha));
    ScalarT const dgam  = x(0);

    // plastic direction
    Tensor const N = (1 / smag) * s;

    // update s
    s -= 2 * mubar * dgam * N;

    // update eqps
    eqps_(cell, pt) = alpha;

    // mechanical source
    if (have_temperature_ == true && delta_time_(0) > 0) {
      source_(cell, pt) =
          (SQ23 * dgam / delta_time_(0) * (Y + H + temperature_(cell, pt))) /
          (density_(cell, pt) * heat_capacity_(cell, pt));
    }

    // exponential map to get Fpnew
    Tensor const A     = dgam * N;
    Tensor const expA  = minitensor::exp(A);
    Tensor const Fpnew = expA * Fpn;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) {
        Fp_(cell, pt, i, j) = Fpnew(i, j);
      }
    }
  } else {
    eqps_(cell, pt) = eqps_old_(cell, pt);

    if (have_temperature_ == true) source_(cell, pt) = 0.0;

    for (int i{0}; i < num_dims_; ++i) {
      for (int j{0}; j < num_dims_; ++j) {
        Fp_(cell, pt, i, j) = Fpn(i, j);
      }
    }
  }

  // update yield surface
  yield_surf_(cell, pt) =
      Y + K * eqps_(cell, pt) +
      sat_mod_ * (1. - std::exp(-sat_exp_ * eqps_(cell, pt)));

  // compute pressure
  ScalarT const p = 0.5 * kappa * (J_(cell, pt) - 1. / (J_(cell, pt)));

  // compute stress
  sigma = p * I + s / J_(cell, pt);

  for (int i(0); i < num_dims_; ++i) {
    for (int j(0); j < num_dims_; ++j) {
      stress_(cell, pt, i, j) = sigma(i, j);
    }
  }
}
}  // namespace LCM
