//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include <string>

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Sacado_ParameterRegistration.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN


namespace PHAL {
template <typename EvalT, typename Traits>
BodySourceBase<EvalT, Traits> :: BodySourceBase(Teuchos::ParameterList & p) :
  dl             (p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  meshSpecs      (p.get<Teuchos::RCP<Albany::MeshSpecsStruct> >("Mesh Specs Struct"))
{
	//Teuchos::RCP<PHX::DataLayout> vector_dl = dl->qp_vector;
    std::vector<PHX::DataLayout::size_type> dim;
    dl->qp_tensor->dimensions(dim);
    numCells = dim[0];
    numQPs = dim[1];
    cellDims = dim[2];
}
	
template <typename EvalT, typename Traits>
BodySource<EvalT, Traits>::
BodySource(Teuchos::ParameterList & p)
: BodySourceBase<EvalT, Traits>(p)
 /*   : body_force_("Body Force", dl->qp_vector),
      density_(p.get<RealType>("Density")),
      weights_("Weights", dl->qp_scalar),
      coordinates_("Coord Vec", dl->qp_vector)*/
{
  std::string const &
  type = p.get("Type", "Gravity");

  if (type == "Gravity") {
    Gravity<EvalT, Traits>  *q = new Gravity<EvalT, Traits>(p);
	m_sources_.push_back( q );
  } else if (type == "Centripetal") {
    Centripetal<EvalT, Traits>  *q = new Centripetal<EvalT, Traits>(p);
	m_sources_.push_back( q );
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, Teuchos::Exceptions::InvalidParameter,
        "Invalid body force type " << type);
  }
  
  PHX::MDField<const ScalarT,Cell,Node,VecDim> dofVec = decltype(dofVec)(
           p.get<std::string>("DOF Name"),
           p.get<Teuchos::RCP<PHX::DataLayout> >("DOF Data Layout"));
//  int const num_cells = workset.numCells;
//  m_outsource_ = Kokkos::createDynRankViewWithType<Kokkos::DynRankView<ScalarT, PHX::Device> >
//         (dofVec.get_view(), "DDN", numCells, numNodes, numDOFsSet);
  
/*
  
  this->addDependentField(weights_);
  this->addEvaluatedField(body_force_);
  this->setName("Body Force" + PHX::typeAsString<EvalT>());*/
}

//
//
//
template <typename EvalT, typename Traits>
void
BodySource<EvalT, Traits>::
postRegistrationSetup(
    typename Traits::SetupData d, PHX::FieldManager<Traits> & fm)
{
 /* this->utils.setFieldData(body_force_, fm);
  if (is_constant_ == false) this->utils.setFieldData(coordinates_, fm);
  if (is_constant_ == false) this->utils.setFieldData(weights_, fm);*/
}

//
//
//
template<typename EvalT, typename Traits>
void
BodySource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int const num_cells = workset.numCells;
/*
  if (is_constant_ == true) {
    for (int cell = 0; cell < num_cells; ++cell) {
      for (int qp = 0; qp < num_qp_; ++qp) {
        for (int dim = 0; dim < num_dim_; ++dim) {
          body_force_(cell, qp, dim) = constant_value_[dim];
        }
      }
    }
  }
  else {

    double omega2 = this->angular_frequency_ * this->angular_frequency_;
    ScalarT qpmass, f_mag;
    MeshScalarT xyz[3], len2, dot, r, f_dir[3];

    for (int cell = 0; cell < num_cells; ++cell) {
      for (std::size_t qp = 0; qp < num_qp_; ++qp) {

        // Determine the qp's distance from the axis of rotation
        len2 = dot = 0.;
        for (std::size_t dim = 0; dim < num_dim_; dim++) {

          xyz[dim] = f_dir[dim] = this->coordinates_(cell, qp, dim)
              - this->rotation_center_[dim];
          dot += xyz[dim] * this->rotation_axis_[dim];
          len2 += xyz[dim] * xyz[dim];
        }
        r = std::sqrt(len2 - dot * dot);

        // Determine the direction of force due to centripetal acceleration
        len2 = 0.;
        for (std::size_t dim = 0; dim < num_dim_; dim++) {

          f_dir[dim] -= this->rotation_axis_[dim] * dot;
          len2 += f_dir[dim] * f_dir[dim];
        }
        double len_reciprocal = 1. / sqrt(len2);
        for (std::size_t dim = 0; dim < num_dim_; dim++) {

          f_dir[dim] *= len_reciprocal;
        }

        // Determine the qp's mass
        // qpmass = weights_(cell,qp) * density_(cell, qp);
        // qp volume * density - Is this right?
        qpmass = weights_(cell, qp) * density_;
        f_mag = qpmass * omega2 * r;
        for (std::size_t dim = 0; dim < num_dim_; dim++)

          this->body_force_(cell, qp, dim) = f_dir[dim] * f_mag;

      }
    }
  }*/
}

template <typename EvalT, typename Traits>
Gravity<EvalT, Traits> :: Gravity(Teuchos::ParameterList & p)
: BodySourceBase<EvalT, Traits>(p)
{
  m_acc_ = p.get<RealType>("Acceleration");
  m_direction_ = p.get<Teuchos::Array<RealType>>("Direction",
        Teuchos::tuple<double>(1.0, 0.0, 0.0));
}

template<typename EvalT, typename Traits>
void Gravity<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int const num_cells = workset.numCells;
  std::cout << this->numQPs ;
}

template <typename EvalT, typename Traits>
Centripetal<EvalT, Traits>::
Centripetal(Teuchos::ParameterList & p)
: BodySourceBase<EvalT, Traits>(p)
{
    rotation_center_ = p.get<Teuchos::Array<RealType>>("Rotation Center",
        Teuchos::tuple<double>(0.0, 0.0, 0.0));
    rotation_axis_ = p.get<Teuchos::Array<RealType>>("Rotation Axis",
        Teuchos::tuple<double>(0.0, 0.0, 0.0));
    angular_frequency_ = p.get<RealType>("Angular Frequency", 0.0);
	
	 // Ensure that axisDirection is normalized
    double len = 0.0;
    for (int i = 0; i < 3; i++){
      len += this->rotation_axis_[i] * this->rotation_axis_[i];
    }

    len = sqrt(len);
    for (int i = 0; i < 3; i++){
      this->rotation_axis_[i] /= len;
    }
	
	coordinates_ = PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim>("Coord Vec", this->dl->qp_vector);
	this->addDependentField(coordinates_);
}


}
