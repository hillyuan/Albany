//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include <MiniTensor_Mechanics.h>
#include <Phalanx_DataLayout.hpp>
#include <Sacado_ParameterRegistration.hpp>
#include <Teuchos_TestForException.hpp>

#ifdef ALBANY_TIMER
#include <chrono>
#endif


//IKT: uncomment the following for debug output
#define DEBUG_OUTPUT

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
CompositeTetMassResidualBase<EvalT, Traits>::
CompositeTetMassResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
 :
      w_bf_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      ct_mass_(p.get<std::string>("Composite Tet Mass Name"), dl->node_vector), 
      out_(Teuchos::VerboseObjectBase::getDefaultOStream())
{
#ifdef DEBUG_OUTPUT 
  *out_ << "IKT CompositeTetMassResidualBase! \n"; 
#endif
  if (p.isParameter("Density"))  
    density_ = p.get<double>("Density"); 

  this->addDependentField(w_bf_);
  this->addEvaluatedField(ct_mass_);

  if (p.isType<bool>("Disable Dynamics"))
    enable_dynamics_ = !p.get<bool>("Disable Dynamics");
  else
    enable_dynamics_ = true;

  if (enable_dynamics_) {
    acceleration_ = decltype(acceleration_)(
        p.get<std::string>("Acceleration Name"), dl->qp_vector);
    this->addDependentField(acceleration_);
  }

  this->setName("CompositeTetMassResidual" + PHX::typeAsString<EvalT>());


  Teuchos::RCP<PHX::DataLayout> vector_dl = dl->node_qp_vector;
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];
  num_cells_ = dims[0]; 
#ifdef DEBUG_OUTPUT
  *out_ << "IKT num_cells_, num_nodes, num_pts_, num_dims = " << num_cells_ << ", " << num_nodes_ << ", " << num_pts_ << ", " << num_dims_ << "\n"; 
#endif

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
 
}

// **********************************************************************
template<typename EvalT, typename Traits>
void CompositeTetMassResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(ct_mass_, fm);
  if (enable_dynamics_) {
    this->utils.setFieldData(acceleration_, fm);
  }
}

template<typename EvalT, typename Traits>
std::vector<double> CompositeTetMassResidualBase<EvalT, Traits>::
compositeTetLocalMassRow(const int row) const 
{
  std::vector<double> mass_row(10); 
  //IKT, question for LCM guys: is ordering of nodes in Albany for composite
  //tet consistent with (C.4) in IJNME paper?  If not, may need to change
  //expression found here.
  //IKT, question for LCM guys: what do mass matrix entries need to be 
  //multiplied by?  I believe element mass is density_*jacobian_det.
  //IKT, question for LCM guys: how to modify residual to have effect of mass 
  //matrix / dDot term??
  switch(row) {
    case 0: 
      mass_row[0] = 1.0/80.0; mass_row[4] = 1.0/160.0;
      mass_row[6] = 1.0/160.0; mass_row[7] = 1.0/160.0;
      break; 
    case 1: 
      mass_row[1] = 1.0/80.0; mass_row[4] = 1.0/160.0;
      mass_row[5] = 1.0/160.0; mass_row[8] = 1.0/160.0;
      break;
    case 2:  
      mass_row[2] = 1.0/80.0; mass_row[5] = 1.0/160.0;
      mass_row[6] = 1.0/160.0; mass_row[9] = 1.0/160.0;
      break; 
    case 3: 
      mass_row[3] = 1.0/80.0; mass_row[7] = 1.0/160.0;
      mass_row[8] = 1.0/160.0; mass_row[9] = 1.0/160.0;
      break;
    case 4:  
      mass_row[0] = 1.0/160.0; mass_row[1] = 1.0/160.0;
      mass_row[4] = 1.0/18.0; mass_row[5] = 13.0/720.0;
      mass_row[6] = 13.0/720.0; mass_row[7] = 13.0/720.0;
      mass_row[8] = 13.0/720.0; mass_row[9] = 1.0/180.0;
      break; 
    case 5: 
      mass_row[1] = 1.0/160.0; mass_row[2] = 1.0/160.0;
      mass_row[4] = 13.0/720.0; mass_row[5] = 1.0/18.0;
      mass_row[6] = 13.0/720.0; mass_row[7] = 1.0/180.0;
      mass_row[8] = 13.0/720.0; mass_row[9] = 13.0/720.0;
      break; 
    case 6:
      mass_row[0] = 1.0/160.0; mass_row[2] = 1.0/160.0;
      mass_row[4] = 13.0/720.0; mass_row[5] = 13.0/720.0;
      mass_row[6] = 1.0/18.0; mass_row[7] = 13.0/720.0;
      mass_row[8] = 1.0/180.0; mass_row[9] = 13.0/720.0;
      break; 
    case 7: 
      mass_row[0] = 1.0/160.0; mass_row[3] = 1.0/160.0;
      mass_row[4] = 13.0/720.0; mass_row[5] = 1.0/180.0;
      mass_row[6] = 13.0/720.0; mass_row[7] = 1.0/18.0;
      mass_row[8] = 13.0/720.0; mass_row[9] = 13.0/720.0;
      break; 
    case 8: 
      mass_row[1] = 1.0/160.0; mass_row[3] = 1.0/160.0;
      mass_row[4] = 13.0/720.0; mass_row[5] = 13.0/720.0;
      mass_row[6] = 1.0/180.0; mass_row[7] = 13.0/720.0;
      mass_row[8] = 1.0/18.0; mass_row[9] = 13.0/720.0;
      break; 
    case 9:
      mass_row[2] = 1.0/160.0; mass_row[3] = 1.0/160.0;
      mass_row[4] = 1.0/160.0; mass_row[5] = 1.0/180.0;
      mass_row[6] = 13.0/720.0; mass_row[7] = 13.0/720.0;
      mass_row[8] = 13.0/720.0; mass_row[9] = 1.0/18.0;
      break; 
    default: 
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                  "Error! invalid value row = " << row << " to compositeTetLocalMassRow! \n"
                                  << "Row must be between 0 and 9.\n"); 
  }
  return mass_row; 
}

template<typename EvalT, typename Traits>
double CompositeTetMassResidualBase<EvalT, Traits>::
computeElementVolScaling(const int cell, const int node) const 
{
  double elt_vol_scale_at_node = 0.0; 
  for (int pt = 0; pt < num_pts_; ++pt) {
    elt_vol_scale_at_node += w_bf_(cell, node, pt);
  }
#ifdef DEBUG_OUTPUT
  if (cell == 0) 
    *out_ << "  IKT node, elt_vol_scale_at_node = " << node << ", " << elt_vol_scale_at_node << "\n"; 
#endif
  return elt_vol_scale_at_node; 
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
CompositeTetMassResidual<PHAL::AlbanyTraits::Residual,Traits>::
CompositeTetMassResidual(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : CompositeTetMassResidualBase<PHAL::AlbanyTraits::Residual,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void CompositeTetMassResidual<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef DEBUG_OUTPUT 
  *(this->out_) << "IKT CompositeTetMassResidual Residual Specialization evaluateFields!\n";
#endif 
 //IKT, FIXME: the following uses numerical cubature to compute the residual contribution.
 //This can also be done using the mass matrix: r = rho*M*a.  This second approach
 //needs to be implemented, and checked if it gives the same result as this first approach
 //for the composite tets. 
 for (int cell = 0; cell < workset.numCells; ++cell) {
   for (int node = 0; node < this->num_nodes_; ++node) {
     for (int pt = 0; pt < this->num_pts_; ++pt) {
       for (int dim = 0; dim < this->num_dims_; ++dim) {
         (this->ct_mass_)(cell, node, dim) +=
           (this->density_) * (this->acceleration_)(cell, pt, dim) * (this->w_bf_)(cell, node, pt);
       }
     }
   }
 }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
CompositeTetMassResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
CompositeTetMassResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : CompositeTetMassResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void CompositeTetMassResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef DEBUG_OUTPUT 
  *(this->out_) << "IKT CompositeTetMassResidual Jacobian Specialization evaluateFields!\n";
#endif 
  bool interleaved = workset.use_interleaved_order;
  double n_coeff = workset.n_coeff;
#ifdef DEBUG_OUTPUT
  *(this->out_) << "  IKT interleaved, n_coeff = " << interleaved << ", " << n_coeff << "\n"; 
#endif 
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int node = 0; node < this->num_nodes_; ++node) { //loop over Jacobian rows 
      const std::vector<double> mass_row = this->compositeTetLocalMassRow(node);
      const double elt_vol_scale_node = this->computeElementVolScaling(cell, node); 
      for (int dim = 0; dim < this->num_dims_; ++dim) {
        typename PHAL::Ref<ScalarT>::type valref = (this->ct_mass_)(cell,node,dim); //get Jacobian row 
        int k;
        for (int i=0; i < this->num_nodes_; ++i) { //loop over Jacobian cols 
          for (int j=0; j < this->num_dims_; j++) {
            if (interleaved == true) k = i*this->num_dims_ + j;
            else k = j*this->num_nodes_ + i;
            valref.fastAccessDx(k) = n_coeff*mass_row[i]*(this->density_)*elt_vol_scale_node;
          }
        }
      }
    }
  }
  
#ifdef DEBUG_OUTPUT
  for (int cell = 0; cell < workset.numCells; ++cell) {
    if (cell == 0) {
      for (int node = 0; node < this->num_nodes_; ++node) {
        for (int dim = 0; dim < this->num_dims_; ++dim) {
          *(this->out_) << "IKT node, dim, resid = " << node << ", " << dim << ", " << (this->ct_mass_)(cell, node, dim) << "\n";
        }
      }
    }
  } 
#endif
//------------------------------------------------------------------------------
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
CompositeTetMassResidual<PHAL::AlbanyTraits::Tangent, Traits>::
CompositeTetMassResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : CompositeTetMassResidualBase<PHAL::AlbanyTraits::Tangent,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void CompositeTetMassResidual<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //IKT, FIXME: fill in!
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
CompositeTetMassResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
CompositeTetMassResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : CompositeTetMassResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl) {}

// **********************************************************************
template<typename Traits>
void CompositeTetMassResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //IKT, FIXME: fill in! 
}

}

