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

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
CompositeTetMassResidualBase<EvalT, Traits>::
CompositeTetMassResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
 :
      w_grad_bf_(
          p.get<std::string>("Weighted Gradient BF Name"),
          dl->node_qp_vector),
      w_bf_(p.get<std::string>("Weighted BF Name"), dl->node_qp_scalar),
      residual_(p.get<std::string>("Residual Name"), dl->node_vector) 
{
  //IKT, FIXME: modify this as needed! 

  if (p.isParameter("Density"))  
    density_ = p.get<double>("Density"); 

  this->addDependentField(w_grad_bf_);
  this->addDependentField(w_bf_);

  this->addEvaluatedField(residual_);

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


  std::vector<PHX::DataLayout::size_type> dims;
  w_grad_bf_.fieldTag().dataLayout().dimensions(dims);
  num_nodes_ = dims[1];
  num_pts_   = dims[2];
  num_dims_  = dims[3];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
 
}

// **********************************************************************
template<typename EvalT, typename Traits>
void CompositeTetMassResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  //IKT, FIXME: modify this as needed!
  
  this->utils.setFieldData(w_grad_bf_, fm);
  this->utils.setFieldData(w_bf_, fm);
  this->utils.setFieldData(residual_, fm);
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
  //IKT, FIXME: fill in!
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
  //IKT, FIXME: fill in!
  //For now, just checking that can call routine
  const std::vector<double> mass_row = this->compositeTetLocalMassRow(0); 
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

