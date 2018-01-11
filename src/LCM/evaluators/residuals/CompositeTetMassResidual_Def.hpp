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

