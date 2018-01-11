//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifdef ALBANY_TIMER
#include <chrono>
#endif
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace LCM {

template<typename EvalT, typename Traits>
CompositeTetMassResidualBase<EvalT, Traits>::
CompositeTetMassResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
{
  //IKT, FIXME: fill in! 
}

// **********************************************************************
template<typename EvalT, typename Traits>
void CompositeTetMassResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  //IKT, FIXME: fill in! 
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

