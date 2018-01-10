//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SCATTER_RESIDUAL_HPP
#define PHAL_SCATTER_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#ifdef ALBANY_EPETRA
#include "Epetra_Vector.h"
#endif

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
#include "Kokkos_Vector.hpp"
#endif

namespace PHAL {
/** \brief Scatters result from the residual fields into the
    global (epetra) data structurs.  This includes the
    post-processing of the AD data type for all evaluation
    types besides Residual.

*/
// **************************************************************
// Base Class for code that is independent of evaluation type
// **************************************************************

template<typename EvalT, typename Traits>
class ScatterResidualBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:
  ScatterResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d)=0;

protected:
  typedef typename EvalT::ScalarT ScalarT;
  Teuchos::RCP<PHX::FieldTag> scatter_operation;
  std::vector< PHX::MDField<ScalarT const,Cell,Node> > val;
  PHX::MDField<ScalarT const,Cell,Node,Dim>  valVec;
  PHX::MDField<ScalarT const,Cell,Node,Dim,Dim> valTensor;
  std::size_t numNodes;
  std::size_t numFieldsBase; // Number of fields gathered in this call
  std::size_t offset; // Offset of first DOF being gathered when numFields<neq

  unsigned short int tensorRank;

  std::vector<double> compositeTetLocalMassRow(const int row) 
    {std::vector<double> vec(10); return vec;}; 

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
protected:
  Albany::AbstractDiscretization::WorksetConn nodeID;
  Kokkos::View<ST*, PHX::Device> fT_kokkos;
  Kokkos::vector<Kokkos::DynRankView<const ScalarT, PHX::Device>, PHX::Device> val_kokkos;

#endif
};

template<typename EvalT, typename Traits> class ScatterResidual;

template<typename EvalT, typename Traits>
class ScatterResidualWithExtrudedParams
  : public ScatterResidual<EvalT, Traits> {

public:

  ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
                                ScatterResidual<EvalT, Traits>(p,dl) {
    extruded_params_levels = p.get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    ScatterResidual<EvalT, Traits>::postRegistrationSetup(d,vm);
  }

  void evaluateFields(typename Traits::EvalData d) {
    ScatterResidual<EvalT, Traits>::evaluateFields(d);
  }

protected:

  typedef typename EvalT::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;
  std::vector<double> compositeTetLocalMassRow(const int row) 
    {std::vector<double> vec(10); return vec;}; 

};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
  std::vector<double> compositeTetLocalMassRow(const int row) 
    {std::vector<double> vec(10); return vec;}; 

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  struct PHAL_ScatterResRank0_Tag{};
  struct PHAL_ScatterResRank1_Tag{};
  struct PHAL_ScatterResRank2_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank2_Tag&, const int& cell) const;

private:
  int numDims;

  typedef ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits> Base;
  using Base::nodeID;
  using Base::fT_kokkos;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank0_Tag> PHAL_ScatterResRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank1_Tag> PHAL_ScatterResRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank2_Tag> PHAL_ScatterResRank2_Policy;

#endif
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
  //IKT, FIXME: probably this function should be moved somewhere else
  std::vector<double> compositeTetLocalMassRow(const int row) const 
    {std::vector<double> mass_row(10); 
     //IKT, question for LCM guys: is ordering of nodes in Albany for composite
     //tet consistent with (C.4) in IJNME paper?  If not, may need to change
     //expression found here.
     //IKT, question for LCM guys: what do mass matrix entries need to be 
     //multiplied by? 
     //IKT, question for LCM guys: how to modify residual to have effect of mass 
     //matrix / dDot term??
     //IKT, question for LCM guys: do we have analytic expression for mass for regular tets or hexes,
     //to facilitated with debugging? 
     if (row == 0) {
       mass_row[0] = 1.0/80.0; mass_row[4] = 1.0/160.0; 
       mass_row[6] = 1.0/160.0; mass_row[7] = 1.0/160.0;  
     }
     else if (row == 1) {
       mass_row[1] = 1.0/80.0; mass_row[4] = 1.0/160.0; 
       mass_row[5] = 1.0/160.0; mass_row[8] = 1.0/160.0;  
     }
     else if (row == 2) {
       mass_row[2] = 1.0/80.0; mass_row[5] = 1.0/160.0; 
       mass_row[6] = 1.0/160.0; mass_row[9] = 1.0/160.0;  
     }
     else if (row == 3) {
       mass_row[3] = 1.0/80.0; mass_row[7] = 1.0/160.0; 
       mass_row[8] = 1.0/160.0; mass_row[9] = 1.0/160.0;  
     }
     else if (row == 4) {
       mass_row[0] = 1.0/160.0; mass_row[1] = 1.0/160.0; 
       mass_row[4] = 1.0/18.0; mass_row[5] = 13.0/720.0; 
       mass_row[6] = 13.0/720.0; mass_row[7] = 13.0/720.0; 
       mass_row[8] = 13.0/720.0; mass_row[9] = 1.0/180.0; 
     }
     else if (row == 5) {
       mass_row[1] = 1.0/160.0; mass_row[2] = 1.0/160.0; 
       mass_row[4] = 13.0/720.0; mass_row[5] = 1.0/18.0; 
       mass_row[6] = 13.0/720.0; mass_row[7] = 1.0/180.0; 
       mass_row[8] = 13.0/720.0; mass_row[9] = 13.0/720.0; 
     }
     else if (row == 6) {
       mass_row[0] = 1.0/160.0; mass_row[2] = 1.0/160.0; 
       mass_row[4] = 13.0/720.0; mass_row[5] = 13.0/720.0; 
       mass_row[6] = 1.0/18.0; mass_row[7] = 13.0/720.0; 
       mass_row[8] = 1.0/180.0; mass_row[9] = 13.0/720.0; 
     }
     else if (row == 7) {
       mass_row[0] = 1.0/160.0; mass_row[3] = 1.0/160.0; 
       mass_row[4] = 13.0/720.0; mass_row[5] = 1.0/180.0; 
       mass_row[6] = 13.0/720.0; mass_row[7] = 1.0/18.0; 
       mass_row[8] = 13.0/720.0; mass_row[9] = 13.0/720.0; 
     }
     else if (row == 8) {
       mass_row[1] = 1.0/160.0; mass_row[3] = 1.0/160.0; 
       mass_row[4] = 13.0/720.0; mass_row[5] = 13.0/720.0; 
       mass_row[6] = 1.0/180.0; mass_row[7] = 13.0/720.0; 
       mass_row[8] = 1.0/18.0; mass_row[9] = 13.0/720.0; 
     }
     else if (row == 9) {
       mass_row[2] = 1.0/160.0; mass_row[3] = 1.0/160.0; 
       mass_row[4] = 1.0/160.0; mass_row[5] = 1.0/180.0; 
       mass_row[6] = 13.0/720.0; mass_row[7] = 13.0/720.0; 
       mass_row[8] = 13.0/720.0; mass_row[9] = 1.0/18.0; 
     }
     return mass_row;}; 

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  struct PHAL_ScatterResRank0_Tag{};
  struct PHAL_ScatterJacRank0_Adjoint_Tag{};
  struct PHAL_ScatterJacRank0_Tag{};
  struct PHAL_ScatterResRank1_Tag{};
  struct PHAL_ScatterJacRank1_Adjoint_Tag{};
  struct PHAL_ScatterJacRank1_Tag{};
  struct PHAL_ScatterCompositeTetMassRank1_Tag{};
  struct PHAL_ScatterResRank2_Tag{};
  struct PHAL_ScatterJacRank2_Adjoint_Tag{};
  struct PHAL_ScatterJacRank2_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank0_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank0_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank1_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterCompositeTetMassRank1_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterResRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank2_Adjoint_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_ScatterJacRank2_Tag&, const int& cell) const;

private:
  int neq, nunk, numDims;
  bool interleaved;
  double n_coeff;  
  Tpetra_CrsMatrix::local_matrix_type JacT_kokkos;

  typedef ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;
  using Base::nodeID;
  using Base::fT_kokkos;
  using Base::val_kokkos;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank0_Tag> PHAL_ScatterResRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank0_Adjoint_Tag> PHAL_ScatterJacRank0_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank0_Tag> PHAL_ScatterJacRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank1_Tag> PHAL_ScatterResRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank1_Adjoint_Tag> PHAL_ScatterJacRank1_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank1_Tag> PHAL_ScatterJacRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterCompositeTetMassRank1_Tag> PHAL_ScatterCompositeTetMassRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterResRank2_Tag> PHAL_ScatterResRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank2_Adjoint_Tag> PHAL_ScatterJacRank2_Adjoint_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, PHAL_ScatterJacRank2_Tag> PHAL_ScatterJacRank2_Policy;

#endif
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
  std::vector<double> compositeTetLocalMassRow(const int row) 
    {std::vector<double> vec(10); return vec;}; 

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

// **************************************************************
// Distributed parameter derivative
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
protected:
  const std::size_t numFields;
  std::vector<double> compositeTetLocalMassRow(const int row) 
    {std::vector<double> vec(10); return vec;}; 

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};

template<typename Traits>
class ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  ScatterResidualWithExtrudedParams(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)  :
                    ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl) {
    extruded_params_levels = p.get< Teuchos::RCP<std::map<std::string, int> > >("Extruded Params Levels");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::postRegistrationSetup(d,vm);
  }
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels;
};

// **************************************************************
}

#endif
