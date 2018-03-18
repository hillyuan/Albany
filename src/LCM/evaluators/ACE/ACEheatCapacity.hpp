//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(ACEheatCapacity_hpp)
#define ACEheatCapacity_hpp

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"

namespace LCM {
///
/// Evaluates heat capacity at integration points
///
template <typename EvalT, typename Traits>
class ACEheatCapacity : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
 public:
  using ScalarT = typename EvalT::ScalarT;

  ACEheatCapacity(Teuchos::ParameterList& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  /// Calculates mixture model heat capacity
  void
  evaluateFields(typename Traits::EvalData workset);

  /// Gets the intrinsic heat capacity values
  ScalarT&
  getValue(const std::string& n);

 private:
  int num_qps_{0};
  int num_dims_{0};

  // contains the mixture model heat capacity value
  PHX::MDField<ScalarT, Cell, QuadPoint> heat_capacity_;

  // contains the intrinsic heat capacity values for ice, water, sediment
  // these values are constant
  ScalarT cp_ice_{0.0};
  ScalarT cp_wat_{0.0};
  ScalarT cp_sed_{0.0};

};
}  // namespace LCM

#endif  // ACEheatCapacity_hpp
