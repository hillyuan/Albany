//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LASER_SOURCE_HPP
#define LASER_SOURCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_Layouts.hpp"

#include "Laser.hpp"

namespace WAFERLG {
  ///
  /// \brief Laser Source
  ///
  /// This evaluator computes the moving laser source as a function of space and time to a 
  /// Phase-change/heat equation problem
  ///
  template<typename EvalT, typename Traits>
  class Laser_Source : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {

  public:

    Laser_Source(Teuchos::ParameterList& p,
		 const Teuchos::RCP<Albany::Layouts>& dl);

    void 
    postRegistrationSetup(typename Traits::SetupData d,
			  PHX::FieldManager<Traits>& vm);

    void 
    evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    ScalarT extinction_coeff;
    ScalarT laser_wavelength; 
    ScalarT laser_pulse_frequency;
    ScalarT laser_beam_radius;
    ScalarT average_laser_power;
    ScalarT instantaneous_laser_power;
    ScalarT reflectivity;
  
    void init_constant_extinction_coeff(ScalarT value_extinction_coeff, Teuchos::ParameterList& p);
    void init_constant_laser_pulse_frequency(ScalarT value_particle_dia, Teuchos::ParameterList& p);
    void init_constant_laser_beam_radius(ScalarT value_laser_beam_radius, Teuchos::ParameterList& p);
    void init_constant_average_laser_power(ScalarT value_average_laser_power, Teuchos::ParameterList& p);
    void init_constant_reflectivity(ScalarT value_reflectivity, Teuchos::ParameterList& p);

    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_;
    PHX::MDField<ScalarT,Cell,QuadPoint> laser_source_;
    PHX::MDField<ScalarT,Dummy> time;
    PHX::MDField<ScalarT,Dummy> deltaTime;

    unsigned int num_qps_;
    unsigned int num_dims_;
    unsigned int num_nodes_;
    unsigned int workset_size_;

    Laser LaserData_;

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidLaser_SourceParameters() const;
  };
}

#endif
