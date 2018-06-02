//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace WAFERLG {

  //**********************************************************************

  template<typename EvalT, typename Traits>
  Energy_Dot<EvalT, Traits>::
  Energy_Dot(const Teuchos::ParameterList& p,
	     const Teuchos::RCP<Albany::Layouts>& dl) :
    T_              (p.get<std::string>("Temperature Name"),
		     dl->qp_scalar),
    T_dot_          (p.get<std::string>("Temperature Time Derivative Name"),
		     dl->qp_scalar),
    time_           (p.get<std::string>("Time Name"),
		     dl->workset_scalar),
    psi1_           (p.get<std::string>("Psi1 Name"),
		     dl->qp_scalar),
    psi2_           (p.get<std::string>("Psi2 Name"),
		     dl->qp_scalar),
    phi1_           (p.get<std::string>("Phi1 Name"),
		     dl->qp_scalar),
    phi2_           (p.get<std::string>("Phi2 Name"),
		     dl->qp_scalar),
    phi1_dot_       (p.get<std::string>("Phi1 Dot Name"),
		     dl->qp_scalar),
    phi2_dot_       (p.get<std::string>("Phi2 Dot Name"),
		     dl->qp_scalar),
    psi1_dot_       (p.get<std::string>("Psi1 Dot Name"),
                     dl->qp_scalar),
    psi2_dot_       (p.get<std::string>("Psi2 Dot Name"),
                     dl->qp_scalar),
    rho_Cp_         (p.get<std::string>("rho_Cp Name"),
		     dl->qp_scalar),
    deltaTime_      (p.get<std::string>("Delta Time Name"),
		     dl->workset_scalar),
    energyDot_      (p.get<std::string>("Energy Rate Name"),
		     dl->qp_scalar)
  {
        
    // dependent field
    this->addDependentField(T_);
    this->addDependentField(T_dot_);
    this->addDependentField(phi1_);
    this->addDependentField(phi2_);
    this->addDependentField(psi1_);
    this->addDependentField(psi2_);
    this->addDependentField(rho_Cp_);
    this->addDependentField(time_);
    this->addDependentField(deltaTime_);

    // evaluated field
    this->addEvaluatedField(energyDot_);
    this->addEvaluatedField(phi1_dot_);
    this->addEvaluatedField(phi2_dot_);
    this->addEvaluatedField(psi1_dot_);
    this->addEvaluatedField(psi2_dot_);

    std::vector<PHX::Device::size_type> dims;
    Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
    scalar_dl->dimensions(dims);
    workset_size_ = dims[0];
    num_qps_ = dims[1];

    // get temperature old variable name
    Temperature_Name_ = p.get<std::string>("Temperature Name") + "_old";
    // Get phi old variable name
    Phi1_old_name_ =  p.get<std::string>("Phi1 Name") + "_old";
    Phi2_old_name_ =  p.get<std::string>("Phi2 Name") + "_old";
    // Get psi old variable name
    Psi1_old_name_ =  p.get<std::string>("Psi1 Name") + "_old";
    Psi2_old_name_ =  p.get<std::string>("Psi2 Name") + "_old";


    // Only verify Change Phase Parameter list because initial Phi already
    // verified inside Phi evaluator (I hope so)
    Teuchos::ParameterList* cond_list =
      p.get<Teuchos::ParameterList*>("Phase Change Parameter List");

    Teuchos::RCP<const Teuchos::ParameterList> reflist =
      this->getValidEnergy_DotParameters();

    cond_list->validateParameters(*reflist, 0,
				  Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);


    // Get volumetric heat capacity
    Cl_ = cond_list->get<double>("Volumetric Heat Capacity Liquid Value", 5.95e6);
    // Get latent heat value
    L_ = cond_list->get<double>("Latent Heat Value", 2.18e9);
    //Latent Heat of Vapourization
    Lv_ = cond_list->get<double>("Latent Heat of Vapourization Value", 2.18e9);
    //Volumetric heat capacity of Vapour
    Cv_ = cond_list->get<double>("Volumetric Heat Capacity Vapour Value", 5.95e6);


    // later we need to verify that user have input proper values. Not done
    // now! But do it soon!


    cond_list = p.get<Teuchos::ParameterList*>("Initial Phi1 Parameter List");
    // Get melting temperature
    Tm_ = cond_list->get("Melting Temperature Value", 1700.0);
    // Get delta temperature value
    Tc_ = cond_list->get("delta Temperature Value", 50.0); 
        
    Temperature_Name_ = p.get<std::string>("Temperature Name") + "_old";
		
				
    cond_list = p.get<Teuchos::ParameterList*>("Initial Phi2 Parameter List");
    // Get Vapourization temperature
    Tv_ = cond_list->get("Vapourization Temperature Value", 3300.0);

    // Get the Volumetic Heat Capacity at the dense state of the material. 
    cond_list = p.get<Teuchos::ParameterList*>("Volumetric Heat Capacity Dense Parameter List");
    Cd = cond_list->get("Value",4.25e6);


    this->setName("Energy_Dot" + PHX::typeAsString<EvalT>());

  }

  //**********************************************************************

  template<typename EvalT, typename Traits>
  void Energy_Dot<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm) {
    this->utils.setFieldData(T_, fm);
    this->utils.setFieldData(T_dot_, fm);
    this->utils.setFieldData(time_, fm);
    this->utils.setFieldData(deltaTime_, fm);
    this->utils.setFieldData(phi1_, fm);
    this->utils.setFieldData(phi2_, fm);
    this->utils.setFieldData(phi1_dot_, fm);
    this->utils.setFieldData(phi2_dot_, fm);
    this->utils.setFieldData(psi1_dot_,fm);
    this->utils.setFieldData(psi2_dot_,fm);
    this->utils.setFieldData(psi1_, fm);
    this->utils.setFieldData(psi2_, fm);
    this->utils.setFieldData(rho_Cp_, fm);
    this->utils.setFieldData(energyDot_, fm);
  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void Energy_Dot<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    //current time
    // time step
    ScalarT dt = deltaTime_(0);

    typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

    if (dt == 0.0) dt = 1.0e-15;
    //grab old temperature
    Albany::MDArray T_old = (*workset.stateArrayPtr)[Temperature_Name_];

    // grab old value of phi
    Albany::MDArray phi1_old = (*workset.stateArrayPtr)[Phi1_old_name_];
    Albany::MDArray phi2_old = (*workset.stateArrayPtr)[Phi2_old_name_];

    // grab old value of psi
    Albany::MDArray psi1_old = (*workset.stateArrayPtr)[Psi1_old_name_];
    Albany::MDArray psi2_old = (*workset.stateArrayPtr)[Psi2_old_name_];

    // Compute Temp rate

    // temporal variables
    // store value of phi at Gauss point, i.e., phi1 = phi1_(cell,qp)
    ScalarT phi1;
    ScalarT phi2;
    // Store value of volumetric heat capacity at Gauss point, i.e., Cs = Rho_Cp_(cell,qp)
    ScalarT Cs;
    // Volumetric heat capacity of solid. For now same as powder.
    //  ScalarT Cd;
    // Variable used to store dp/dphi1 = 30*phi1^2*(1-2*phi1+phi1^2)
    ScalarT dpdphi1;
    ScalarT dgdphi2;
    // Variable used to store time derivative of phi
    //ScalarT phi1_dot;
    // Variable used to compute p = phi^3 * (10-15*phi+6*phi^2)
    ScalarT p;
    ScalarT g;



    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
      {
	for (std::size_t qp = 0; qp < num_qps_; ++qp)
	  {

	    // compute dT/dt using finite difference
	    T_dot_(cell, qp) = (T_(cell, qp) - T_old(cell, qp)) / dt;

	    phi1 = phi1_(cell, qp);
	    dpdphi1 = 30.0 * phi1 * phi1 * (1.0 - 2.0 * phi1 + phi1 * phi1);
	    phi2 = phi2_(cell, qp);
	    dgdphi2 = 30.0 * phi2 * phi2 * (1.0 - 2.0 * phi2 + phi2 * phi2);

	    phi1_dot_(cell,qp) = ( phi1_(cell,qp) - phi1_old(cell,qp) ) / dt;
	    phi2_dot_(cell,qp) = ( phi2_(cell,qp) - phi2_old(cell,qp) ) / dt;

	    Cs = rho_Cp_(cell, qp);

	    p = phi1 * phi1 * phi1 * (10.0 - 15.0 * phi1 + 6.0 * phi1 * phi1);
	    g = phi2 * phi2 * phi2 * (10.0 - 15.0 * phi2 + 6.0 * phi2 * phi2);

	    energyDot_(cell, qp) = (Cs + p * (Cl_ - Cs)) * T_dot_(cell, qp) +
	      dpdphi1 * (L_ + (Cl_ - Cs) * (T_(cell, qp) - Tm_)) * phi1_dot_(cell,qp) +
	      dgdphi2 * (Lv_ + Cv_ * (T_(cell, qp) - Tv_)) * phi2_dot_(cell,qp) + g * Cv_ * T_dot_(cell, qp);
		
	  }
      }


    
  }


  //**********************************************************************

  //**********************************************************************

  template<typename EvalT, typename Traits>
  Teuchos::RCP<const Teuchos::ParameterList>
  Energy_Dot<EvalT, Traits>::
  getValidEnergy_DotParameters() const {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid Energy Dot Params"));

    valid_pl->set<double>("Volumetric Heat Capacity Liquid Value", 5.95e6);
    valid_pl->set<double>("Latent Heat Value", 2.18e9);
    valid_pl->set<double>("Latent Heat of Vapourization Value", 2.18e9);
    valid_pl->set<double>("Volumetric Heat Capacity Vapour Value", 5.95e6);

    return valid_pl;
  }

  //**********************************************************************

}
