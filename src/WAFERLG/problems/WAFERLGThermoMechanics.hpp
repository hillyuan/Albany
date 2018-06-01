//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef WAFERLGTHERMOMECHANICS_HPP
#define WAFERLGTHERMOMECHANICS_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TestForException.hpp"

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_MaterialDatabase.hpp"

namespace Albany {

  ///
  /// \brief Definition for the Phase problem
  ///
  class WAFERLGThermoMechanics : public AbstractProblem
  {
  public:
 
    WAFERLGThermoMechanics(const Teuchos::RCP<Teuchos::ParameterList>& params,
			   const Teuchos::RCP<ParamLib>& param_lib,
			   const int num_dims,
			   Teuchos::RCP<const Teuchos::Comm<int> >& commT);

    ~WAFERLGThermoMechanics();

    virtual 
    int spatialDimension() const { return num_dims_; }

    virtual 
    bool useSDBCs() const {return false; }

    virtual
    void buildProblem(
		      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
		      StateManager& stateMgr);

    virtual 
    Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
		    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
		    const Albany::MeshSpecsStruct& meshSpecs,
		    Albany::StateManager& stateMgr,
		    Albany::FieldManagerChoice fmchoice,
		    const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    Teuchos::RCP<const Teuchos::ParameterList> 
    getValidProblemParameters() const;
  

  

  private:

    WAFERLGThermoMechanics(const WAFERLGThermoMechanics&);
    
    WAFERLGThermoMechanics& operator=(const WAFERLGThermoMechanics&);

  public:

    template <typename EvalT> 
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
			PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
			const Albany::MeshSpecsStruct& meshSpecs,
			Albany::StateManager& stateMgr,
			Albany::FieldManagerChoice fmchoice,
			const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(
				      const std::vector<std::string>& nodeSetIDs);
    
    void constructNeumannEvaluators(
				    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  protected:

    int num_dims_;

    Teuchos::RCP<Albany::MaterialDatabase> material_db_;
  

    Teuchos::RCP<Albany::Layouts> dl_;

  };

}

//******************************************************************************

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_SaveStateField.hpp"

#include "rho_Cp.hpp"
#include "Phi1.hpp"
#include "Psi1.hpp"
#include "Phi2.hpp"
#include "Psi2.hpp"

#include "ThCond.hpp"
#include "Phase_Source.hpp" 
#include "Laser_Source.hpp"
#include "Phase_Residual.hpp"
#include "Energy_Dot.hpp"
#include "WAFERLG_Time.hpp"
#include "Energy.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::WAFERLGThermoMechanics::constructEvaluators(
						    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
						    const Albany::MeshSpecsStruct& meshSpecs,
						    Albany::StateManager& stateMgr,
						    Albany::FieldManagerChoice fieldManagerChoice,
						    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;
  using PHAL::AlbanyTraits;

  // Collect problem-specific response parameters
  Teuchos::RCP<Teuchos::ParameterList> pFromProb = Teuchos::rcp(
								new Teuchos::ParameterList("Response Parameters from Problem"));

  // // Problem is transient
  // TEUCHOS_TEST_FOR_EXCEPTION(
  //     number_of_time_deriv != 1,
  //     std::logic_error,
  //     "WAFERLGThermoMechanics must be defined as a transient calculation.");

  const CellTopologyData* const elem_top = &meshSpecs.ctd;

  std::string eb_name = meshSpecs.ebName;
  // check name of element blocks. Must be Powder and Solid
  // M. Juha: I am forcing this here because I need to pass the
  // properties of both materials to ThermalCond evaluator. Anyway,
  // for this analysis we will always have powder and solid (substrate) as
  // materials. Please, if you find a better way to do it, then change it!
  // check element block names
  if ((eb_name != "Powder_Region") && (eb_name != "Solid_Region")){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"*** Element block names must be Powder_Region or Solid_Region ***\n")
      }
  
  // check names of material names
  std::string material_name;
  material_name = material_db_->getElementBlockParam<std::string>(eb_name,"material");
  if ( (material_name != "Solid") && (material_name != "Powder") ){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"*** Material name must be Powder or Solid ***\n")
      }
 
  RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
    intrepid_basis = Albany::getIntrepid2Basis(*elem_top);

  RCP<shards::CellTopology> elem_type = 
    rcp(new shards::CellTopology (elem_top));

  Intrepid2::DefaultCubatureFactory cubFactory;

  RCP <Intrepid2::Cubature<PHX::Device> > elem_cubature = 
    cubFactory.create<PHX::Device, RealType, RealType>(*elem_type, meshSpecs.cubatureDegree);

  const int workset_size = meshSpecs.worksetSize;
  const int num_vertices = elem_type->getNodeCount();
  const int num_nodes = intrepid_basis->getCardinality();
  const int num_qps = elem_cubature->getNumPoints();

  *out << "Field Dimensions: Workset=" << workset_size 
       << ", Vertices= "               << num_vertices
       << ", Nodes= "                  << num_nodes
       << ", QuadPts= "                << num_qps
       << ", Dim= "                    << num_dims_ 
       << std::endl;

  dl_ = rcp(new Albany::Layouts(
				workset_size,num_vertices,num_nodes,num_qps,num_dims_));

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);

  Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;

  Teuchos::ArrayRCP<string> dof_names(1);
  Teuchos::ArrayRCP<string> dof_names_dot(1);
  Teuchos::ArrayRCP<string> resid_names(1);

  dof_names[0] = "Temperature";
  dof_names_dot[0] = "Temperature_dot";
  resid_names[0] = "Temperature Residual";

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator(
						false,dof_names,dof_names_dot));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(
						 false,resid_names));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFInterpolationEvaluator(dof_names_dot[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructMapToPhysicalFrameEvaluator(
						    elem_type,elem_cubature));

  fm0.template registerEvaluator<EvalT>
    (evalUtils.constructComputeBasisFunctionsEvaluator(
						       elem_type,intrepid_basis,elem_cubature));

  { // Time
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Time"));
   
    // Input
    p->set<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout",
					  dl_->workset_scalar);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    // Output
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
   
    // Register evaluator
    ev = Teuchos::rcp(new WAFERLG::Time<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Register state variable
    p = stateMgr.registerStateVariable("Time", dl_->workset_scalar,
				       dl_->dummy, eb_name, "scalar", 0.0, true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  
  { // Temperature
     
    double temp(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Initial Temperature")) {
      temp = material_db_->
	getElementBlockParam<double>(eb_name, "Initial Temperature");
    }
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
							  new Teuchos::ParameterList("Save Temperature"));
    p = stateMgr.registerStateVariable("Temperature",
				       dl_->qp_scalar,
				       dl_->dummy,
				       eb_name,
				       "scalar",
				       temp,
				       true,
				       false);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
     
  }

  { //Phi1
    Teuchos::RCP<ParameterList> p = rcp(new ParameterList("Phi1 parameters"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Initial Phi1");

    // Input
    p->set<string>("Temperature Name","Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Phi1 Name","Phi1");
    p->set<string>("Psi1 Name","Psi1");

    //  //Need these to compute responses later:
    RealType Tm = param_list.get<RealType>("Melting Temperature Value");    // Melting temperature
    pFromProb->set<RealType>("Melting Temperature Value",Tm);

    
    ev = rcp(new WAFERLG::Phi1<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
    
    p = stateMgr.registerStateVariable("Phi1", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", 0.0, true);
    
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev); 

  }

  { //Psi1
    RCP<ParameterList> p = rcp(new ParameterList("Psi1 parameters"));
    
    double psi1_initial(0.0);
    if (material_db_->isElementBlockSublist(eb_name, "Initial Psi1")) 
      {
	Teuchos::ParameterList& param = 
	  material_db_->getElementBlockSublist(eb_name, "Initial Psi1"); 
        psi1_initial = param.get<double>("Psi1");
      }
    
    Teuchos::ParameterList& param_list = 
      material_db_->getElementBlockSublist(eb_name, "Initial Psi1"); 

    // Input
    p->set<string>("Phi1 Name","Phi1");
    p->set<string>("Temperature Name","Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list); 

    //Output
    p->set<string>("Psi1 Name","Psi1");

    ev = rcp(new WAFERLG::Psi1<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable("Psi1", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", psi1_initial, true);
   

    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev); 
  }

  { //Phi2
    Teuchos::RCP<ParameterList> p = rcp(new ParameterList("Phi2 parameters"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Initial Phi2");

    // Input
    p->set<string>("Temperature Name","Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Phi2 Name","Phi2");
    p->set<string>("Psi2 Name","Psi2");

    //  //Need these to compute responses later:
    RealType Tv = param_list.get<RealType>("Vapourization Temperature Value");    // Vapourization temperature
    pFromProb->set<RealType>("Vapourization Temperature Value",Tv);

    
    ev = rcp(new WAFERLG::Phi2<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
    
    p = stateMgr.registerStateVariable("Phi2", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", 0.0, true);
    
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev); 

  }

  { //Psi2
    RCP<ParameterList> p = rcp(new ParameterList("Psi2 parameters"));
    
    double psi2_initial(0.0);
    if (material_db_->isElementBlockSublist(eb_name, "Initial Psi2")) 
      {
	Teuchos::ParameterList& param = 
	  material_db_->getElementBlockSublist(eb_name, "Initial Psi2"); 
        psi2_initial = param.get<double>("Psi2");
      }
    
    Teuchos::ParameterList& param_list = 
      material_db_->getElementBlockSublist(eb_name, "Initial Psi2"); 

    // Input
    p->set<string>("Phi2 Name","Phi2");
    p->set<string>("Temperature Name","Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list); 

    //Output
    p->set<string>("Psi2 Name","Psi2");

    ev = rcp(new WAFERLG::Psi2<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    p = stateMgr.registerStateVariable("Psi2", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", psi2_initial, true);
   

    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev); 
  }
  
  
  


  { // Thermal Conductivity
    RCP<ParameterList> p = rcp(new ParameterList("Thermal Conductivity"));
    
   
    // get data for powder
    Teuchos::ParameterList& param_list_powder =
      material_db_->getElementBlockSublist("Powder_Region", "Thermal Conductivity");  
    
    // get data for substrate (solid))
    Teuchos::ParameterList& param_list_solid =
      material_db_->getElementBlockSublist("Solid_Region", "Thermal Conductivity");  

    //Input
    p->set<string>("Coordinate Name","Coord Vec");
    p->set<Teuchos::ParameterList*>("Powder Parameter List", &param_list_powder);
    p->set<Teuchos::ParameterList*>("Solid Parameter List", &param_list_solid);

    //Output
    p->set<string>("Thermal Conductivity Name", "k");
    p->set<string>("Psi1 Name", "Psi1");

    ev = rcp(new WAFERLG::ThCond<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // rho_Cp
    RCP<ParameterList> p = rcp(new ParameterList("Specific Heat"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "rho_Cp");    

    //Input
    p->set<string>("Coordinate Name","Coord Vec");

    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);
    



    //Output
    p->set<string>("rho_Cp Name", "rho_Cp");

    ev = rcp(new WAFERLG::rho_Cp<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  

  
  { // Source Function
    RCP<ParameterList> p = rcp(new ParameterList("Source Function"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Source");

    //Input
    p->set<string>("Coordinate Name","Coord Vec");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Source Name", "Source");
    
    ev = rcp(new WAFERLG::Phase_Source<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
	
    p = stateMgr.registerStateVariable("Source", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", 0.0, true);
   

    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev); 

  }
  
  { // Laser Source Function
    RCP<ParameterList> p = rcp(new ParameterList("Laser Source Function"));

    Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, "Laser Source"); 

    //Input
    p->set<string>("Coordinate Name","Coord Vec");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    //Output
    p->set<string>("Laser Source Name", "Laser Source");
    
    ev = rcp(new WAFERLG::Laser_Source<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
	
    p = stateMgr.registerStateVariable("Laser Source", dl_->qp_scalar,
				       dl_->dummy, eb_name, "scalar", 0.0, true);
   

    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }  
  
  
  { // Energy dot
    RCP<ParameterList> p = rcp(new ParameterList("Energy Rate Params"));

    // take phase change parameter list
    Teuchos::ParameterList& param_list_phase =
      material_db_->getElementBlockSublist(eb_name, "Phase Change Properties");
	
    //Input
    p->set<string>("Temperature Name","Temperature");
    p->set<string>("Temperature Time Derivative Name","Temperature_dot");
    p->set<string>("Phi1 Name","Phi1");
    p->set<string>("Phi2 Name","Phi2");
    p->set<string>("Phi1 Dot Name","Phi1_dot");
    p->set<string>("Phi2 Dot Name","Phi2_dot");
    p->set<string>("Psi1 Name","Psi1");
    p->set<string>("Psi2 Name","Psi2");
    p->set<string>("Psi1 Dot Name","Psi1_dot");
    p->set<string>("Psi2 Dot Name","Psi2_dot");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");
    p->set<string>("rho_Cp Name", "rho_Cp");
    p->set<Teuchos::ParameterList*>("Phase Change Parameter List", &param_list_phase);


    // take initial Phi1 parameter list
    Teuchos::ParameterList& param_list_phi1 =
      material_db_->getElementBlockSublist(eb_name, "Initial Phi1");
    p->set<Teuchos::ParameterList*>("Initial Phi1 Parameter List", &param_list_phi1);
	
    // take initial Phi2 parameter list
    Teuchos::ParameterList& param_list_phi2 =
      material_db_->getElementBlockSublist(eb_name, "Initial Phi2");
    p->set<Teuchos::ParameterList*>("Initial Phi2 Parameter List", &param_list_phi2);

    // take initial Psi1 parameter list
    Teuchos::ParameterList& param_list_psi1 =
      material_db_->getElementBlockSublist(eb_name, "Initial Psi1");
    p->set<Teuchos::ParameterList*>("Initial Psi1 Parameter List", &param_list_psi1);
	
    // take initial Psi2 parameter list
    Teuchos::ParameterList& param_list_psi2 =
      material_db_->getElementBlockSublist(eb_name, "Initial Psi2");
    p->set<Teuchos::ParameterList*>("Initial Psi2 Parameter List", &param_list_psi2);



    // take rho_Cp parameter list
    Teuchos::ParameterList& param_list_rho_Cp =
      material_db_->getElementBlockSublist(eb_name, "rho_Cp");
    p->set<Teuchos::ParameterList*>("Volumetric Heat Capacity Dense Parameter List", &param_list_rho_Cp);

    //Output
    p->set<string>("Energy Rate Name", "Energy Rate");

    //  //Need these to compute responses later:
    RealType Cl = param_list_phase.get<RealType>("Volumetric Heat Capacity Liquid Value");        // Volumetric heat capacity in liquid
    RealType L  = param_list_phase.get<RealType>("Latent Heat Value");    // Latent heat of fusion/melting
    RealType Cv = param_list_phase.get<RealType>("Volumetric Heat Capacity Vapour Value");
    RealType Lv = param_list_phase.get<RealType>("Latent Heat of Vapourization Value");


    pFromProb->set<RealType>("Volumetric Heat Capacity Liquid Value",Cl);
    pFromProb->set<RealType>("Latent Heat Value",L);

    ev = rcp(new WAFERLG::Energy_Dot<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  

  { // Phase Residual
    RCP<ParameterList> p = rcp(new ParameterList("u Resid"));

    //Input
    p->set<string>("Weighted BF Name","wBF");
    p->set<string>("Weighted Gradient BF Name","wGrad BF");
    p->set<string>("Temperature Name","Temperature");
    p->set<string>("Temperature Gradient Name","Temperature Gradient");
    p->set<string>("Temperature Time Derivative Name","Temperature_dot");
    p->set<string>("Thermal Conductivity Name","k");
    p->set<string>("rho_Cp Name","rho_Cp");
    p->set<string>("Source Name","Source");
    p->set<string>("Laser Source Name","Laser Source");
    p->set<string>("Phi1 Name","Phi1");
    p->set<string>("Phi2 Name","Phi2");
    p->set<string>("Psi1 Name","Psi1");
    p->set<string>("Psi2 Name","Psi2");

    p->set<string>("Energy Rate Name", "Energy Rate");
    p->set<string>("Time Name","Time");
    p->set<string>("Delta Time Name","Delta Time");
    

    


    //Output
    p->set<string>("Residual Name", "Temperature Residual");

    ev = rcp(new WAFERLG::Phase_Residual<EvalT,AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
    fm0.requireField<EvalT>(res_tag);
    return res_tag.clone();
  }

  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
    return respUtils.constructResponses(fm0, *responseList, pFromProb, stateMgr);
  }

  return Teuchos::null;
}

#endif
