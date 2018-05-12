//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <iostream>

#include "Albany_FstrSTKMeshStruct.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <Shards_BasicTopologies.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <Albany_STKNodeSharing.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

//#include <stk_mesh/fem/FEMHelpers.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "Albany_Utils.hpp"

Albany::FstrSTKMeshStruct::FstrSTKMeshStruct (const Teuchos::RCP<Teuchos::ParameterList>& params,
                                              const Teuchos::RCP<const Teuchos_Comm>& commT) :
  GenericSTKMeshStruct (params, Teuchos::null)
{
  std::string fname = params->get("FSTR Input Mesh File Name", "fstr.msh");
  
  numDim = 3;   // we consider 3-dimensional case only
  std::vector<std::string> entity_rank_names = stk::mesh::entity_rank_names();
  if(this->buildEMesh)
    entity_rank_names.push_back("FAMILY_TREE");
  metaData->initialize (this->numDim, entity_rank_names);

 // if (commT->getRank() == 0)
 // {
    std::ifstream ifile;
    ifile.open(fname.c_str());
    if (!ifile.is_open())
    {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Cannot open mesh file '" << fname << "'.\n");
    }
	
	loadData( ifile );
	
//	ifile.clear(); // clear bad state after eof
 //   ifile.seekg( 0 );
	
//	loadData( ifile );
//  }

  unsigned int k;
  for (k=0; k<nsNames.size(); ++k)
  {
	nsPartVec.insert( std::make_pair(nsNames[k], &metaData->declare_part(nsNames[k], stk::topology::NODE_RANK) ) );
  }
  int n_elset = esNames.size();
  for (k=0; k<n_elset ; ++k)
  {
    partVec.insert( std::make_pair(k,&metaData->declare_part(esNames[k], stk::topology::ELEMENT_RANK) ) );
	ebNameToIndex.insert( std::make_pair(esNames[k], k) );
  }
  for (k=0; k<ssNames.size(); ++k)
  {
	ssPartVec.insert( std::make_pair(ssNames[k], &metaData->declare_part(ssNames[k], metaData->side_rank()) ) );
  }
  
  const int cub      = params->get("Cubature Degree",3);
  int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
	
  this->allElementBlocksHaveSamePhysics=false;
  this->meshSpecs.resize(n_elset);
  for (int eb=0; eb<n_elset; eb++) {
    const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[eb]).getCellTopologyData();
	int worksetSize = this->computeWorksetSize(worksetSizeMax, NumElems);   // Numelems not known yet
    this->meshSpecs[eb] = Teuchos::rcp(new Albany::MeshSpecsStruct(
          ctd, numDim, cub, nsNames, ssNames, worksetSize, partVec[eb]->name(),
          this->ebNameToIndex, this->interleavedOrdering));
  }
  
  double NumElemNodesD = NumElemNodes;
  Teuchos::broadcast<LO,ST>(*commT, 0, &NumElemNodesD);

  params->validateParameters(*getValidDiscretizationParameters(), 0);
  
  Teuchos::broadcast<LO,LO>(*commT, 0, &NumElemNodes);
  switch (NumElemNodes)
  {
    case 3:
      stk::mesh::set_cell_topology<shards::Triangle<3> >(*partVec[0]);
  //    stk::mesh::set_cell_topology<shards::Line<2> >(*ssPartVec[ssn]);
      break;
    case 4:
      if (NumSideNodes==3)
      {
        stk::mesh::set_cell_topology<shards::Tetrahedron<4> >(*partVec[0]);
   //     stk::mesh::set_cell_topology<shards::Triangle<3> >(*ssPartVec[ssn]);
      }
      else
      {
        stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*partVec[0]);
    //    stk::mesh::set_cell_topology<shards::Line<2> >(*ssPartVec[ssn]);
      }
      break;
    case 8:
      stk::mesh::set_cell_topology<shards::Hexahedron<8> >(*partVec[0]);
    //  stk::mesh::set_cell_topology<shards::Quadrilateral<4> >(*ssPartVec[ssn]);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Invalid number of element nodes (you should have got an error before though).\n");
  }

  numDim = 2;
  //int worksetSizeMax = params->get<int>("Workset Size", DEFAULT_WORKSET_SIZE);
  //int worksetSize = this->computeWorksetSize(worksetSizeMax, NumElems);
  //const CellTopologyData& ctd = *metaData->get_cell_topology(*partVec[0]).getCellTopologyData();
  //cullSubsetParts(ssNames, ssPartVec);
  //this->meshSpecs[0] = Teuchos::rcp (
  //    new Albany::MeshSpecsStruct (ctd, numDim, cub, nsNames, ssNames,
     //                              worksetSize, partVec[0]->name(), ebNameToIndex,
   //                                this->interleavedOrdering));

  this->initializeSideSetMeshStructs(commT);
}

Albany::FstrSTKMeshStruct::~FstrSTKMeshStruct()
{
  for (int i(0); i<5; ++i)
    delete[] tetra[i];
  for (int i(0); i<5; ++i)
    delete[] trias[i];
  for (int i(0); i<9; ++i)
    delete[] hexas[i];
  for (int i(0); i<5; ++i)
    delete[] quads[i];
  for (int i(0); i<3; ++i)
    delete[] lines[i];

  delete[] tetra;
  delete[] trias;
  delete[] hexas;
  delete[] quads;
  delete[] lines;
}

void Albany::FstrSTKMeshStruct::setFieldAndBulkData(
    const Teuchos::RCP<const Teuchos_Comm>& commT,
    const Teuchos::RCP<Teuchos::ParameterList>& params,
    const unsigned int neq_,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const unsigned int worksetSize,
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req)
{
  this->SetupFieldData(commT, neq_, req, sis, worksetSize);

  metaData->commit();

  bulkData->modification_begin(); // Begin modifying the mesh

  // Only proc 0 has loaded the file
  if (commT->getRank()==0)
  {
    stk::mesh::PartVector singlePartVec(1);
    unsigned int ebNo = 0; //element block #???
    int sideID = 0;

    AbstractSTKFieldContainer::IntScalarFieldType* proc_rank_field = fieldContainer->getProcRankField();
    AbstractSTKFieldContainer::VectorFieldType* coordinates_field =  fieldContainer->getCoordinatesField();

    singlePartVec[0] = nsPartVec["Node"];

    for (int i = 0; i < NumNodes; i++)
    {
      stk::mesh::Entity node = bulkData->declare_entity(stk::topology::NODE_RANK, i + 1, singlePartVec);

      double* coord;
      coord = stk::mesh::field_data(*coordinates_field, node);
      coord[0] = points[i].x;
      coord[1] = points[i].x;
      if (numDim==3)
        coord[2] = points[i].x;
    }

    for (int i = 0; i < NumElems; i++)
    {
      singlePartVec[0] = partVec[ebNo];
      stk::mesh::Entity elem = bulkData->declare_entity(stk::topology::ELEMENT_RANK, i + 1, singlePartVec);

      for (int j = 0; j < NumElemNodes; j++)
      {
        stk::mesh::Entity node = bulkData->get_entity(stk::topology::NODE_RANK, elems[j][i]);
        bulkData->declare_relation(elem, node, j);
      }

      int* p_rank = stk::mesh::field_data(*proc_rank_field, elem);
      p_rank[0] = commT->getRank();
    }

    std::string partName;
    stk::mesh::PartVector nsPartVec_i(1), ssPartVec_i(2);
    ssPartVec_i[0] = ssPartVec["BoundarySide"]; // The whole boundary side
    for (int i = 0; i < NumSides; i++)
    {
      std::map<int,int> elm_count;
      partName = bdTagToNodeSetName[sides[NumSideNodes][i]];
      nsPartVec_i[0] = nsPartVec[partName];

      partName = bdTagToSideSetName[sides[NumSideNodes][i]];
      ssPartVec_i[1] = ssPartVec[partName];

      stk::mesh::Entity side = bulkData->declare_entity(metaData->side_rank(), i + 1, ssPartVec_i);
      for (int j=0; j<NumSideNodes; ++j)
      {
        stk::mesh::Entity node_j = bulkData->get_entity(stk::topology::NODE_RANK,sides[j][i]);
        bulkData->change_entity_parts (node_j,nsPartVec_i); // Add node to the boundary nodeset
        bulkData->declare_relation(side, node_j, j);

        int num_e = bulkData->num_elements(node_j);
        const stk::mesh::Entity* e = bulkData->begin_elements(node_j);
        for (int k(0); k<num_e; ++k)
        {
          ++elm_count[bulkData->identifier(e[k])];
        }
      }

      // We have to find out what element has this side as a side. We check the node connectivity
      // In particular, the element that is connected to all NumSideNodes nodes is the one.
      bool found = false;

      for (auto e : elm_count)
        if (e.second==NumSideNodes)
        {
          stk::mesh::Entity elem = bulkData->get_entity(stk::topology::ELEM_RANK, e.first);
          found = true;
          int num_sides = bulkData->num_sides(elem);
          bulkData->declare_relation(elem,side,num_sides);
          break;
        }

      TEUCHOS_TEST_FOR_EXCEPTION (found==false, std::logic_error, "Error! Cannot find element connected to side " << i+1 << ".\n");
    }

  }
  bulkData->modification_end();

#ifdef ALBANY_ZOLTAN
  // Gmsh is for sure using a serial mesh. We hard code it here, in case the user did not set it
  params->set<bool>("Use Serial Mesh", true);

  // Refine the mesh before starting the simulation if indicated
  uniformRefineMesh(commT);

  // Rebalance the mesh before starting the simulation if indicated
  rebalanceInitialMeshT(commT);
#endif

  // Loading required input fields from file
  this->loadRequiredInputFields (req,commT);

  // Finally, perform the setup of the (possible) side set meshes (including extraction if of type SideSetSTKMeshStruct)
  this->finalizeSideSetMeshStructs(commT, side_set_req, side_set_sis, worksetSize);

  fieldAndBulkDataSet = true;
}

Teuchos::RCP<const Teuchos::ParameterList> Albany::FstrSTKMeshStruct::getValidDiscretizationParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getValidGenericSTKParameters("Valid ASCII_DiscParams");
  validPL->set<std::string>("Gmsh Input Mesh File Name", "mesh.msh",
      "Name of the file containing the 2D mesh, with list of coordinates, elements' connectivity and boundary edges' connectivity");

  return validPL;
}

// -------------------------------- Read method ---------------------------- //
void Albany::FstrSTKMeshStruct::loadData(std::ifstream& f)
{
	int n_elset = 0;
	
    std::string s;
    while (true)
    {
      std::getline(f, s);
	  if (f)
      {
	    std::string upper(s);
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
		if (upper.find("*NODE") == static_cast<std::string::size_type>(0))
        {
            // Some sections that begin with *NODE are actually
            // "*NODE OUTPUT" sections which we want to skip.  I
            // have only seen this with a single space, but it would
            // probably be more robust to remove whitespace before
            // making this check.
            if (upper.find("*NODE OUTPUT") == static_cast<std::string::size_type>(0))
                continue;

            // Some *Node sections also specify an Nset name on the same line.
            // Look for one here.
            std::string nset_name = this->parse_label(s, "nset");
			if (nset_name == "")
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Unnamed nset encountered!\n");
		    nsNames.push_back(nset_name);

            // Process any lines of comments that may be present
            this->process_and_discard_comments(f);

            // Read a block of nodes
		   node_struct nd;
           char c;
           std::string line;
           while (f.peek() != '*' && f.peek() != EOF)
           {
             std::getline(f, line);
	         line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
	         std::stringstream ss(line);
	         ss >> nd.id >>c >> nd.x >>c  >> nd.y >> c >> nd.z;
	//  std::cout << entry.first <<" "<< entry.second[0] <<" "  << entry.second[1] 
        	  points.push_back(nd);
           }    
        }
		else if (upper.find("*ELEMENT") == static_cast<std::string::size_type>(0))
        {
		  // m = parseElems(upper, f, nodes);
	    }
	    else if (upper.find("*NSET") == static_cast<std::string::size_type>(0))
        {
	        std::string nset_name = this->parse_label(s, "nset");
            if (nset_name == "")
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Unnamed nset encountered!\n");
		    nsNames.push_back(nset_name);
	    }
	    else if (upper.find("*ESET") == static_cast<std::string::size_type>(0))
        {
	   	    std::string elset_name = this->parse_label(s, "elset");
            if (elset_name == "")
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Unnamed elset encountered!\n");
			n_elset++;
			nsNames.push_back(elset_name);
	    }
		else if (upper.find("*SURFACE,") == static_cast<std::string::size_type>(0))
        {
            std::string sideset_name = this->parse_label(s, "name");
			if (sideset_name == "")
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Error! Unnamed surface encountered!\n");
			// type of surface element may be nonhomogeneous!
			ssNames.push_back(sideset_name);
		}
	  } 
	
	  if (f.eof()) break;
    }
}

std::string Albany::FstrSTKMeshStruct::parse_label(std::string line, std::string label_name) const
{
  // Handle files which have weird line endings from e.g. windows.
  // You can check what kind of line endings you have with 'cat -vet'.
  // For example, some files may have two kinds of line endings like:
  //
  // 4997,^I496,^I532,^I487,^I948^M$
  //
  // and we don't want to deal with this when extracting a label, so
  // just remove all the space characters, which should include all
  // kinds of remaining newlines.  (I don't think Abaqus allows
  // whitespace in label names.)
  line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());

  // Do all string comparisons in upper-case
  std::string upper_line(line), upper_label_name(label_name);
  std::transform(upper_line.begin(), upper_line.end(), upper_line.begin(), ::toupper);
  std::transform(upper_label_name.begin(), upper_label_name.end(), upper_label_name.begin(), ::toupper);

  // Get index of start of "label="
  size_t label_index = upper_line.find(upper_label_name + "=");

  if (label_index != std::string::npos)
    {
      // Location of the first comma following "label="
      size_t comma_index = upper_line.find(",", label_index);

      // Construct iterators from which to build the sub-string.
      // Note: The +1 while initializing beg is to skip past the "=" which follows the label name
      std::string::iterator
        beg = line.begin() + label_name.size() + 1 + label_index,
        end = (comma_index == std::string::npos) ? line.end() : line.begin() + comma_index;

      return std::string(beg, end);
    }

  // The label index was not found, return the empty string
  return std::string("");
}

void Albany::FstrSTKMeshStruct::process_and_discard_comments(std::ifstream& _in)
{
  std::string dummy;
  while (true)
    {
      // We assume we are at the beginning of a line that may be
      // comments or may be data.  We need to only discard the line if
      // it begins with **, but we must avoid calling std::getline()
      // since there's no way to put that back.
      if (_in.peek() == '*')
        {
          // The first character was a star, so actually read it from the stream.
          _in.get();

          // Peek at the next character...
          if (_in.peek() == '*')
            {
              // OK, second character was star also, by definition this
              // line must be a comment!  Read the rest of the line and discard!
              std::getline(_in, dummy);
            }
          else
            {
              // The second character was _not_ a star, so put back the first star
              // we pulled out so that the line can be parsed correctly by somebody
              // else!
              _in.unget();

              // Finally, break out of the while loop, we are done parsing comments
              break;
            }
        }
      else
        {
          // First character was not *, so this line must be data! Break out of the
          // while loop!
          break;
        }
    }
}

bool Albany::FstrSTKMeshStruct::detect_generated_set(std::string upper) const
{
  // Avoid issues with weird line endings, spaces before commas, etc.
  upper.erase(std::remove_if(upper.begin(), upper.end(), isspace), upper.end());

  // Check each comma-separated value in "upper" to see if it is the generate flag.
  std::string cell;
  std::stringstream line_stream(upper);
  while (std::getline(line_stream, cell, ','))
    if (cell == "GENERATE")
      return true;

  return false;
}
