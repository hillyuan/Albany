//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GMSH_STK_MESH_STRUCT_HPP
#define ALBANY_GMSH_STK_MESH_STRUCT_HPP

#include "Albany_GenericSTKMeshStruct.hpp"
//#include <Ionit_Initializer.h>

namespace Albany
{
	
struct node_struct
{
	int id;
	double x,y,z;
};

class FstrSTKMeshStruct : public GenericSTKMeshStruct
{
  public:

  FstrSTKMeshStruct (const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<const Teuchos_Comm>& commT);

  ~FstrSTKMeshStruct();

  void setFieldAndBulkData (const Teuchos::RCP<const Teuchos_Comm>& commT,
                            const Teuchos::RCP<Teuchos::ParameterList>& params,
                            const unsigned int neq_,
                            const AbstractFieldContainer::FieldContainerRequirements& req,
                            const Teuchos::RCP<Albany::StateInfoStruct>& sis,
                            const unsigned int worksetSize,
                            const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis = {},
                            const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req = {});

  //! Flag if solution has a restart values -- used in Init Cond
  bool hasRestartSolution() const {return false; }

  //! If restarting, convenience function to return restart data time
  double restartDataTime() const {return -1.0; }

  private:

  Teuchos::RCP<const Teuchos::ParameterList> getValidDiscretizationParameters() const;

 // void loadMesh (const std::string& fname);
  void loadData( std::ifstream& );

  int NumElemNodes; // Number of nodes per element (e.g. 3 for Triangles)
  int NumSideNodes; // Number of nodes per side (e.g. 2 for a Line)
  int NumNodes; //number of nodes
  int NumElems; //number of elements
  int NumSides; //number of sides

  std::map<int,std::string> bdTagToNodeSetName;
  std::map<int,std::string> bdTagToSideSetName;
  std::vector<node_struct> points;

  // Only some will be used, but it's easier to have different pointers
  int** hexas;
  int** tetra;
  int** quads;
  int** trias;
  int** lines;

  // These pointers will be set equal to two of the previous group, depending on dimension
  // NOTE: do not call delete on these pointers! Delete the previous ones only!
  int** elems;
  int** sides;
  
  std::vector<std::string> nsNames;
  std::vector<std::string> ssNames;
  std::vector<std::string> esNames;
  
  /**
   * This function parses a label of the form foo=bar from a
   * comma-delimited line of the form
   * ..., foo=bar, ...
   * The input to the function in this case would be foo, the
   * output would be bar
   */
  std::string parse_label(std::string line, std::string label_name) const;
  
  /**
   * Any of the various sections can start with some number of lines
   * of comments, which start with "**".  This function discards
   * any lines of comments that it finds from the stream, leaving
   * trailing data intact.
   */
  void process_and_discard_comments(std::ifstream&);
  
  /**
   * \returns \p true if the input string is a generated elset or nset,
   * false otherwise.
   *
   * The input string is assumed to already be in all upper
   * case. Generated nsets are assumed to have the following format:
   * *Nset, nset=Set-1, generate
   */
  bool detect_generated_set(std::string upper) const;
};

} // Namespace Albany

#endif // ALBANY_FSTR_STK_MESH_STRUCT_HPP
