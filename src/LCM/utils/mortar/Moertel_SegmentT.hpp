//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOERTEL_SEGMENTT_HPP
#define MOERTEL_SEGMENTT_HPP

#include <ctime>
#include <iostream>
#include <map>
#include <vector>

#include "Moertel_FunctionT.hpp"

/*!
\brief MOERTEL: namespace of the Moertel package

The Moertel package depends on \ref Epetra, \ref EpetraExt, \ref Teuchos,
\ref Amesos, \ref ML and \ref AztecOO:<br>
Use at least the following lines in the configure of Trilinos:<br>
\code
--enable-moertel 
--enable-epetra 
--enable-epetraext
--enable-teuchos 
--enable-ml
--enable-aztecoo --enable-aztecoo-teuchos 
--enable-amesos
\endcode

*/

namespace MoertelT
{

MOERTEL_TEMPLATE_STATEMENT
class InterfaceT;

MOERTEL_TEMPLATE_STATEMENT
class NodeT;

/*!
\class Segment

\brief <b> A virtual class as a basis for different types of interface segments</b>

This class serves as a (not pure) virtual base class to several types of interface segments.

The \ref MOERTEL::Segment class supports the ostream& operator <<

\author Glen Hansen (gahanse@sandia.gov)

*/

// Helper class for segment functions
struct Segment_Traits { };

//template <size_t DIM,\
//          class SEGT = Segment_Traits,\
//          class FUNCT = SegFunction_Traits>
SEGMENT_TEMPLATE_STATEMENT
class SegmentT
{
public:
  
  /*!
  \brief Type of segment
         
   \param seg_none : default value
   \param seg_Linear1D : linear 1D segment with 2 nodes
   \param seg_BiLinearQuad : linear 2D triangle with 3 nodes
   \param seg_BiLinearTri : linear 2D quadrilateral with 4 nodes
  */

  // @{ \name Constructors and destructors

  /*!
  \brief Standard Constructor
  
  \param Id : A unique positive Segment id. Does not need to be continous among segments
  \param nnode : Number of nodes this segment is attached to
  \param nodeId : Pointer to vector length nnode holding unique positive 
                  node ids of nodes this segment is attached to
  \param outlevel : Level of output to stdout to be generated by this class (0-10)
  */
  SegmentT(int id, int nnode, int* nodeId, int outlevel);

  SegmentT(int id, const std::vector<int>& nodeId, int outlevel);
  
  /*!
  \brief Empty Constructor

  This constructor is used together with \ref Pack and \ref UnPack for
  communicating segments
  
  \param outlevel : Level of output to stdout to be generated by this class (0-10)
  */
  SegmentT(int outlevel);
  
  /*!
  \brief Copy Constructor

  Makes a deep copy of a Segment
  
  */
  SegmentT(MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & old);
  
  /*!
  \brief Destructor

  */
  virtual ~SegmentT();
  
  //@}

  // @{ \name Methods implemented by this class

  /*!
  \brief Return level of output to be generated by this class (0-10)

  */
  int OutLevel() { return outputlevel_; }
  
  /*!
  \brief Return unique id of this Segment

  */
  int Id() const { return Id_; }
  
  /*!
  \brief Return number of nodes attached to this Segment

  */
  int Nnode() const { return nodeId_.size(); }
  
  /*!
  \brief Return type of Segment

  */
//  MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::SegmentType Type() const { return stype_; }
  
  /*!
  \brief Return view of node ids of nodes attached to this Segment

  */
  const int* NodeIds() const { return &(nodeId_[0]); }
  
  /*!
  \brief Return pointer to vector of length \ref Nnode() of 
   pointers to Nodes attached to this Segment

  */
  MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** Nodes() { return &(nodeptr_[0]); }
  
  /*!
  \brief Return number of functions defined on this Segment

  */
  int Nfunctions() { return functions_.size(); }
  
  /*!
  \brief Return FunctionType of a function with the Id id

  \param id : function id to lookup the type for
  */
//  MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::FunctionType FunctionType(int id);
  
  /*!
  \brief Attach a function to this Segment
  
  Will attach a function to this Segment under the function Id id.
  Segment will not take ownership of func but will store a deep copy

  \param id : unique function id to store function
  \param func : Function to store in this Segment
  */
  bool SetFunction(int id, MoertelT::FunctionT<FUNCT>* func);

  /*!
  \brief Evaluate a function with a certain id
  
  Will evaluate the function with Id id at a given local coordinate

  \param id (in): unique function id 
  \param xi (in): Segment local coordinates where to evaluate the function
  \param val (out): Vector holding function values at xi on output. If NULL on input, 
                    function will not evaluate values.
  \param valdim (in): length of val
  \param deriv (out): Vector holding function derivatives at xi on output, 
                      should be of length 2*valdim in most cases. If NULL on input, 
                      function will not evaluate derivatives.
  */
  bool EvaluateFunction(int id, const double* xi, double* val, int valdim, double* deriv);
  
  /*!
  \brief Build normal at a node adjacent to this Segment
  
  \param nid : global unique node id
  */
  double* BuildNormalAtNode(int nid);
  
  /*!
  \brief Get pointers to Nodes attached to this Segment from the Interface this Segment resides on
  
  */
  bool GetPtrstoNodes(MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)& interface);


  /*!
  \brief Get pointers to Nodes attached to this Segment from a vector of Node pointers
  
  */
  bool GetPtrstoNodes(std::vector<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)*>& nodes);

  /*!
  \brief Print this Segment
  
  */
  virtual bool Print() const;
  
  /*!
  \brief Get segment-local node id from global node id nid
  
  */
  int GetLocalNodeId(int nid);

  //@}
  // @{ \name Pure virtual methods of this class

  /*!
  \brief Deep copy the derived class and return pointer to it
  
  */
//  virtual MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)* Clone() = 0;

  /*!
  \brief Pack some data from this class to an int vector of length size so
         it can be communicated using MPI
  
  */
  virtual int* Pack(int* size) = 0;
  
  /*!
  \brief Unpack some data an int vector and store data in this class
  
  */
  virtual bool UnPack(int* pack) = 0;
  
  /*!
  \brief Build an outward normal at segment coordinates xi
  
  */
  virtual double* BuildNormal(double* xi) = 0;
  
  /*!
  \brief Compute and return the area of this Segment
  
  */
  virtual double Area() = 0;

  /*!
  \brief Build the basis vectors and metric tensor at a given local coord in this segment
  
  */
  virtual double Metric(double* xi, double g[], double G[][3]) = 0;

  /*!
  \brief Get local coords of a node attached to this segment with \b local node Id lid 
  
  */
  virtual bool LocalCoordinatesOfNode(int lid, double* xi) = 0;

  //@}

protected:

  int                                       Id_;         // this segments unique id
  int                                       outputlevel_;
  std::vector<int>                               nodeId_;     // vector of unique node ids 
  std::vector<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)*>                    nodeptr_;    // vector with ptrs to nodes adj to me

  std::map< int,Teuchos::RCP<MoertelT::FunctionT<FUNCT> > > functions_;  // functions that live on this segment

};

// Template specializations

/*!
\class Segment_Linear1D

\brief <b> A class to define a 2-noded linear 1D Segment</b>

This class defines a 2-noded linear 1D interface surface segment.

<b>Important:</b><br>
Upon construction, the user must assure that the nodes attached to this segment are
given in counter-clockwise order:<br>
<pre>


      <--|       o Node 1
  domain |       | 
       --        | --> outward normal n to Seg 2
                 | 
                 | Seg 2
       Seg 1     |
    o------------o Node 0
   Node 0 |  Node 1
          | 
          | outward normal n to Seg 1
          | 
          v

</pre>
The reason for this is that the outward normal to the interface is implicitly defined by
the orientation of the segments. There is no way for the MOERTEL package to check the
orientation of the segments as they are passed in by the user and most obscure results
will be produced when the orientation is uncorrect!

*/


   struct Linear1DSeg {

     int* Pack(int* size);
     bool UnPack(int* pack);
     double* BuildNormal(double* xi);
     double Area();
     double Metric(double* xi, double g[], double G[][3]);
     bool LocalCoordinatesOfNode(int lid, double* xi);

   };

/*!
\class Segment_BiLinearTri

\brief <b> A class to define a 3-noded triangle 2D Segment</b>

This class defines a 3-noded linear 2D triangle interface surface segment.

<b>Important:</b><br>
Upon construction, the user must assure that the nodes attached to this segment are
given in counter-clockwise order such that the outward normal to the domain
points out from the screen:<br>
<pre>
                   Node 2
                     o
                   / |
                  /  |
                 /   |
                /    |                <-------| 
               /     |         domain surface | 
              /      |                   ------  
             /       |
            /        |
           /         |
          o----------o
        Node 0     Node 1

</pre>
The reason for this is that the outward normal to the interface is implicitly defined by
the orientation of the segments. There is no way for the MOERTEL package to check the
orientation of the segments as they are passed in by the user and most obscure results
will be produced when the orientation is uncorrect!

*/

   struct BiLinearTriSeg {

     int* Pack(int* size);
     bool UnPack(int* pack);
     double* BuildNormal(double* xi);
     double Area();
     double Metric(double* xi, double g[], double G[][3]);
     bool LocalCoordinatesOfNode(int lid, double* xi);

   };

/*!
\class Segment_BiLinearQuad

\brief <b> A class to define a 4-noded quadrilateral 2D Segment</b>

This class defines a 4-noded linear 2D quadrilateral interface surface segment.

<b>Important:</b><br>
Upon construction, the user must assure that the nodes attached to this segment are
given in counter-clockwise order such that the outward normal to the domain
points out from the screen:<br>
<pre>
        Node 3     Node 2
          o----------o
          |          |
          |          |           
          |          |                   <-------| 
          |          |            domain surface | 
          o----------o                      ------  
        Node 0     Node 1

</pre>
The reason for this is that the outward normal to the interface is implicitly defined by
the orientation of the segments. There is no way for the MOERTEL package to check the
orientation of the segments as they are passed in by the user and most obscure results
will be produced when the orientation is uncorrect!

<b>Important:</b><br>
There is currently no full support for quadrilateral interface discretizations. However,
when quads are added to a \ref MOERTEL::Interface they will be split into
2 triangles internally that are then used to perform the integration. The orientation of
the 2 triangles resulting from 1 quad is consistent with the orientation of the quad.

*/

   struct BiLinearQuadSeg {

     int* Pack(int* size);
     bool UnPack(int* pack);
     double* BuildNormal(double* xi);
     double Area();
     double Metric(double* xi, double g[], double G[][3]);
     bool LocalCoordinatesOfNode(int lid, double* xi);

   };


} // namespace MOERTEL

// << operator
SEGMENT_TEMPLATE_STATEMENT
std::ostream& operator << (std::ostream& os, const MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)& seg);

#ifndef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_SegmentT_Def.hpp"
#endif

#endif // MOERTEL_SEGMENT_H
