//added for the python module version
#include "skeleton_param_module.hpp"
#include <Python.h>
//added comment to start rebuild
#define CGAL_EIGEN3_ENABLED
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <fstream>
#include <boost/foreach.hpp>
typedef CGAL::Simple_cartesian<double>                        Kernel2;
typedef Kernel2::Point_3                                       Point;
typedef CGAL::Polyhedron_3<Kernel2>                            Polyhedron;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor    vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> Skeletonization;
typedef Skeletonization::Skeleton                             Skeleton;
typedef Skeleton::vertex_descriptor                           Skeleton_vertex;
typedef Skeleton::edge_descriptor                             Skeleton_edge;

#if 1
#include <CGAL/IO/OFF_reader.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;



//only needed for the display of the skeleton as maximal polylines
struct Display_polylines{
    const Skeleton& skeleton;
    std::ofstream& out;
    int polyline_size;
    std::stringstream sstr;
    Display_polylines(const Skeleton& skeleton, std::ofstream& out)
    : skeleton(skeleton), out(out)
    {}
    void start_new_polyline(){
        polyline_size=0;
        sstr.str("");
        sstr.clear();
    }
    void add_node(Skeleton_vertex v){
        ++polyline_size;
        sstr << " " << skeleton[v].point;
    }
    void end_polyline()
    {
        out << polyline_size << sstr.str() << "\n";
    }
};
// This example extracts a medially centered skeleton from a given mesh.
int calcification_param(const char* location_with_filename,
                        double max_triangle_angle,
                      double quality_speed_tradeoff,
                      double medially_centered_speed_tradeoff,
                      double area_variation_factor,
                      int max_iterations,
                      bool is_medially_centered,
                      double min_edge_length,
                      double edge_length_multiplier,
                      bool print_parameters
    )
{
    //std::ifstream input((argc>1)?argv[1]:"data/elephant.off");

    //WORKED FOR ELEPHANT EXAMPLE BELOW
    /*
    std::string location = "/Users/brendancelii/Documents/C++_code/mesh_skeleton/";
    std::string filename = "elephant";
    */

    //std::string location = "/Users/brendancelii/Google Drive/Xaq Lab/Final_Blender/Blender_Annotations_Tool/Automated_Pipeline/temp/";
    //std::string filename = "neuron_648518346341366885"; //generted by mesh lab and doesn't work

    //std::string location = "/Users/brendancelii/Documents/C++_code/mesh_skeleton/";
    //std::string filename = "filled_cleaned_up"; //generted by mesh lab and doesn't work

    //std::ifstream input(location + filename + ".off");
    std::string location_with_filename_str(location_with_filename);
    std::string input_file_name = location_with_filename_str + ".off";
    std::ifstream input(input_file_name.c_str());

    /* Old way of importing mesh
    Polyhedron mesh;
    input >> mesh;
    if (!CGAL::is_triangle_mesh(mesh))
    {
        std::cout << "Input geometry is not triangulated." << std::endl;
        return EXIT_FAILURE;
    }
    */

    //New way of importing mesh
    Polyhedron mesh;
    if (!input)
    {
        std::cerr << "Cannot open file " << std::endl;
        return 2;
    }
    std::vector<K::Point_3> points;
    std::vector< std::vector<std::size_t> > polygons;
    if (!CGAL::read_OFF(input, points, polygons))
    {
        std::cerr << "Error parsing the OFF file " << std::endl;
        return 3;
    }
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    if(!CGAL::is_closed(mesh)){
        std::cerr << "Not closed mesh"<<std::endl;
        return 4;
    }
    if (CGAL::is_closed(mesh) && (!CGAL::Polygon_mesh_processing::is_outward_oriented(mesh)))
        CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);

    //end of new way



    if(mesh.empty()){
        return 4;
    }
    if(( !CGAL::is_triangle_mesh(mesh))){
        return 6;
    }
    

    Skeleton skeleton;
    Skeletonization mcs(mesh);

    /* Older way of computing the skeleton
    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);
    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";
    */
    

    /* NEW WAY OF COMPUTING THE SKELETONS WITH PARAMETER SETTING */

    //Add in the extra setters that could control skeletonization
    // Algorithm termination parameters
    
    /*
     The settings of the parameters in the constructor:
     
     max_triangle_angle() == 1.91986
     quality_speed_tradeoff() == 0.1
     medially_centered_speed_tradeoff() == 0.2
     area_variation_factor() == 0.0001
     max_iterations() == 500
     is_medially_centered() == true
     min_edge_length() == 0.002 * the length of the diagonal of the bounding box of tmesh
     
     
     */
    
    if(print_parameters){
        std::cout << "MAX ITERATIONS initially " << mcs.max_iterations() << "\n";
        std::cout << "AREA VARIATION initially " << mcs.area_variation_factor() << "\n";
        std::cout << "medial centered initially " << mcs.is_medially_centered() << "\n";
        std::cout << "medially_centered_speed_tradeoff initially " << mcs.medially_centered_speed_tradeoff() << "\n";
        std::cout << "quality_speed_tradeoff initially " << mcs.quality_speed_tradeoff() << "\n";
        std::cout << "max_triangle_angle initially " << mcs.max_triangle_angle() << "\n";
        std::cout << "min_edge_length initially " << mcs.min_edge_length() << "\n";
    }


    //initially MAX INTERATIONS = 500
    //Purpose: Maximum number of iterations performed by contract_until_convergence()
    
    mcs.set_max_iterations(max_iterations);
    

    //INITIALLY 0.0001
    //Purpose: The convergence is considered to be reached if the variation of the area of the meso-skeleton after one iteration is smaller than
    
    mcs.set_area_variation_factor(area_variation_factor);
    

    //*** Vertex Motion Parameters **//
    //initially is_medially_centered = true
    //Purpose: If true, the meso-skeleton placement will be attracted by an approximation of the medial axis of the mesh during the contraction steps, so will be the result skeleton.
    
    mcs.set_is_medially_centered(is_medially_centered);
    

    //initially medially_centered_speed_tradeoff == 0.2  , only active if mcs.is_medially_centered == True
    // Purpose: Controls the smoothness of the medial approximation: increasing this value results in a (less smooth) skeleton closer to the medial axis, as well as a lower convergence speed. It is only used if is_medially_centered()==true. This parameter corresponds to wM in the original publication
    
    mcs.set_medially_centered_speed_tradeoff(medially_centered_speed_tradeoff);
    

    //initially quality_speed_tradeoff == 0.1
    // Purpose: Controls the velocity of movement and approximation quality: decreasing this value makes the mean curvature flow based contraction converge faster, but results in a skeleton of lower quality. This parameter corresponds to wH in the original publication
    
    mcs.set_quality_speed_tradeoff(quality_speed_tradeoff);
    
    //***** these were added on  5/31 ***//
    
    //**** Local remeshing step
    // other parameters that can be set to control skeleton
    // During the local remeshing step, a triangle will be split if it has an angle larger than (double)
    
    mcs.set_max_triangle_angle(max_triangle_angle);
    
    
    // During the local remeshing step, an edge will be collapse if it is length is less than (double)
    
    if(min_edge_length> 0){
        if(print_parameters){
            std::cout << "setting the edge length by absolute length" << "\n";
        }
        mcs.set_min_edge_length(min_edge_length);
    }
    else{
        if(print_parameters){
            std::cout << "setting the edge length by multiplier" << "\n";
        }
        mcs.set_min_edge_length(double(mcs.min_edge_length())/0.002*edge_length_multiplier);
    }
    
    
    

    if(print_parameters){
        std::cout << "min_edge_length AFTER = " << mcs.min_edge_length() << "\n";
        std::cout << "max_triangle_angle AFTER = " << mcs.max_triangle_angle() << "\n";
        std::cout << "quality_speed_tradeoff AFTER = " << mcs.quality_speed_tradeoff() << "\n";
        std::cout << "set_medially_centered_speed_tradeoff AFTER = " << mcs.medially_centered_speed_tradeoff() << "\n";
        std::cout << "medial centered AFTER " << mcs.is_medially_centered() << "\n";
        std::cout << "AREA VARIATION AFTER " << mcs.area_variation_factor() << "\n";
        std::cout << "MAX ITERATIONS AFTER " << mcs.max_iterations() << "\n";
    }

    
    

    // 1. Contract the mesh by mean curvature flow.
    mcs.contract_geometry();
    // 2. Collapse short edges and split bad triangles.
    mcs.collapse_edges();
    mcs.split_faces();
    // 3. Fix degenerate vertices.
    mcs.detect_degeneracies();
    // Perform the above three steps in one iteration.
    mcs.contract();
    // Iteratively apply step 1 to 3 until convergence.
    mcs.contract_until_convergence();
    // Convert the contracted mesh into a curve skeleton and
    // get the correspondent surface points
    mcs.convert_to_skeleton(skeleton);

    //initially is_medially_centered = true

    if(print_parameters){
        std::cout << "medial centered END " << mcs.is_medially_centered() << "\n";
        std::cout << "set_medially_centered_speed_tradeoff END = " << mcs.medially_centered_speed_tradeoff() << "\n";
        std::cout << "quality_speed_tradeoff END = " << mcs.quality_speed_tradeoff() << "\n";
        std::cout << "Max iterations END = " << mcs.max_iterations() << "\n";
        std::cout << "AREA VARIATION END " << mcs.area_variation_factor() << "\n";
    }


    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";
    // Output all the edges of the skeleton.
    std::ofstream output(location_with_filename_str + "_skeleton.cgal");


    Display_polylines display(skeleton,output);
    CGAL::split_graph_into_polylines(skeleton, display);
    output.close();

    /* old way of outputting
    BOOST_FOREACH(Skeleton_edge e, edges(skeleton))
    {
        const Point& s = skeleton[source(e, skeleton)].point;
        const Point& t = skeleton[target(e, skeleton)].point;
        output << "2 "<< s << " " << t << "\n";
    }
    output.close();
    */


    // Output all the edges of the skeleton.
    //std::ofstream output("skel-poly.cgal");
    
    //output.open("correspondance-poly.cgal");

    /* Output skeleton points and the corresponding surface points
    output.open(output_location + filename + "correspondance-poly-.cgal");
    BOOST_FOREACH(Skeleton_vertex v, vertices(skeleton))
    BOOST_FOREACH(vertex_descriptor vd, skeleton[v].vertices)
    output << "2 " << skeleton[v].point << " "
    << get(CGAL::vertex_point, mesh, vd)  << "\n";
    */
    return EXIT_SUCCESS;
}



static PyObject* calcification_param_C(PyObject *self, PyObject *args)
{
    
    const char *location_with_filename;

    //for controlling the parameters
    double max_triangle_angle;
    double quality_speed_tradeoff;
    double medially_centered_speed_tradeoff;
    double area_variation_factor;
    int max_iterations;
    int is_medially_centered;
    double min_edge_length;
    double edge_length_multiplier;
    int print_parameters;




    int output_int;
    
    if (!PyArg_ParseTuple(args,"sddddiiddi",&location_with_filename,
                            &max_triangle_angle,
                            &quality_speed_tradeoff,
                            &medially_centered_speed_tradeoff,
                            &area_variation_factor,
                            &max_iterations,
                            &is_medially_centered,
                            &min_edge_length,
                            &edge_length_multiplier,
                            &print_parameters))
        return NULL;
    
    output_int = calcification_param(location_with_filename,
                                    max_triangle_angle,
                                    quality_speed_tradeoff,
                                    medially_centered_speed_tradeoff,
                                    area_variation_factor,
                                    max_iterations,
                                    is_medially_centered,
                                    min_edge_length,
                                    edge_length_multiplier,
                                    print_parameters);
    return Py_BuildValue("i",output_int);
    
}



static PyMethodDef calcification_param_Methods[] =
{
    { "calcification_param", calcification_param_C, METH_VARARGS, "calculates the mesh skeleton with parameter specifications" },
    { NULL,NULL,0, NULL }
    
};

static struct PyModuleDef calcification_param_Module =
{
    PyModuleDef_HEAD_INIT,
    "calcification_param_Module",
    "CGAL Skeleton Module with parameters",
    -1,
    calcification_param_Methods
};

PyMODINIT_FUNC PyInit_calcification_param_Module(void )
{
    return PyModule_Create(&calcification_param_Module);
}




