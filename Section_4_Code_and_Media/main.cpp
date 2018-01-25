/////////////////////////////////////////////////////
// Research Internship Assignement
// Section 4
// Dr. Sebastian Scherer 
// By - Mayank Roy
// IIT Delhi
//////////////////////////////////////////////////////

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <stdio.h>
#include <string.h>

// filtering
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>



using namespace cv;
using namespace std;


/**
 * Load data for this assignment.
 * @param fname The JSON input filename.
 * @param left_fnames The output left images of the stereo pair.
 * @param right_fnames The output right images of the stereo pair.
 * @param poses The 6D poses of the camera when the images were taken.
 *
 * This will probably throw an exception if there's something wrong
 * with the json file.
 */
void LoadMetadata(const std::string& fname,
                  std::vector<std::string>& left_fnames,
                  std::vector<std::string>& right_fnames,
                  std::vector<Eigen::Affine3d>& poses,
//		  Eigen::Affine3d &pose,	
		  Eigen::Quaterniond quat[5],
		  Eigen::Vector3d trans[5]) {
  namespace bpt = boost::property_tree;
  bpt::ptree pt;
  bpt::read_json(fname, pt);
  int i = 0;
  for (bpt::ptree::iterator itr=pt.begin();
       itr != pt.end(); ++itr) {
    bpt::ptree::value_type v(*itr);
    bpt::ptree entry(v.second);
    std::string left_fname( entry.get<std::string>("left") );
    std::string right_fname( entry.get<std::string>("right") );
    left_fnames.push_back(left_fname);
    right_fnames.push_back(right_fname);
    Eigen::Vector3d t(entry.get<double>("pose.translation.x"),
                      entry.get<double>("pose.translation.y"),
                      entry.get<double>("pose.translation.z"));
    trans[i] = t;
    Eigen::Quaterniond q(entry.get<double>("pose.rotation.w"),
                         entry.get<double>("pose.rotation.x"),
                         entry.get<double>("pose.rotation.y"),
                         entry.get<double>("pose.rotation.z"));
    quat[i] = q;
    i++;

    Eigen::Affine3d aff = Eigen::Translation3d(t*100) * q;
    //*pose = Eigen::Translation3d(t) * q;
    poses.push_back(aff);
  }
}

/**
 * Load calibration data.
 * Note this is basically the ROS CameraInfo message.
 * See
 * http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
 * http://wiki.ros.org/image_pipeline/CameraInfo
 * for reference.
 *
 * Note: you probably don't need all the parameters ;)
 */
void LoadCalibration(const std::string& fname,
                     int &width,
                     int &height,
                     cv::Mat& D,
                     cv::Mat& K,
                     cv::Mat& R,
                     cv::Mat& P) {
  namespace bpt = boost::property_tree;
  bpt::ptree pt;
  bpt::read_json(fname, pt);
  width = pt.get<int>("width");
  height = pt.get<int>("height");
  {
    bpt::ptree &spt(pt.get_child("D"));
    D.create(5, 1, CV_32FC1);
    int i=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++i) {
      D.at<float>(i,0) = itr->second.get<float>("");
    }
  }
  {
    bpt::ptree &spt(pt.get_child("K"));
    K.create(3, 3, CV_32FC1);
    int ix=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++ix) {
      int i=ix/3, j=ix%3;
      K.at<float>(i,j) = itr->second.get<float>("");
    }
  }
  {
    bpt::ptree &spt(pt.get_child("R"));
    R.create(3, 3, CV_32FC1);
    int ix=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++ix) {
      int i=ix/3, j=ix%3;
      R.at<float>(i,j) = itr->second.get<float>("");
    }
  }
  {
    bpt::ptree &spt(pt.get_child("P"));
    P.create(3, 4, CV_32FC1);
    int ix=0;
    for (bpt::ptree::iterator itr=spt.begin(); itr != spt.end(); ++itr, ++ix) {
      int i=ix/4, j=ix%4;
      P.at<float>(i,j) = itr->second.get<float>("");
    }
  }
}



/* Class for converting Image to cloud
Includes disparity map and transformations required and object instances.
*/

class Image_Cloud
{
	public:
	Mat left;
	Mat right;
	Mat disp;
	Mat disparity;
	Mat d3Image;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  	pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud;
	Eigen::Affine3d pose;
	Image_Cloud()
	{
		cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
		filtered_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr  (new pcl::PointCloud<pcl::PointXYZRGB>);
	}	

	//Find Disparity in Images

	void computeDisparity(int SADsize, int numDisp, int filtCap, int minDisp, int uniqRatio) 
	{	
		Mat g1, g2;
		cvtColor(left, g1, CV_BGR2GRAY);
		cvtColor(right, g2, CV_BGR2GRAY);
		StereoSGBM sbm;
        	sbm.SADWindowSize = SADsize;
	        sbm.numberOfDisparities = numDisp;
	        sbm.preFilterCap = filtCap;
	        sbm.minDisparity = minDisp;
        	sbm.uniquenessRatio = uniqRatio;
	        sbm.fullDP = true;
	        //sbm.P1 = 216;
	        //sbm.P2 = 864;
	        sbm(g1, g2, disparity);
		normalize(disparity, disp, 0, 255, CV_MINMAX, CV_8U);
	}

	//Filter outliers to reduce noise

	void  filterOutliers()
	{
//  		std::cout << "Cloud before filtering: " << std::endl;
//	  	std::cout << *cloud << std::endl;
  		// Create the filtering object
  		//Filtering based on mean radius
  		pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  		sor.setInputCloud (cloud);
	  	sor.setMeanK (50);
  		sor.setStddevMulThresh (0.4);
  		sor.filter (*filtered_cloud);
//  		std::cout << "Cloud after filtering: " << std::endl;
//  		std::cout << *filtered_cloud << std::endl;
	}

	//Transform cloud to allign to world frame
	void dispCloud(int numDisp)
	{
    		const double max_z = 2.0e6;
		disparity.convertTo(disp, CV_64F);
		pcl::PointXYZRGB p;
	    	Vec3d point ;
		Eigen::Vector4d homg_point;
		Eigen::Vector4d world_point;
			
		double px, py, cx = 312.227073669434 , cy = 252.058246612549, f = 323.451202597659, b = 0.414614047;	
//		double px, py, cx = 326.315072140693 , cy = 245.698824887744, f = 373.56855147796, b = 0.414;	
		for (int i=0; i < disp.rows; i++) 
		{
		    for (int j=0; j < disp.cols; j++) 
		    {	
			double d  = disp.at<double>(i,j)/(16*numDisp);
			px= j;
			py = i;

			homg_point[2] = f*b/d;
			homg_point[0] = (px-cx)*homg_point[2]/f;
			homg_point[1] = (py- cy)*homg_point[2]/f;
			homg_point[3] = 1;

		        if(fabs(homg_point[2]) < 1 || fabs(homg_point[2]) > max_z) continue;
			world_point =  pose.matrix() * homg_point;
			//cout<<world_point<<"   :   "<<homg_point<<endl;
			p.x = world_point[0]/world_point[3];
			p.y = world_point[1]/world_point[3];
			p.z = world_point[2]/world_point[3];
			cv::Vec3b bgr(left.at<cv::Vec3b>(i, j));
			p.b = bgr[0];
			p.g = bgr[1];
			p.r = bgr[2];
			cloud->points.push_back( p );
		    }
	  	}
	}
};





int main(int argc, char *argv[]) 
{
  if (argc < 4) 
  {
    std::cerr << "Usage: " << argv[0] << " JSON_DATA_FILE JSON_LEFT_CALIB_FILE JSON_RIGHT_CALIB_FILE\n";
    return -1;
  }

  // load metadata
  std::vector<std::string> left_fnames, right_fnames;
  std::vector<Eigen::Affine3d> poses;
  Eigen::Quaterniond quat[5];
  Eigen::Vector3d trans[5];
  Image_Cloud view[5]; 
  LoadMetadata(argv[1], left_fnames, right_fnames, poses, quat, trans);
  // load calibration info.
  // note: you should load right as well
  int left_w, left_h;
  cv::Mat left_D, left_K, left_R, left_P;
  LoadCalibration(argv[2], left_w, left_h, left_D, left_K, left_R, left_P);
  int right_w,right_h;
  cv::Mat right_D, right_K, right_R, right_P;
  LoadCalibration(argv[3], right_w, right_h, right_D, right_K, right_R, right_P);

// Varying parameters for various image pairs
  int SADsize[5] = {15,15,15,15,15};
  int numDisp[5]= {80,80,80,64,80};
  int minDisp[5] = {1,2,1,2,1};
  int uniqRatio[5] = {10,10,10,10,10};
  int filtCap[5] = {50,50,50,50,50};
  
  pcl::PointCloud<pcl::PointXYZRGB> totalCloud;

  for(int i = 0; i<5; i++)
  {
  	std::cout << "loading " << left_fnames[i] << " ... ";
	view[i].left = cv::imread(left_fnames[i]);
   	view[i].right = cv::imread(right_fnames[i]);
	if (view[i].left.empty() || view[i].right.empty()) 
	{
    		std::cerr << "image not found.\n";
    		return -1;
  	} 
	else 
	{
	std::cout << "loaded image file with size " << view[i].left.cols << "x" << view[i].left.rows << "\n";
  	}

  	// then you should do some stereo magic. Feel free to use
  	// OpenCV, other 3rd party library or roll your own.
	cv::Mat disp;
  	view[i].computeDisparity(SADsize[i],numDisp[i],filtCap[i],minDisp[i],uniqRatio[i]);
  	// etc.
  	// finally compute the output point cloud from one or more stereo pairs.
	Rect roi1, roi2;

	view[i].pose = poses[i];
	view[i].dispCloud(numDisp[i]);
	
	view[i].cloud->width = (int) view[i].cloud->points.size ();
	view[i].cloud->height = 1;
	view[i].filterOutliers();
	view[i].filtered_cloud->width = (int) view[i].filtered_cloud->points.size ();
	view[i].filtered_cloud->height = 1;

	cout<<*(view[i].filtered_cloud)<<endl;
  	// open it with pcl_viewer. then press 'r' and '5' to see the rgb.
  	//pcl::io::savePCDFileASCII("out.pcd", pc);
  	std::stringstream viewNames;
  	viewNames << "view"<< i << ".pcd";
  	std::string viewName = viewNames.str();
  	char *name = const_cast<char*>(viewName.c_str());
  	std::cout << "saving a pointcloud to "<<viewNames.str()<<"\n";
  	pcl::io::savePCDFileASCII(name,*(view[i].filtered_cloud));
//	cout<<poses[i].matrix()<<endl;
	//Concatinatin Clouds
  	totalCloud += *(view[i].filtered_cloud);
}
  	pcl::io::savePCDFileASCII("final.pcd",totalCloud);
 	cout<<"Written final cloud"<<endl; 
  return 0;
}








//ICP Related code - 


/*
//std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
  
  // Create a PCLVisualizer object
	PointCloud::Ptr result (new PointCloud), source (new PointCloud), target (new PointCloud);
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;
  
  for (int i = 0; i < 4; ++i)
  {
    *source = *view[i].filtered_cloud;
    *target = *view[i+1].filtered_cloud;

    // Add visualization data
//    showCloudsLeft(source, target);

    PointCloud::Ptr temp (new PointCloud);
//    PCL_INFO ("Aligning %s (%d) with %s (%d).\n", data[i-1].f_name.c_str (), source->points.size (), data[i].f_name.c_str (), target->points.size ());
    pairAlign (source, target, temp, pairTransform, false);

    //transform current pair into the global transform
    pcl::transformPointCloud (*temp, *result, GlobalTransform);

    //update the global transform
    GlobalTransform = GlobalTransform * pairTransform;
		//save aligned pair, transformed into the first cloud's frame
    std::stringstream ss;
    ss << i << ".pcd";
    pcl::io::savePCDFile (ss.str (), *result, true);

  }
*/

//  return 0;
//}


///////////////////////////////////////////////////////////
// Implementation of ICP approach to align point clouds
// Some basic allignment is necessary
///////////////////////////////////////////////

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 */
/*
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_representation.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/visualization/pcl_visualizer.h>

*/
/*

//convenient typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// This is a tutorial so we can afford having global variables 
//our visualizer
pcl::visualization::PCLVisualizer *p;
//its left and right viewports
int vp_1, vp_2;




//convenient structure to handle our pointclouds

struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};

struct PCDComparator
{
  bool operator () (const PCD& p1, const PCD& p2)
  {
    return (p1.f_name < p2.f_name);
  }
};


// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
//    out[4] = p.r;
//    out[5] = p.g;
//    out[6] = p.b;

    out[3] = p.curvature;
  }
};

////////////////////////////////////////////////////////////////////////////////
** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  *
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  //
  // Downsample for consistency and speed
  // \note enable this for large datasets
  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.05, 0.05, 0.05);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }


  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
  
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  //
  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (0.01);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (100);  
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);



  //
  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations (2);
  for (int i = 0; i < 3; ++i)
  {
    PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputSource (points_with_normals_src);
    reg.align (*reg_result);

		//accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

		//if the difference between this transformation and the previous one
		//is smaller than the threshold, refine the process by reducing
		//the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () / 10);
    
    prev = reg.getLastIncrementalTransformation ();

    // visualize current state
 //   showCloudsRight(points_with_normals_tgt, points_with_normals_src);
  }

	//
  // Get the transformation from target to source
  targetToSource = Ti.inverse();

  //
  // Transform target back in source frame
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);


  //add the source to the transformed target
  *output += *cloud_src;
  
  final_transform = targetToSource;
 }
*/
