// NARF KeyPoints
// http://pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction
// Importante
// http://www.jeffdelmerico.com/wp-content/uploads/2014/03/pcl_tutorial.pdf


//transform pointclouds
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <geometry_msgs/Twist.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfhrgb.h>


pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2 (new pcl::PointCloud<pcl::PointXYZRGB>);

using namespace pcl;

bool empieza = true;

bool driveKeyboard(ros::Publisher cmd_vel_pub_)
{
  std::cout << "Type a command and then press enter.  "
    "Use '+' to move forward, 'l' to turn left, "
    "'r' to turn right, '.' to exit.\n";

  //we will be sending commands of type "twist"
  geometry_msgs::Twist base_cmd;

  char cmd[50];
  //while(nh_.ok()){

    std::cin.getline(cmd, 50);
    if(cmd[0]!='+' && cmd[0]!='l' && cmd[0]!='r' && cmd[0]!='.')
    {
      std::cout << "unknown command:" << cmd << "\n";
      //continue;
    }

    base_cmd.linear.x = base_cmd.linear.y = base_cmd.angular.z = 0;   
    //move forward
    if(cmd[0]=='+'){
      base_cmd.linear.x = 0.25;
    } 
    //turn left (yaw) and drive forward at the same time
    else if(cmd[0]=='l'){
      base_cmd.angular.z = 0.75;
      base_cmd.linear.x = 0.25;
    } 
    //turn right (yaw) and drive forward at the same time
    else if(cmd[0]=='r'){
      base_cmd.angular.z = -0.75;
      base_cmd.linear.x = 0.25;
    } 
    //quit
    else if(cmd[0]=='.'){
      //break;
    }
    else{
      base_cmd.angular.z = 0;
      base_cmd.linear.x = 0;
    }

    //publish the assembled command
    cmd_vel_pub_.publish(base_cmd);
  //}
  return true;
}

void detect_keypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, PointCloud<PointWithScale> &keypoints, float min_scale,int nr_octaves,int nr_scales_per_octave,float min_contrast)
{
  //cout << "entro en la deteccion de los keypoints" << endl;
  PointCloud<pcl::PointWithScale>::Ptr result (new PointCloud<PointWithScale>);
  PointCloud<pcl::PointWithScale>::Ptr prueba (new PointCloud<PointWithScale>);
  SIFTKeypoint<PointXYZRGB, PointWithScale> sift_detect;
  // Use a FLANN-based KdTree to perform neighborhood searches
  search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ> ());
  // Set the detection parameters
  sift_detect.setScales(min_scale, nr_octaves, nr_scales_per_octave);
  sift_detect.setMinimumContrast(min_contrast);
  // Set the input
  sift_detect.setInputCloud(points);
  // Detect the keypoints and store them in "keypoints_out"
  //cout << "a punto de compute" << endl;
  sift_detect.compute(*result);
  //cout << "PAso el coumpute" << endl;
  prueba = result;

  PointCloud<pcl::PointWithScale>::Ptr keypoints_ptr (new PointCloud<PointWithScale>);
  copyPointCloud(*prueba, *keypoints_ptr);

  //std::cout << "No of SIFT points in the result are " << keypoints_ptr->points.size () << std::endl;

  copyPointCloud(*keypoints_ptr,keypoints);
}

void simpleVis (){
  	pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
	while(!viewer.wasStopped()){
	  viewer.showCloud (cloud_2);
	  boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
	}

}

void PFH(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals, PointCloud<PointWithScale>::Ptr &keypoints, 
          PointCloud<PFHSignature125> &descriptors_out, float feature_radius){
  cout << "entro en el pfh" << endl;
  cout << "Dentro del pfh, keypoints: " << keypoints->points.size() << endl;

  PointCloud<PFHSignature125>::Ptr result(new PointCloud<PFHSignature125>);
  PointCloud<PFHSignature125>::Ptr prueba(new PointCloud<PFHSignature125>);
  
  // Creamos un objreoto PFHEstimation
  PFHEstimation<PointXYZRGB, Normal, PFHSignature125> pfh_est;
  pfh_est.setSearchMethod(search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));

  //Especifica el radio de la caracteristica pfh
  pfh_est.setRadiusSearch(feature_radius);

  PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb (new PointCloud<PointXYZRGB>);
  copyPointCloud(*keypoints, *keypoints_xyzrgb);

  // Usar todos los puntos para aalizar la estructura local de la nube
  pfh_est.setSearchSurface(points);
  pfh_est.setInputNormals(normals);

  // Pero solo computa las caracteristicas de los keypoints
  pfh_est.setInputCloud(keypoints_xyzrgb);


  cout << "a punto de computar" << endl;
  // Computa las caracteristicas
  pfh_est.compute(*result);

  prueba = result;

  PointCloud<PFHSignature125>::Ptr result_ptr(new PointCloud<PFHSignature125>);
  copyPointCloud(*prueba,*result_ptr);

  //std::cout << "No of PFH points in the descriptors are " << result_ptr->points.size () << std::endl;

  copyPointCloud(*result_ptr,descriptors_out);

}

void compute_surface_normals(PointCloud<PointXYZRGB>::Ptr &points, float normal_radius, PointCloud<Normal>::Ptr &normals_out){
  NormalEstimation<PointXYZRGB,Normal> norm_est;
  norm_est.setSearchMethod(search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));
  norm_est.setRadiusSearch(normal_radius);
  norm_est.setInputCloud(points);
  norm_est.compute(*normals_out);
}

void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& msg)
{
  if(empieza==true){
    cout << "entro por el if" << endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>(*msg));  
    copyPointCloud(*cloud, *cloud_1);
    cout << "Puntos capturados_1: " << cloud->size() << endl;
    empieza = false;
  }
  else{
    cout << "entro por el else" << endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_aux (new pcl::PointCloud<pcl::PointXYZRGB>(*msg));  
    copyPointCloud(*cloud_aux, *cloud_2);
    cout << "Puntos capturados_2: " << cloud_aux->size() << endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints_1 (new pcl::PointCloud<pcl::PointWithScale>);
    PointCloud<PFHSignature125>::Ptr cloudDescriptors_1 (new pcl::PointCloud<PFHSignature125>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints_2 (new pcl::PointCloud<pcl::PointWithScale>);
    PointCloud<PFHSignature125>::Ptr cloudDescriptors_2 (new pcl::PointCloud<PFHSignature125>);

    pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;
    vGrid.setInputCloud (cloud_1);
    vGrid.setLeafSize (0.05f, 0.05f, 0.05f);
    vGrid.filter (*cloud_filtered_1);
    cout << "Puntos tras VG_1: " << cloud_filtered_1->size() << endl;
    cloud_1 = cloud_filtered_1;

    vGrid.setInputCloud (cloud_2);
    vGrid.filter (*cloud_filtered_2);
    cout << "Puntos tras VG_2: " << cloud_filtered_2->size() << endl;
    cloud_2 = cloud_filtered_2;  
    
    pcl::PointCloud<pcl::Normal>::Ptr normals_1 (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_2 (new pcl::PointCloud<pcl::Normal>);
    const float normal_radius = 0.03;

    compute_surface_normals(cloud_filtered_1, normal_radius, normals_1);  
    compute_surface_normals(cloud_filtered_2, normal_radius, normals_2); 

    // Detectamos los keypoints 
    //cout << "antes de entrar en la deteccion de los keypoints" << endl;
    detect_keypoints(cloud_1, *pcKeyPoints_1, 0.005f, 6, 4, 0.005f);
    detect_keypoints(cloud_2, *pcKeyPoints_2, 0.005f, 6, 4, 0.005f);

    std::cout << "No of SIFT points in the keypoints_1 are " << pcKeyPoints_1->points.size () << std::endl;
    std::cout << "No of SIFT points in the keypoints_2 are " << pcKeyPoints_2->points.size () << std::endl;
    //cout << "paso la deteccion" << endl;
    if(pcKeyPoints_1->size() > 10 && pcKeyPoints_2->size() > 10){
      cout << "paso por el if" << endl;
      // features
      PFH(cloud_filtered_1, normals_1, pcKeyPoints_1, *cloudDescriptors_1, 0.05f);
      PFH(cloud_filtered_2, normals_2, pcKeyPoints_2, *cloudDescriptors_2, 0.05f);

      std::cout << "No of PFH points in the descriptors_1 are " << cloudDescriptors_1->points.size () << std::endl;
      std::cout << "No of PFH points in the descriptors_2 are " << cloudDescriptors_2->points.size () << std::endl;
    } 
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZRGB> >("/camera/depth/points", 1, callback);
  // Descomentar para teleoperar
  //ros::Publisher cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/teleop", 1);
  boost::thread t(simpleVis);

  while(ros::ok())
  {
    // Esto funciona pero habria que buscar la manera de hacerlo solo cuando queramos y no siempre
  	//driveKeyboard(cmd_vel_pub_);
		ros::spinOnce();
  }

}
