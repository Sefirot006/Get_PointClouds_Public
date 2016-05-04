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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr visu_pc (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr visu_pc2 (new pcl::PointCloud<pcl::PointXYZRGB>);

using namespace pcl;

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

void detect_keypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points,float min_scale,int nr_octaves,int nr_scales_per_octave,float min_contrast)
{
  PointCloud<pcl::PointWithScale> result;
  SIFTKeypoint<PointXYZRGB, PointWithScale> sift_detect;
  // Use a FLANN-based KdTree to perform neighborhood searches
  search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ> ());
  // Set the detection parameters
  sift_detect.setScales(min_scale, nr_octaves, nr_scales_per_octave);
  sift_detect.setMinimumContrast(min_contrast);
  // Set the input
  sift_detect.setInputCloud(points);
  // Detect the keypoints and store them in "keypoints_out"
  sift_detect.compute(result);
  std::cout << "No of SIFT points in the result are " << result.points.size () << std::endl;
}



void simpleVis ()
{
  	pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
	while(!viewer.wasStopped())
	{
	  viewer.showCloud (visu_pc);
	  boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
	}

}

void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& msg)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>(*msg));
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

	cout << "Puntos capturados: " << cloud->size() << endl;

	pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;
	vGrid.setInputCloud (cloud);
	vGrid.setLeafSize (0.05f, 0.05f, 0.05f);
	vGrid.filter (*cloud_filtered);

	cout << "Puntos tras VGrid.Filter: " << cloud_filtered->size() << endl;

	visu_pc = cloud_filtered;
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
	
	//1. Extracción de características.
	//Este paso nos devolverá un conjunto de características Ci, que será el resultado de aplicar 
	//un detector y un descriptor de características. Habrá que experimentar con las opciones 
	//disponibles para determinar cuál es el más adecuado (por tiempo de ejecución y eficacia). 
    detect_keypoints(visu_pc, 0.005f, 6, 4, 0.005f);

	//2. Encontrar emparejamientos.
	//Usaremos el método que proporciona PCL para encontrar las correspondencias. 
	//El resultado de este paso es un con-junto de emparejamiento. 


    //3. Determinar la mejor transformación que explica los emparejamientos.
    //Es posible que haya muchos malos emparejamientos, por ello en este paso 
    //tenemos que determinar la mejor transformación que explica los 
    //emparejamientos encontrados. Para ello, usaremos el algoritmo RANSAC.
    

	//4. Aplicar filtro de reducción VoxelGrid + Construcción del mapa.
	//Por último, hay que construir el mapa. Como cada toma de la Kinect tiene 
	//aproximadamente 300.000 puntos, en el momento que tengamos unas cuantas 
	//tomas vamos a manejar demasiados puntos, por lo que hay que proceder a reducirlos. 
	//Para ellos, podemos usar el filtro de reducción VoxelGrid, disponible en PCL. 
    

  }
}
