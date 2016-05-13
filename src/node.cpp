// NARF KeyPoints
// http://pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction
// Importante
// http://www.jeffdelmerico.com/wp-content/uploads/2014/03/pcl_tutorial.pdf

#include <boost/foreach.hpp>
#include <geometry_msgs/Twist.h>
#include <pcl-1.7/pcl/keypoints/harris_3d.h>
#include <pcl/common/io.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>


pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2 (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapa (new pcl::PointCloud<pcl::PointXYZRGB>);

using namespace pcl;

//pcl::visualization::PCLVisualizer::Ptr viewer;

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
    if(cmd[0]!='+' && cmd[0]!='l' && cmd[0]!='r' && cmd[0]!='.') {
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

void SIFTdetect_keypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &points, PointCloud<PointWithScale> &keypoints, float min_scale,int nr_octaves,int nr_scales_per_octave,float min_contrast)
{
  //cout << "entro en la deteccion de los keypoints" << endl;
  PointCloud<pcl::PointWithScale>::Ptr result (new PointCloud<PointWithScale>);
  PointCloud<pcl::PointWithScale>::Ptr aux (new PointCloud<PointWithScale>);
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
  aux = result;

  PointCloud<pcl::PointWithScale>::Ptr keypoints_ptr (new PointCloud<PointWithScale>);
  copyPointCloud(*aux, *keypoints_ptr);

  //std::cout << "No of SIFT points in the result are " << keypoints_ptr->points.size () << std::endl;

  copyPointCloud(*keypoints_ptr,keypoints);
}

void HARRISdetect_keypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB> &keypoints){
  cout << "Keypoints con HARRIS" << endl;

  PointCloud<PointXYZI>::Ptr result (new PointCloud<PointXYZI>);
  PointCloud<PointXYZI>::Ptr aux (new PointCloud<PointXYZI>);

  HarrisKeypoint3D<PointXYZRGB,PointXYZI> *harris = new HarrisKeypoint3D<PointXYZRGB,PointXYZI> (HarrisKeypoint3D<PointXYZRGB,PointXYZI>::HARRIS);

  harris->setNonMaxSupression(true);
  harris->setRadius(0.001f);
  harris->setRadiusSearch(0.005f);
  harris->setMethod(HarrisKeypoint3D<PointXYZRGB,PointXYZI>::LOWE);
  harris->setInputCloud(cloud);

  harris->compute(*result);
  aux = result;
  PointCloud<PointXYZRGB>::Ptr keypoints_ptr (new PointCloud<PointXYZRGB>);
  copyPointCloud(*aux, *keypoints_ptr);

  std::cout << "Nº of HARRIS points in the result are " << keypoints_ptr->points.size () << std::endl;

  copyPointCloud(*keypoints_ptr,keypoints);
}

void simpleVis (){
      pcl::visualization::CloudViewer viewer ("Cloud_1 Viewer");
    //pcl::visualization::CloudViewer viewer_2 ("Cloud_2 Viewer");
    while(!viewer.wasStopped()){
	  viewer.showCloud (mapa);
    //viewer_2.showCloud (cloud_2);
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

void PFHRGB(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals, PointCloud<PointXYZRGB>::Ptr &keypoints, 
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
    // Copia la primera nube de puntos en la segunda 
    copyPointCloud(*cloud, *cloud_1);
    copyPointCloud(*cloud, *mapa);
    cout << "Puntos capturados_1: " << cloud->size() << endl;
    empieza = false;
  }
  else{
    cout << "entro por el else" << endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_aux (new pcl::PointCloud<pcl::PointXYZRGB>(*msg));  
    copyPointCloud(*cloud_aux, *cloud_2);
    cout << "Puntos capturados_2: " << cloud_2->size() << endl;

    // Declaraciones
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloud<pcl::PointXYZRGB>::Ptr pcKeyPoints_1XYZ (new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints_1 (new pcl::PointCloud<pcl::PointWithScale>);
    PointCloud<PFHSignature125>::Ptr cloudDescriptors_1 (new pcl::PointCloud<PFHSignature125>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_2 (new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloud<pcl::PointXYZRGB>::Ptr pcKeyPoints_2XYZ (new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints_2 (new pcl::PointCloud<pcl::PointWithScale>);
    PointCloud<PFHSignature125>::Ptr cloudDescriptors_2 (new pcl::PointCloud<PFHSignature125>);

    // Filtrado de la nube y quitamos los NaN
    pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;
    vGrid.setInputCloud (cloud_1);
    vGrid.setLeafSize (0.05f, 0.05f, 0.05f);
    vGrid.filter (*cloud_filtered_1);
    cout << "Puntos tras VG_1: " << cloud_filtered_1->size() << endl;
    
    std::vector<int> indices;
    cloud_filtered_1->is_dense = false;
    removeNaNFromPointCloud<PointXYZRGB>(*cloud_filtered_1, *cloud_filtered_1, indices);
    cout << "Quitamos los NAN y quedan: " << cloud_filtered_1->size() << endl;
    //cloud_1 = cloud_filtered_1;
    copyPointCloud(*cloud_filtered_1, *cloud_1);
    cout << "Comprobacion del copy(tiene que dar los mismos puntos que en la linea anterior): " << cloud_1->size() << endl;

    // Filtrado de la nube y quitamos los NaN
    vGrid.setInputCloud (cloud_2);
    vGrid.filter (*cloud_filtered_2);
    cout << "Puntos tras VG_2: " << cloud_filtered_2->size() << endl;

    std::vector<int> indices_2;
    cloud_filtered_2->is_dense = false;
    removeNaNFromPointCloud<PointXYZRGB>(*cloud_filtered_2, *cloud_filtered_2, indices_2);
    cout << "Quitamos los NAN y quedan: " << cloud_filtered_2->size() << endl;

    copyPointCloud(*cloud_filtered_2, *cloud_2);
    cout << "Comprobacion del copy(tiene que dar los mismos puntos que en la linea anterior): " << cloud_2->size() << endl;


    pcl::PointCloud<pcl::Normal>::Ptr normals_1 (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_2 (new pcl::PointCloud<pcl::Normal>);
    const float normal_radius = 0.03;

    // Estas son para el SIFT
    //compute_surface_normals(cloud_1, normal_radius, normals_1);  
    //compute_surface_normals(cloud_2, normal_radius, normals_2); 

    // Detectamos los keypoints 
    //cout << "antes de entrar en la deteccion de los keypoints" << endl;
    //SIFTdetect_keypoints(cloud_1, *pcKeyPoints_1, 0.005f, 6, 4, 0.005f);
    //SIFTdetect_keypoints(cloud_2, *pcKeyPoints_2, 0.005f, 6, 4, 0.005f);

    HARRISdetect_keypoints(cloud_1, *pcKeyPoints_1XYZ);
    HARRISdetect_keypoints(cloud_2, *pcKeyPoints_2XYZ);

    // Esto para el HARRIS
    compute_surface_normals(pcKeyPoints_1XYZ, normal_radius, normals_1);  
    compute_surface_normals(pcKeyPoints_2XYZ, normal_radius, normals_2); 

    //std::cout << "No of SIFT points in the keypoints_1 are " << pcKeyPoints_1->points.size () << std::endl;
    //std::cout << "No of SIFT points in the keypoints_2 are " << pcKeyPoints_2->points.size () << std::endl;
    //cout << "paso la deteccion" << endl;
    
    if((pcKeyPoints_1->size() > 10 && pcKeyPoints_2->size() > 10) || (pcKeyPoints_1XYZ->size() > 10 && pcKeyPoints_2XYZ->size() > 10)){
      cout << "paso por el if" << endl;
      // features SIFT
      //PFH(cloud_1, normals_1, pcKeyPoints_1 , *cloudDescriptors_1, 0.05f);
      //PFH(cloud_2, normals_2, pcKeyPoints_2, *cloudDescriptors_2, 0.05f);

      // features HaRRIs
      PFHRGB(cloud_1, normals_1, pcKeyPoints_1XYZ , *cloudDescriptors_1, 0.05f);
      PFHRGB(cloud_2, normals_2, pcKeyPoints_2XYZ, *cloudDescriptors_2, 0.05f);

      std::cout << "No of PFH points in the descriptors_1 are " << cloudDescriptors_1->points.size () << std::endl;
      std::cout << "No of PFH points in the descriptors_2 are " << cloudDescriptors_2->points.size () << std::endl;

      // Con el HARRIS si que funciona
      
      // CorrespondenceRejactorSampleConsensus
      /**
      registration::CorrespondenceEstimation<PFHSignature125,PFHSignature125> corr_est;
      corr_est.setInputSource(cloudDescriptors_2);
      corr_est.setInputTarget(cloudDescriptors_1);

      cout << "antes de determinar las correspondencias" << endl;      

      boost::shared_ptr<Correspondences> correspondences (new Correspondences);
      //corr_est.determineCorrespondences (*correspondences);
      corr_est.determineReciprocalCorrespondences (*correspondences);

      cout << "ya se han determinado las correspondencias" << endl;

      boost::shared_ptr<Correspondences> correspondences_result_rej_sac (new Correspondences);
      registration::CorrespondenceRejectorSampleConsensus<PointXYZRGB> corr_rej_sac;
      corr_rej_sac.setInputSource(cloud_2);
      corr_rej_sac.setInputTarget(cloud_1);
      // ransac
      corr_rej_sac.setInlierThreshold(0.1);
      corr_rej_sac.setMaximumIterations(1000);
      corr_rej_sac.setInputCorrespondences(correspondences);
      corr_rej_sac.getCorrespondences(*correspondences_result_rej_sac);

      Eigen::Matrix4f transform_res_from_SAC = corr_rej_sac.getBestTransformation();

      cout << "Despues de todo el lio nos quedamos con: " << correspondences_result_rej_sac->size() << " correspondencias" << endl;
      cout << "transform from SAC: " << endl;
      cout <<  transform_res_from_SAC  << endl;
      */

      // Fin del CorrespondenceRejactorSampleConsensus

      // Esto va folladisimo en comparacion con el CorrespondenceRejactorSampleConsensus
      // TransformationEstimationSVD
      boost::shared_ptr<pcl::Correspondences> correspondences (new pcl::Correspondences);
      pcl::registration::CorrespondenceEstimation<PointXYZRGB, PointXYZRGB> corr_est;
      corr_est.setInputSource (cloud_2);
      corr_est.setInputTarget (cloud_1);
      corr_est.determineReciprocalCorrespondences (*correspondences);

      Eigen::Matrix4f transform_res_from_SVD;
      registration::TransformationEstimationSVD<PointXYZRGB, PointXYZRGB> trans_est_svd;
      trans_est_svd.estimateRigidTransformation(*cloud_2, *cloud_1,
                                                *correspondences,
                                                transform_res_from_SVD);

      cout << "Despues de todo el lio nos quedamos con: " << correspondences->size() << " correspondencias" << endl;
      cout << "transform from SAC: " << endl;
      cout <<  transform_res_from_SVD  << endl;

      // Fin del TransformationEstimationSVD

      /**
      // nuevo ICP
      IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
      icp.setInputSource(cloud_2);
      icp.setInputTarget(cloud_1);

      PointCloud<PointXYZRGB> final;
      icp.align(final);
      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      std::cout << icp.getFinalTransformation() << std::endl;
      Eigen::Matrix4f matrix_icp = icp.getFinalTransformation();
      */
      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
      // transformpointcloud
      transformPointCloud(*cloud_1,*transformed_cloud,transform_res_from_SVD);

      cout << "Despues de la transformacion" << endl;
      
      //cout << "la cloud_1 tiene: " << cloud_1->points.size() << " puntos" << endl;
      //cout << "la transformed_cloud tiene: " << transformed_cloud->points.size() << " puntos" << endl;
  
      for(int i=0;i<transformed_cloud->points.size();i++){
        mapa->points.push_back(transformed_cloud->points[i]);
      } 
      
      // Esta mierda no funciona
      //mapa += transformed_cloud;

      copyPointCloud(*cloud_2, *cloud_1);

    } 
  }
}

/**

void verEmparejamientos(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant, pcl::PointCloud<pcl::PointXYZRGB>::Ptr n_ant){
    //pcl::visualization::PCLVisualizer::Ptr viewer;//objeto viewer

    //constructor/inicialización:
    viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);

    //viewer->addCoordinateSystem (1.0); //podriamos dibujar un sistema de coordenadas si quisieramos
    viewer->initCameraParameters ();

    //visualizar nube “cloud”;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);   //esto es el manejador de color de la nube "cloud"
    if (!viewer->updatePointCloud (cloud,rgb, "cloud")) //intento actualizar la nube y si no existe la creo.
        viewer->addPointCloud(cloud,rgb,"cloud");

    //para “bloquear” el visualizador podemos poner un bucle que atienda a los eventos. obviamente es una solución temporal ya que nuestro programa no se puede quedar bloqueado visualizando. 
    while (!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep (boost::posix_time::microseconds (10));
    }

    //Una vez visto un ejemplo de como dibujar una nube sobre el visualizador de pcl, procedemos a visualizar las correspondencias. El objeto correspondencias de PCL tiene un conjunto de correspondencias que no es mas que dos índices a los puntos de los features. Por tanto, las nubes a pasarle a esta función tienen que ser las correspondientes que se hayan usado para las correspondencias.
    //TODO -> nube anterior
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant; // = 
    Eigen::Affine3f transfrom_translation = pcl::getTransformation(5.0, 0, 0, 0, 0, 0);

    pcl::PointCloud<pcl::PointXYZ>::Ptr n_ant_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant_transformed (new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::transformPointCloud (*cloud_ant, *cloud_ant_transformed,transfrom_translation);
    //pcl::transformPointCloud (*n_ant, *n_ant_transformed,transfrom_translation);


     //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> green(cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud(cloud);
    if (!viewer->updatePointCloud (cloud,rgbcloud, "cloudn1")) //intento actualizar la nube y si no existe la creo.
        viewer->addPointCloud(cloud,rgbcloud,"cloudn1");

    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red(cloud_ant, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbcloud_ant(cloud_ant_transformed);
    if (!viewer->updatePointCloud (cloud_ant_transformed,rgbcloud_ant, "cloudn2")) //intento actualizar la nube y si no existe la creo.
        viewer->addPointCloud(cloud_ant_transformed,rgbcloud_ant,"cloudn2");
    
    std::string corresname="correspondences";
    if (!viewer->updateCorrespondences<pcl::PointXYZ>(n_current,n_ant_transformed,correspondeces_sac,1)) //intento actualizar la nube y si no existe la creo.
        viewer->addCorrespondences<pcl::PointXYZ>(n_current,n_ant_transformed,correspondeces_sac,1, corresname);
}
*/

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZRGB> >("/camera/depth/points", 1, callback);
  // Descomentar para teleoperar
  ros::Publisher cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/teleop", 1);
  boost::thread t(simpleVis);

  while(ros::ok())
  {
    // Esto funciona pero habria que buscar la manera de hacerlo solo cuando queramos y no siempre
	//driveKeyboard(cmd_vel_pub_);
	ros::spinOnce();
	//viewer->spinOnce(1);
    
    //1. Extracción de características.
    //Este paso nos devolverá un conjunto de características Ci, que será el resultado de aplicar 
    //un detector y un descriptor de características. Habrá que experimentar con las opciones 
    //disponibles para determinar cuál es el más adecuado (por tiempo de ejecución y eficacia). 
    //feature_detector(visu_pc, 0.005f, 6, 4, 0.005f);

    //2. Encontrar emparejamientos.
    //Usaremos el método que proporciona PCL para encontrar las correspondencias. 
    //El resultado de este paso es un conjunto de emparejamiento.
    

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
