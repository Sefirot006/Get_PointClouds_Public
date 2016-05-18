
// NARF KeyPoints
// http://pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction
// Importante
// http://www.jeffdelmerico.com/wp-content/uploads/2014/03/pcl_tutorial.pdf

#include <boost/foreach.hpp>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/Twist.h>
#include <pcl-1.7/pcl/keypoints/harris_3d.h>
#include <pcl-1.7/pcl/keypoints/narf_keypoint.h>
#include <pcl-1.7/pcl/range_image/range_image.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/common/io.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/cvfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
//#include <pcl/registration/correspondence_rejection.h>
//#include <pcl/registration/correspondence_rejection_distance.h>
//#include <pcl/registration/correspondence_rejection_median_distance.h>
//#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/correspondence_rejection_var_trimmed.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
//#include <pcl/registration/transformation_estimation_lm.h>
//#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

#include <time.h>

const float normal_radius = 0.05f;
const float feature_radius = 0.05f;

//ros::NodeHandle nh;
ros::ServiceClient client;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapa (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcKeyPoints_antXYZ (new pcl::PointCloud<pcl::PointXYZRGB>);
//pcl::PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints_ant (new pcl::PointCloud<pcl::PointWithScale>);
//pcl::PointCloud<pcl::Normal>::Ptr normals_ant (new pcl::PointCloud<pcl::Normal>);
//pcl::PointCloud<pcl::PFHSignature125>::Ptr cloudDescriptors_ant (new pcl::PointCloud<pcl::PFHSignature125>);
pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloudDescriptors_ant (new pcl::PointCloud<pcl::FPFHSignature33>);
//pcl::PointCloud<pcl::VFHSignature308>::Ptr cloudDescriptors_ant (new pcl::PointCloud<pcl::VFHSignature308>);

using namespace pcl;

//pcl::visualization::PCLVisualizer::Ptr viewer;

bool empieza = true;

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
  //cout << "Paso el coumpute" << endl;
  aux = result;

  PointCloud<pcl::PointWithScale>::Ptr keypoints_ptr (new PointCloud<PointWithScale>);
  copyPointCloud(*aux, *keypoints_ptr);

  //std::cout << "No of SIFT points in the result are " << keypoints_ptr->points.size () << std::endl;
  copyPointCloud(*keypoints_ptr,keypoints);
}

void
narf_keypoints_detect(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &result)
{
  clock_t start, end;
  start = clock();
  pcl::RangeImage range;

  //Header-information for Range Image
  float angularResolution = (float) (  0.2f * (M_PI/180.0f));  //   0.5 degree in radians
  float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  //   360.0 degree in radians
  float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  //   180.0 degree in radians
  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
  float noiseLevel = 0.00;
  float minRange = 0.0f;
  int borderSize = 1;

  range.createFromPointCloud (*cloud, angularResolution, maxAngleWidth, maxAngleHeight, sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);

  //Extracting NARF-Keypoints
  pcl::RangeImageBorderExtractor range_image_ba;
  float support_size = 0.2f; //?

  //Keypoints
  pcl::NarfKeypoint narf_keypoint (&range_image_ba);
  narf_keypoint.setRangeImage (&range);
  narf_keypoint.getParameters ().support_size = support_size;
  pcl::PointCloud<int> keypoints_ind;
  narf_keypoint.compute (keypoints_ind);

  result->width = keypoints_ind.points.size();
  result->height = 1;
  result->is_dense = false;
  result->points.resize (result->width * result->height);

  int ind_count=0;

  //source XYZ-CLoud
  for (size_t i = 0; i < keypoints_ind.points.size(); i++)
  {
    ind_count = keypoints_ind.points[i];
    result->points[i].x = range.points[ind_count].x;
    result->points[i].y = range.points[ind_count].y;
    result->points[i].z = range.points[ind_count].z;
  }

  end = clock();

  cout << "Saliendo del NARF en (" << (end-start)/(double)CLOCKS_PER_SEC << ")" << endl;
}

void HARRISdetect_keypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB> &keypoints){
  cout << "Keypoints con HARRIS" << endl;
  clock_t start, end;
  start = clock();

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
   end = clock();

  cout << "Saliendo del HARRIS en (" << (end-start)/(double)CLOCKS_PER_SEC << ")" << endl;
}

void simpleVis(){
    //pcl::visualization::CloudViewer viewer ("Cloud Viewer");
    pcl::visualization::CloudViewer viewer ("Cloud_1 Viewer");
    //pcl::visualization::CloudViewer viewer_2 ("Cloud_2 Viewer");
    while(!viewer.wasStopped()){
	     viewer.showCloud (mapa);
       //viewer_2.showCloud (cloud_2);
       boost::this_thread::sleep(boost::posix_time::milliseconds(1000));

    }
}

void PFH(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals, PointCloud<PointWithScale>::Ptr &keypoints,
          float feature_radius, PointCloud<PFHSignature125> &descriptors_out)
{
  cout << "Entro en pfh(PFH)" << endl;
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

  cout << "A punto de computar" << endl;
  // Computa las caracteristicas
  pfh_est.compute(*result);

  prueba = result;

  PointCloud<PFHSignature125>::Ptr result_ptr(new PointCloud<PFHSignature125>);
  copyPointCloud(*prueba,*result_ptr);

  //std::cout << "No of PFH points in the descriptors are " << result_ptr->points.size () << std::endl;

  copyPointCloud(*result_ptr,descriptors_out);

}

void PFHRGB(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals, PointCloud<PointXYZRGB>::Ptr &keypoints,
          float feature_radius, PointCloud<PFHSignature125> &descriptors_out)
{
  clock_t start, end;
  start = clock();
  cout << "Entro en pfh(PFHRGB)" << endl;
  cout << "Dentro del pfh, keypoints: " << keypoints->points.size() << endl;
  cout << "Dentro del pfh, mapa: " << mapa->points.size() << endl;
  cout << "Dentro del pfh, normals: " << normals->points.size() << endl;

  PointCloud<PFHSignature125>::Ptr result(new PointCloud<PFHSignature125>);
  PointCloud<PFHSignature125>::Ptr prueba(new PointCloud<PFHSignature125>);

  // Creamos un objreoto PFHEstimation
  PFHEstimation<PointXYZRGB, Normal, PFHSignature125> pfh_est;
  pfh_est.setSearchMethod(search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));

  //Especifica el radio de la caracteristica pfh
  pfh_est.setRadiusSearch(feature_radius);

  PointCloud<pcl::PointXYZRGB>::Ptr keypoints_xyzrgb (new PointCloud<PointXYZRGB>);
  copyPointCloud(*keypoints, *keypoints_xyzrgb);
  cout << "Dentro del pfh, keypointsCopiados: " << keypoints_xyzrgb->size() << endl;

  // Usar todos los puntos para aalizar la estructura local de la nube
  pfh_est.setSearchSurface(points);
  pfh_est.setInputNormals(normals);

  // Pero solo computa las caracteristicas de los keypoints
  pfh_est.setInputCloud(keypoints_xyzrgb);

  cout << "A punto de computar" << endl;
  // Computa las caracteristicas
  pfh_est.compute(*result);

  cout << "Computo" << endl;

  prueba = result;

  PointCloud<PFHSignature125>::Ptr result_ptr(new PointCloud<PFHSignature125>);
  copyPointCloud(*prueba,*result_ptr);

  //std::cout << "No of PFH points in the descriptors are " << result_ptr->points.size () << std::endl;
  copyPointCloud(*result_ptr,descriptors_out);

  end = clock();

  cout << "Saliendo del pfh en (" << (end-start)/(double)CLOCKS_PER_SEC << ")" << endl;
}

void FPFH(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals, PointCloud<PointXYZRGB>::Ptr &keypoints,
          float feature_radius, PointCloud<FPFHSignature33> &descriptors_out)
{
  clock_t start, end;
  start = clock();
  cout << "Entro en fpfh" << endl;
  FPFHEstimation<PointXYZRGB, Normal, FPFHSignature33> fpfh;
  fpfh.setSearchSurface(points);
  fpfh.setInputCloud (keypoints);
  fpfh.setInputNormals (normals);

  search::KdTree<PointXYZRGB>::Ptr tree (new search::KdTree<PointXYZRGB>);
  fpfh.setSearchMethod (tree);

  PointCloud<FPFHSignature33>::Ptr fpfhs (new PointCloud<FPFHSignature33> ());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  fpfh.setRadiusSearch (0.05);

  cout << "A punto de computar" << endl;
  // Compute the features
  fpfh.compute (*fpfhs);

  copyPointCloud(*fpfhs,descriptors_out);

  end = clock();

  cout << "Saliendo del fpfh en (" << (end-start)/(double)CLOCKS_PER_SEC << ") con: " << fpfhs->points.size() << endl;
  // fpfhs->points.size () should have the same size as the input cloud->points.size ()*

}

void VFH(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals,
          float feature_radius, PointCloud<VFHSignature308> &descriptors_out) {
  clock_t start, end;
  start = clock();
  cout << "Entro en VFH" << endl;
  pcl::VFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> vfh;
  vfh.setInputCloud (points);
  vfh.setInputNormals (normals);

  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  vfh.setSearchMethod (tree);
  vfh.setNormalizeBins(true);
  vfh.setNormalizeDistance(false);

  cout << "A punto de computar" << endl;
  vfh.compute (descriptors_out);
  cout << "Computado!" << endl;

  end = clock();

  cout << "Saliendo del VFH en (" << (end-start)/(double)CLOCKS_PER_SEC << ") con: " << descriptors_out.points.size() << endl;
}

void CVFH(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals,
          float feature_radius, PointCloud<VFHSignature308> &descriptors_out) {
  clock_t start, end;
  start = clock();
  cout << "Entro en CVFH" << endl;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  pcl::CVFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::VFHSignature308> cvfh;
  cvfh.setInputCloud(points);
  cvfh.setInputNormals(normals);
  cvfh.setSearchMethod(kdtree);

  // Set the maximum allowable deviation of the normals,
  // for the region segmentation step.
  cvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
  // Set the curvature threshold (maximum disparity between curvatures),
  // for the region segmentation step.
  cvfh.setCurvatureThreshold(1.0);
  // Set to true to normalize the bins of the resulting histogram,
  // using the total number of points. Note: enabling it will make CVFH
  // invariant to scale just like VFH, but the authors encourage the opposite.
  cvfh.setNormalizeBins(false);

  cout << "A punto de computar" << endl;
  cvfh.compute(descriptors_out);
  cout << "Computado!" << endl;

  end = clock();

  cout << "Saliendo del VFH en (" << (end-start)/(double)CLOCKS_PER_SEC << ") con: " << descriptors_out.points.size() << endl;
}


void compute_surface_normals(PointCloud<PointXYZRGB>::Ptr &points, float normal_radius, PointCloud<Normal>::Ptr &normals_out){
  NormalEstimation<PointXYZRGB,Normal> norm_est;
  norm_est.setSearchMethod(search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));
  norm_est.setRadiusSearch(normal_radius);
  norm_est.setInputCloud(points);
  norm_est.compute(*normals_out);
}

void filter_cloud(PointCloud<PointXYZRGB>::Ptr &cloud, PointCloud<PointXYZRGB>::Ptr &cloud_filtered){
  pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;
  vGrid.setInputCloud (cloud);
  vGrid.setLeafSize (0.05f, 0.05f, 0.05f);
  vGrid.filter (*cloud_filtered);
  cloud_filtered->is_dense = false;
}

void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& msg){
  // Declaraciones
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      cloud_filtered   (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      pcKeyPoints_XYZ  (new pcl::PointCloud<pcl::PointXYZRGB>);
  //pcl::PointCloud<pcl::PFHSignature125>::Ptr  cloudDescriptors (new pcl::PointCloud<pcl::PFHSignature125>);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr  cloudDescriptors (new pcl::PointCloud<pcl::FPFHSignature33>);
  //pcl::PointCloud<pcl::VFHSignature308>::Ptr  cloudDescriptors (new pcl::PointCloud<pcl::VFHSignature308>);
  //pcl::PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints      (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  std::vector<int> indices;
  //pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;

  //Lectura de la nube actual.                            -> cloud
  copyPointCloud(*msg, *cloud);
  cout << "Puntos capturados: " << cloud->size() << endl;
  //Eliminado de los NaN de la nube actual.      -> cloud
  removeNaNFromPointCloud(*cloud, *cloud, indices);
  cout << "Quitamos los NAN y quedan: " << cloud->size() << endl;
  //micro_filter_cloud(cloud, cloud);
  //Detección de características                          -> pcKeyPoints_XYZ
  HARRISdetect_keypoints(cloud, *pcKeyPoints_XYZ);
  //narf_keypoints_detect(cloud, pcKeyPoints_XYZ);
  //Si detectamos un número de caracteristicas suficientes...
  if(pcKeyPoints_XYZ->size() > 10){
    cout << "Paso por el if" << endl;
    //Cálculo de normales a la superficie.                -> normals
    //compute_surface_normals(pcKeyPoints_XYZ, normal_radius, normals);
    compute_surface_normals(cloud, normal_radius, normals);

    //Extracción de características.                      -> cloudDescriptors
    //PFHRGB(cloud, normals, pcKeyPoints_XYZ, feature_radius, *cloudDescriptors);
    //std::cout << "Nº of PFH points in the descriptors_cloud are " << cloudDescriptors->points.size() << std::endl;
    FPFH(cloud, normals, pcKeyPoints_XYZ, feature_radius, *cloudDescriptors);

    //indices.clear();
    //cout << "Antes de los NAN y quedan: " << cloudDescriptors->size() << endl;
    //removeNaNFromPointCloud(*cloudDescriptors, *cloudDescriptors, indices);
    //cout << "Quitamos los NAN y quedan: " << cloudDescriptors->size() << endl;

    //VFH(cloud, normals, feature_radius, *cloudDescriptors);
    //CVFH(cloud, normals, feature_radius, *cloudDescriptors);
    std::cout << "Nº of FPFH points in the descriptors_cloud are " << cloudDescriptors->points.size() << std::endl;
  }

  //Si es la primera nube...
  if(empieza==true){
    cout << "Es primera nube." << endl;
    copyPointCloud(*cloud, *cloud_ant);
    //filter_cloud(cloud, cloud_filtered);
    //copyPointCloud(*cloud_filtered, *mapa);

    //Con la nube filtrada inicial se inicializa el mapa.
    filter_cloud(cloud, mapa);
    empieza = false;

    // Almacenando la nube en disco
    //pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);
    //cout << "Saved " << cloud->points.size () << " data points to test_pcd.pcd." << std::endl;
  }

  //Si no es la primera nube...
  else{
    cout << "Es nube siguiente." << endl;

    //Si se detectó un número de caracteristicas suficientes...
    if(pcKeyPoints_antXYZ->size() > 10){
      std::cout << "Nº of PFH points in the descriptors_cloud_ant are " << cloudDescriptors_ant->points.size() << std::endl;

      // Prueba de otra basura
      /**
      SampleConsensusInitialAlignment<PointXYZRGB, PointXYZRGB, FPFHSignature33> sac;

      sac.setMinSampleDistance(0.01);
      sac.setMaxCorrespondenceDistance(0.01);
      sac.setMaximumIterations(1);
      sac.setInputSource(cloud);
      sac.setSourceFeatures(cloudDescriptors);

      sac.setInputTarget(cloud_ant);
      sac.setTargetFeatures (cloudDescriptors_ant);

      PointCloud<PointXYZRGB>::Ptr aligned_source(new PointCloud<PointXYZRGB>);

      cout << "Antes del alineo1" << endl;
      sac.align(*aligned_source);
      cout << "Despues del alineo1" << endl;

      Eigen::Matrix4f initial_T = sac.getFinalTransformation();

      cout << "despues del initial_T" << endl;


      IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
      icp.setMaxCorrespondenceDistance (0.01);
      icp.setRANSACOutlierRejectionThreshold (0.01);
      icp.setTransformationEpsilon (0.1);
      icp.setMaximumIterations (100);

      icp.setInputSource(pcKeyPoints_XYZ);
      icp.setInputTarget (pcKeyPoints_antXYZ);

      PointCloud<PointXYZRGB>::Ptr registration_output(new PointCloud<PointXYZRGB>);
      cout << "Antes del alineo2" << endl;
      icp.align (*registration_output);
      cout << "Despues del alineo2" << endl;

      Eigen::Matrix4f refined_T = icp.getFinalTransformation()*initial_T;


      std::cout << refined_T << std::endl;


      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
      // transformpointcloud
      transformPointCloud(*transformed_cloud, *transformed_cloud, refined_T);

      cout << "Después de la transformación (ICP)." << endl;
      */

      ///////////////////////////////////////////
      // CorrespondenceRejactorSampleConsensus //
      ///////////////////////////////////////////

      //Determinación de correspondencias por CorrespondenceRejactorSampleConsensus.     -> transform_res_from_SAC
      //registration::CorrespondenceEstimation<PFHSignature125,PFHSignature125> corr_est;
      registration::CorrespondenceEstimation<FPFHSignature33,FPFHSignature33> corr_est;
      //registration::CorrespondenceEstimation<VFHSignature308,VFHSignature308> corr_est;
      corr_est.setInputSource(cloudDescriptors);
      corr_est.setInputTarget(cloudDescriptors_ant);
      //corr_est.setMaxCorrespondenceDistance(0.01);

      cout << "Antes de determinar las correspondencias." << endl;

      boost::shared_ptr<Correspondences> correspondences (new Correspondences);
      corr_est.determineCorrespondences (*correspondences);
      //corr_est.determineReciprocalCorrespondences (*correspondences);

      cout << "Ya se han determinado las correspondencias." << endl;

      boost::shared_ptr<Correspondences> correspondences_result_rej_sac (new Correspondences);
      registration::CorrespondenceRejectorSampleConsensus<PointXYZRGB> corr_rej_sac;
      corr_rej_sac.setInputSource(pcKeyPoints_XYZ);
      corr_rej_sac.setInputTarget(pcKeyPoints_antXYZ);
      // ransac
      corr_rej_sac.setInlierThreshold(0.01);
      corr_rej_sac.setMaximumIterations(100);

      corr_rej_sac.setInputCorrespondences(correspondences);

      cout << "Antes de obtener las correspondencias." << endl;
      corr_rej_sac.getCorrespondences(*correspondences_result_rej_sac);
      cout << "Ya se han obtenido las correspondencias." << endl;

      Eigen::Matrix4f transform_res_from_SAC = corr_rej_sac.getBestTransformation();

      cout << "Después de todo el lío nos quedamos con: " << correspondences->size() << " ó: " << correspondences_result_rej_sac->size() << " correspondencias." << endl;
      cout << "Matriz de transformación por Sample Consensus: " << endl;
      cout << transform_res_from_SAC << endl;

      ///////////////////////////////////////////////////
      // Fin del CorrespondenceRejactorSampleConsensus //
      ///////////////////////////////////////////////////

      // Esta mierda tampoco funciona...
      /*
           // TransformationEstimationSVD
      boost::shared_ptr<pcl::Correspondences> correspondences_2 (new pcl::Correspondences);
      pcl::registration::CorrespondenceEstimation<PFHSignature125, PFHSignature125> corr_est;
      corr_est.setInputSource (cloudDescriptors);
      corr_est.setInputTarget (cloudDescriptors_ant);
      corr_est.determineReciprocalCorrespondences (*correspondences_2);

      Eigen::Matrix4f transform_res_from_SVD;
      registration::TransformationEstimationSVD<PointXYZRGB, PointXYZRGB> trans_est_svd;
      trans_est_svd.estimateRigidTransformation(*pcKeyPoints_XYZ, *pcKeyPoints_antXYZ,
                                                *correspondences_2,
                                                transform_res_from_SVD);

      cout << "Despues de todo el lio nos quedamos con: " << correspondences_2->size() << " correspondencias" << endl;
      cout << "transform from SAC: " << endl;
      cout <<  transform_res_from_SVD  << endl;
      */


      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
      //Aplicar la transformación a la nube de puntos filtrada.         -> transformed_cloud
      transformPointCloud(*cloud, *transformed_cloud, transform_res_from_SAC);
      cout << "Después de la transformación (RANSAC)." << endl;

      //Recogemos la nube transformada desde RANSAC.
      //Método ICP
      IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
      icp.setInputSource(pcKeyPoints_XYZ);
      icp.setInputTarget(pcKeyPoints_antXYZ);

      //Parámetros a probar para mejorar los resultados en cuanto a tiempo y en cuanto a obtención de la distancia mínima global:
      /*
      int nr_iterations = 1000;
      int epsilon ;
      int distance ;
      icp.setMaximumIterations(nr_iterations);
      icp.setTransformationEpsilon(epsilon);
      icp.setEuclideanFitnessEpsilon(distance);
      icp.setMaxCorrespondenceDistance(distance);
      icp.setRANSACOutlierRejectionThreshold(distance);
      */

      PointCloud<PointXYZRGB> final;
      icp.align(final);
      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      Eigen::Matrix4f matrix_icp = icp.getFinalTransformation();
      std::cout << matrix_icp << std::endl;

      // transformpointcloud
      transformPointCloud(*transformed_cloud, *transformed_cloud, matrix_icp);

      cout << "Después de la transformación (ICP)." << endl;

      //////////////////////////////////////////////////
      // Trabajar siempre sobre la nube transformada. //
      //////////////////////////////////////////////////

      *cloud_ant = *transformed_cloud;
      filter_cloud(transformed_cloud, transformed_cloud);
      std::cout << "Nº de puntos de la nube filtrada antes de añadir a mapa: " << transformed_cloud->points.size() << std::endl;
      //TODO REVISAR ESTO...
      //swap(cloud_ant,cloud);
      *mapa += *transformed_cloud;
      std::cout << "Nº de puntos total en mapa: " << mapa->points.size() << std::endl;
    }
  }
  //Volcado de actual a anterior.
  *cloudDescriptors_ant = *cloudDescriptors;
  *pcKeyPoints_antXYZ = *pcKeyPoints_XYZ;
  //*normals_ant = *normals;

}

  //TODO
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_map (new pcl::PointCloud<pcl::PointXYZRGB>);
  //vGrid.setInputCloud (mapa);
  //vGrid.filter (*cloud_map);
  //cout << "Puntos del mapa antes del filtrado y de la visualización: " << mapa->points.size() << endl;
  //cout << "Puntos tras VG_mapa antes de la visualización: " << cloud_map->size() << endl;


// Estas dos funciones son para el codigo de prueba con las dos nubes
void simpleVisPrueba(PointCloud<pcl::PointXYZRGB>::Ptr cloud_prueba_1){
    //pcl::visualization::CloudViewer viewer ("Cloud Viewer");
    pcl::visualization::CloudViewer viewer ("Cloud_1 Viewer");
    //pcl::visualization::CloudViewer viewer_2 ("Cloud_2 Viewer");
    while(!viewer.wasStopped()){
       viewer.showCloud (cloud_prueba_1);
       //viewer_2.showCloud (cloud_2);
       boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
    }
}

// Estas dos funciones son para el codigo de prueba con las dos nubes
void simpleMov(){
    char cmd[50];
    //ros::ServiceClient client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
    gazebo_msgs::SetModelState setmodelstate;
    gazebo_msgs::ModelState modelstate;
    while(ros::ok()){
       boost::this_thread::sleep(boost::posix_time::milliseconds(3000));
       modelstate.model_name = "mobile_base";
       modelstate.reference_frame = "world";
       modelstate.pose.position.x = 0;
       modelstate.pose.position.y = 0;
       modelstate.pose.orientation.z = 0;
       setmodelstate.request.model_state = modelstate;
       client.call(setmodelstate);

       boost::this_thread::sleep(boost::posix_time::milliseconds(100));
       std::cin.getline(cmd, 50);

       if(cmd[0]!='w' && cmd[0]!='a' && cmd[0]!='s' && cmd[0]!='d') {
           std::cout << "Comando desconocido:" << cmd << "\n";
           continue;
       }
       modelstate.pose.position.x = modelstate.pose.position.y = modelstate.pose.orientation.z = 0;
       //move forward
       if(cmd[0]=='w') {
           modelstate.pose.position.x = 0.25;
       }
       //turn left (yaw) and drive forward at the same time
       else if(cmd[0]=='a') {
           //modelstate.angular.z = 0.75;
           modelstate.pose.position.x = 0.25;
       }
       //turn right (yaw) and drive forward at the same time
       else if(cmd[0]=='d') {
           //modelstate.angular.z = -0.75;
           modelstate.pose.position.x = 0.25;
       }
       //quit
       else if(cmd[0]=='s') {
         modelstate.pose.position.x = -0.25;
       }
       modelstate.model_name = "mobile_base";
       modelstate.reference_frame = "world";
       setmodelstate.request.model_state = modelstate;
       client.call(setmodelstate);
       std::cout << "Ciclo:" << cmd << "\n";
    }
}

void unirPuntos(PointCloud<pcl::PointXYZRGB>::Ptr cloud_prueba_1, PointCloud<pcl::PointXYZRGB>::Ptr cloud_prueba_2){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      cloud_filtered_1   (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      pcKeyPoints_XYZ_1  (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PFHSignature125>::Ptr  cloudDescriptors_1 (new pcl::PointCloud<pcl::PFHSignature125>);
  std::vector<int> indices;

  pcl::PointCloud<pcl::Normal>::Ptr normals_1 (new pcl::PointCloud<pcl::Normal>);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      cloud_filtered_2   (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr      pcKeyPoints_XYZ_2  (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PFHSignature125>::Ptr  cloudDescriptors_2 (new pcl::PointCloud<pcl::PFHSignature125>);
  std::vector<int> indices_2;

  pcl::PointCloud<pcl::Normal>::Ptr normals_2 (new pcl::PointCloud<pcl::Normal>);



  filter_cloud(cloud_prueba_1, cloud_filtered_1);
  filter_cloud(cloud_prueba_2, cloud_filtered_2);
  //*cloud_filtered = *cloud;
  cout << "Puntos tras VG: " << cloud_filtered_1->size() << endl;
  cout << "Puntos tras VG: " << cloud_filtered_2->size() << endl;
  //Eliminado de los NaN de la nube filtrada actual.      -> cloud_filtered
  removeNaNFromPointCloud<PointXYZRGB>(*cloud_filtered_1, *cloud_filtered_1, indices);
  removeNaNFromPointCloud<PointXYZRGB>(*cloud_filtered_2, *cloud_filtered_2, indices_2);
  cout << "Quitamos los NAN y quedan: " << cloud_filtered_1->size() << endl;
  cout << "Quitamos los NAN y quedan: " << cloud_filtered_2->size() << endl;
  //Detección de características                          -> pcKeyPoints_XYZ
  HARRISdetect_keypoints(cloud_filtered_1, *pcKeyPoints_XYZ_1);
  HARRISdetect_keypoints(cloud_filtered_2, *pcKeyPoints_XYZ_2);
  //Si detectamos un número de caracteristicas suficientes...
  if(pcKeyPoints_XYZ_1->size() > 10 && pcKeyPoints_XYZ_2->size()>10){
    cout << "Paso por el if" << endl;
    //Cálculo de normales a la superficie.                -> normals
    compute_surface_normals(pcKeyPoints_XYZ_1, normal_radius, normals_1);
    compute_surface_normals(pcKeyPoints_XYZ_2, normal_radius, normals_2);
    //Extracción de características.                      -> cloudDescriptors
    PFHRGB(cloud_filtered_1, normals_1, pcKeyPoints_XYZ_1, feature_radius, *cloudDescriptors_1);
    PFHRGB(cloud_filtered_2, normals_2, pcKeyPoints_XYZ_2, feature_radius, *cloudDescriptors_2);
    std::cout << "Nº of PFH points in the descriptors_cloud_filtered are " << cloudDescriptors_1->points.size() << std::endl;
    std::cout << "Nº of PFH points in the descriptors_cloud_filtered are " << cloudDescriptors_2->points.size() << std::endl;

    ///////////////////////////////////////////
    // CorrespondenceRejactorSampleConsensus //
    ///////////////////////////////////////////
    //Determinación de correspondencias por CorrespondenceRejactorSampleConsensus.     -> transform_res_from_SAC
    registration::CorrespondenceEstimation<PFHSignature125,PFHSignature125> corr_est;
    corr_est.setInputSource(cloudDescriptors_2);
    corr_est.setInputTarget(cloudDescriptors_1);

    cout << "Antes de determinar las correspondencias." << endl;

    boost::shared_ptr<Correspondences> correspondences (new Correspondences);
    //corr_est.determineCorrespondences (*correspondences);
    corr_est.determineCorrespondences (*correspondences);

    cout << "Ya se han determinado las correspondencias." << endl;

    boost::shared_ptr<Correspondences> correspondences_result_rej_sac (new Correspondences);
    registration::CorrespondenceRejectorSampleConsensus<PointXYZRGB> corr_rej_sac;
    corr_rej_sac.setInputSource(pcKeyPoints_XYZ_2);
    corr_rej_sac.setInputTarget(pcKeyPoints_XYZ_1);
    // ransac
    corr_rej_sac.setInlierThreshold(0.1);
    corr_rej_sac.setMaximumIterations(1000);
    corr_rej_sac.setInputCorrespondences(correspondences);
    corr_rej_sac.getCorrespondences(*correspondences_result_rej_sac);

    Eigen::Matrix4f transform_res_from_SAC = corr_rej_sac.getBestTransformation();

    cout << "Después de todo el lío nos quedamos con: " << correspondences->size() << " ó: " << correspondences_result_rej_sac->size() << " correspondencias." << endl;
    cout << "Matriz de transformación por Sample Consensus: " << endl;
    cout << transform_res_from_SAC << endl;

    ///////////////////////////////////////////////////
    // Fin del CorrespondenceRejactorSampleConsensus //
    ///////////////////////////////////////////////////


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
    //Aplicar la transformación a la nube de puntos filtrada.         -> transformed_cloud
    transformPointCloud(*cloud_filtered_1, *transformed_cloud, transform_res_from_SAC);

    cout << "Después de la transformación." << endl;


    //////////////////////////////////////////////////
    // Trabajar siempre sobre la nube transformada. //
    //////////////////////////////////////////////////
    *cloud_prueba_1 += *transformed_cloud;

    cout << "llamada a slimplevis";
    simpleVisPrueba(cloud_prueba_1);
  }
  else{
    cout << "ERROR!" << endl;
  }
}

// Fin de las funciones para probar con dos nubes de puntos

/*
void callbackStates(const gazebo_msgs::ModelStates& msg){
  int i = -1;
  while (i<1000) {
    if(msg.name[++i] == "mobile_base") break;
  }
  //ros::geometry_msgs/Pose[i] pos = msg->pose;
  gazebo_msgs::ModelState msgcopy = new gazebo_msgs::ModelState::Ptr();
  msgcopy.model_name = "mobile_base";
  msgcopy.pose.position.y = msg[i].pose.position.y;
  msgcopy.pose.position.x = msg[i].pose.position.x;
  msgcopy.pose.position.z = msg[i].pose.position.z;
  //msgcopy.twist = *msg->twist[i];
  msgcopy.reference_frame = "world";
}
*/

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  //_nh = nh;
  //ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZRGB> >("/camera/depth/points", 1, callback);
  // Descomentar para teleoperar
  //ros::Publisher cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/teleop", 1);
  boost::thread t(simpleVis);
  client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  boost::thread t2(simpleMov);

  //ros::ServiceClient client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  //gazebo_msgs::SetModelState setmodelstate;
  //gazebo_msgs::ModelState modelstate;
  //modelstate.model_name = "mobile_base";
  //modelstate.twist.angular.z = 0.1;
  ros::ServiceClient client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  gazebo_msgs::SetModelState setmodelstate;
  gazebo_msgs::ModelState modelstate;
  //modelstate.twist.angular.z = 0.03;
  //modelstate.pose.orientation.z = 0;
  //setmodelstate.request.model_state = modelstate;
  //client.call(setmodelstate);

  //ros::Subscriber submodel = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, callbackStates);

  modelstate.model_name = "mobile_base";
  modelstate.reference_frame = "world";
  modelstate.pose.position.x = 0;
  modelstate.pose.orientation.z = 0;
  setmodelstate.request.model_state = modelstate;
  client.call(setmodelstate);

  while(ros::ok())
  {
    //modelstate = submodel.getModelStates();
    //modelstate.pose.orientation.x  = 0;
    //modelstate.pose.orientation.y  = 0;


    /*
    modelstate.model_name = "mobile_base";
    modelstate.reference_frame = "world";
    modelstate.pose.position.x += 0.002;
    modelstate.pose.orientation.z += 0.002;
    setmodelstate.request.model_state = modelstate;
    if (client.call(setmodelstate))
    {
      ROS_INFO("BRILLIANT!!!");
      ROS_INFO("%f",modelstate.pose.position.x);
    }
    else
    {
      ROS_ERROR("Failed to call service ");
      return 1;
    }
    */


    // Esto funciona pero habria que buscar la manera de hacerlo solo cuando queramos y no siempre
	  //driveKeyboard(cmd_vel_pub_);
    ros::spinOnce();
    //cout << "__________________________________________________________\n";
    //viewer->spinOnce(1);


    ///////////////////
    //   Pendiente:  //
    ///////////////////
    //- Generar fichero de recorrido por las estancias, para ir leyendo las posiciones y rotaciones desde las que se irán obteniendo las nubes de puntos.
    //- Volcado de mapa final a fichero para que los profesores puedan cargarlo con un programa externo que les proporcionaremos.

    ///////////////
    //   Notas:  //
    ///////////////
    //- Posiblemente tengamos que trabajar con nubes en crudo, y una vez se hayan obtenido las coincidencias entre ellas, almacenar la nube filtrada y transformada en el mapa.
    //- Creo que sería conveniente realizar un "filtrado de puntos repetidos", desde la nube transformada a el mapa, para que este contenga el mínimo posible de puntos repetidos.


    //////////////////////////////////////////////
    //  Fases para la construcción de mapa 3D.  //
    //////////////////////////////////////////////


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

  // Fin codigo ppal

  // Probando con dos nubes solo
  // Para probar con las dos nubes, copiar lo que hay dentro de la carpeta nubes en la raiz del catkin_ws,comentar todo lo anterior y descomentar esta parte
  /*
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_prueba_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_prueba_2 (new pcl::PointCloud<pcl::PointXYZRGB>);


  if ((pcl::io::loadPCDFile<pcl::PointXYZRGB> ("test_1.pcd", *cloud_prueba_1) == -1) ||
      (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("test_2.pcd", *cloud_prueba_2) == -1)) //* load the file
  {
    PCL_ERROR ("Couldn't read file test_1.pcd or test_2.pcd \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud_prueba_1->width * cloud_prueba_1->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;

  std::cout << "Loaded "
            << cloud_prueba_2->width * cloud_prueba_2->height
            << " data points from test_pcd.pcd with the following fields: "
            << std::endl;

  unirPuntos(cloud_prueba_1,cloud_prueba_2);
  */

  // Fin codigo de pureba de dos nubes
}
