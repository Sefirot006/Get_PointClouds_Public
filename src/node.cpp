// NARF KeyPoints
// http://pointclouds.org/documentation/tutorials/narf_keypoint_extraction.php#narf-keypoint-extraction
// Importante
// http://www.jeffdelmerico.com/wp-content/uploads/2014/03/pcl_tutorial.pdf

#include <boost/foreach.hpp>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/ModelStates.h>
#include <geometry_msgs/Twist.h>
#include <pcl-1.7/pcl/keypoints/harris_3d.h>
#include <pcl/common/io.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
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
#include <pcl/registration/icp.h>
//#include <pcl/registration/transformation_estimation_lm.h>
//#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

const float normal_radius = 0.05f;
const float feature_radius = 0.05f;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr mapa (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ant (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcKeyPoints_antXYZ (new pcl::PointCloud<pcl::PointXYZRGB>);
//pcl::PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints_ant (new pcl::PointCloud<pcl::PointWithScale>);
//pcl::PointCloud<pcl::Normal>::Ptr normals_ant (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::PFHSignature125>::Ptr cloudDescriptors_ant (new pcl::PointCloud<pcl::PFHSignature125>);

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
          float feature_radius, PointCloud<PFHSignature125> &descriptors_out){
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
          float feature_radius, PointCloud<PFHSignature125> &descriptors_out){
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

  cout << "Saliendo del pfh" << endl;
}

void FPFH(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals, PointCloud<PointXYZRGB>::Ptr &keypoints,
          float feature_radius, PointCloud<FPFHSignature33> &descriptors_out)
{
  FPFHEstimation<PointXYZRGB, Normal, FPFHSignature33> fpfh;
  fpfh.setInputCloud (points);
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

  cout << "Saliendo del pfh" << endl;
  // fpfhs->points.size () should have the same size as the input cloud->points.size ()*

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
  pcl::PointCloud<pcl::PFHSignature125>::Ptr  cloudDescriptors (new pcl::PointCloud<pcl::PFHSignature125>);
  //pcl::PointCloud<pcl::PointWithScale>::Ptr pcKeyPoints      (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  std::vector<int> indices;
  //pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;

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


  ////////////////////////////////////////////////
  // De esta forma se repiten menos operaciones //
  // por lo que si conseguimos que funcione,    //
  // lo hará mucho más rápido y más ligero.     //
  ////////////////////////////////////////////////
  //Lectura de la nube actual.                            -> cloud
  copyPointCloud(*msg, *cloud);
  cout << "Puntos capturados: " << cloud->size() << endl;
  //Filtrado de la nube actual.                           -> cloud_filtered
  //filter_cloud(cloud, cloud_filtered);
  *cloud_filtered = *cloud;
  cout << "Puntos tras VG: " << cloud_filtered->size() << endl;
  //Eliminado de los NaN de la nube filtrada actual.      -> cloud_filtered
  removeNaNFromPointCloud<PointXYZRGB>(*cloud_filtered, *cloud_filtered, indices);
  cout << "Quitamos los NAN y quedan: " << cloud_filtered->size() << endl;
  //Detección de características                          -> pcKeyPoints_XYZ
  HARRISdetect_keypoints(cloud_filtered, *pcKeyPoints_XYZ);
  //Si detectamos un número de caracteristicas suficientes...
  if(pcKeyPoints_XYZ->size() > 10){
    cout << "Paso por el if" << endl;
    //Cálculo de normales a la superficie.                -> normals
    //compute_surface_normals(pcKeyPoints_XYZ, normal_radius, normals);
    compute_surface_normals(cloud_filtered, normal_radius, normals);
    //Extracción de características.                      -> cloudDescriptors
    PFHRGB(cloud_filtered, normals, pcKeyPoints_XYZ, feature_radius, *cloudDescriptors);
    std::cout << "Nº of PFH points in the descriptors_cloud_filtered are " << cloudDescriptors->points.size() << std::endl;
  }

  //Si es la primera nube...
  if(empieza==true){
    cout << "Es primera nube." << endl;
    copyPointCloud(*cloud_filtered, *cloud_ant);
    filter_cloud(cloud, cloud_filtered);
    copyPointCloud(*cloud_filtered, *mapa);
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
      std::cout << "Nº of PFH points in the descriptors_cloud_filtered_ant are " << cloudDescriptors_ant->points.size() << std::endl;

      
      ///////////////////////////////////////////
      // CorrespondenceRejactorSampleConsensus //
      ///////////////////////////////////////////
      //Determinación de correspondencias por CorrespondenceRejactorSampleConsensus.     -> transform_res_from_SAC
      registration::CorrespondenceEstimation<PFHSignature125,PFHSignature125> corr_est;
      corr_est.setInputSource(cloudDescriptors);
      corr_est.setInputTarget(cloudDescriptors_ant);

      cout << "Antes de determinar las correspondencias." << endl;

      boost::shared_ptr<Correspondences> correspondences (new Correspondences);
      //corr_est.determineCorrespondences (*correspondences);
      corr_est.determineReciprocalCorrespondences (*correspondences);

      cout << "Ya se han determinado las correspondencias." << endl;

      boost::shared_ptr<Correspondences> correspondences_result_rej_sac (new Correspondences);
      registration::CorrespondenceRejectorSampleConsensus<PointXYZRGB> corr_rej_sac;
      corr_rej_sac.setInputSource(pcKeyPoints_XYZ);
      corr_rej_sac.setInputTarget(pcKeyPoints_antXYZ);
      // ransac
      corr_rej_sac.setInlierThreshold(0.020);
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
      transformPointCloud(*cloud_filtered, *transformed_cloud, transform_res_from_SAC);

      cout << "Después de la transformación (RANSAC)." << endl;

      /*
      // nuevo ICP
      IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
      icp.setInputSource(transformed_cloud);
      icp.setInputTarget(cloud_ant);

      PointCloud<PointXYZRGB> final;
      icp.align(final);
      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      std::cout << icp.getFinalTransformation() << std::endl;
      Eigen::Matrix4f matrix_icp = icp.getFinalTransformation();


      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud_2 (new pcl::PointCloud<pcl::PointXYZRGB> ());
      // transformpointcloud
      transformPointCloud(*cloud,*transformed_cloud_2,matrix_icp);

      cout << "Despues de la transformacion 2" << endl;
      */

      //Recogemos la nube transformada desde RANSAC.
      // nuevo ICP
      IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
      //icp.setInputSource(cloud_filtered);
      //icp.setInputSource(transformed_cloud);
      //icp.setInputTarget(cloud_ant);
      icp.setInputSource(pcKeyPoints_XYZ);
      icp.setInputTarget(pcKeyPoints_antXYZ);
      //icp.setInputTarget(mapa);

      PointCloud<PointXYZRGB> final;
      icp.align(final);
      std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
      std::cout << icp.getFinalTransformation() << std::endl;
      Eigen::Matrix4f matrix_icp = icp.getFinalTransformation();

      //pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
      // transformpointcloud
      transformPointCloud(*transformed_cloud, *transformed_cloud, matrix_icp);
      //cout << "Matriz de transformación por ICP: " << endl;
      //cout << matrix_icp << endl;

      cout << "Después de la transformación (ICP)." << endl;

      //////////////////////////////////////////////////
      // Trabajar siempre sobre la nube transformada. //
      //////////////////////////////////////////////////

      *cloud_ant = *transformed_cloud;
      filter_cloud(transformed_cloud, transformed_cloud);

      *mapa += *transformed_cloud;

  //TODO REVISAR ESTO...
      //ANTES:
      //swap(cloud_ant,cloud);
      //AHORA:
    }
  }
  //Volcado de actual a anterior.
  *cloudDescriptors_ant = *cloudDescriptors;
  *pcKeyPoints_antXYZ = *pcKeyPoints_XYZ;
  //*normals_ant = *normals;

}


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
    corr_est.determineReciprocalCorrespondences (*correspondences);

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


void callbackStates(const gazebo_msgs::ModelStates::ConstPtr& msg){

}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "sub_pcl");

  double rotation = 0.0;

  int i = 0;
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZRGB> >("/camera/depth/points", 1, callback);
  // Descomentar para teleoperar
  //ros::Publisher cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/teleop", 1);
  boost::thread t(simpleVis);

  //ros::ServiceClient client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  //gazebo_msgs::SetModelState setmodelstate;
  //gazebo_msgs::ModelState modelstate;
  //modelstate.model_name = "mobile_base";
  //modelstate.twist.angular.z = 0.1;
  ros::ServiceClient client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  gazebo_msgs::SetModelState setmodelstate;
  gazebo_msgs::ModelState modelstate;
  modelstate.model_name = "mobile_base";
  //modelstate.twist.angular.z = 0.03;
  //modelstate.pose.orientation.z = 0;
  //setmodelstate.request.model_state = modelstate;
  //client.call(setmodelstate);

  ros::Subscriber submodel = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, callbackStates);

  while(ros::ok())
  {
    //modelstate.pose.orientation.x  = 0;
    //modelstate.pose.orientation.y  = 0;
    modelstate.pose.orientation.z += 0.2;
    setmodelstate.request.model_state = modelstate;
    //client.call(setmodelstate);

    // Esto funciona pero habria que buscar la manera de hacerlo solo cuando queramos y no siempre
	  //driveKeyboard(cmd_vel_pub_);
    ros::spinOnce();
    cout << "__________________________________________________________\n";
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
