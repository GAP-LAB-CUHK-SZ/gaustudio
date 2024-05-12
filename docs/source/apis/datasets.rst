.. _`Using existing data`:

Using existing data
===================================


The field of 3D computer vision has seen a rapid proliferation of diverse datasets in recent years. These datasets play a 
crucial role in driving research and innovation, serving as benchmarks for evaluating algorithms and models. However, the 
sheer volume and variety of available datasets can be overwhelming, making it challenging for researchers to navigate and 
identify the most suitable ones for their tasks.

A significant challenge is the time-consuming and error-prone process of processing raw data and writing custom dataloaders 
for each dataset. This overhead can distract from the core focus of the research, slowing down the pace of innovation.

To address this, GauStudio aims to provide a curated overview of the most commonly used 3D data repositories and datasets, 
with three key contributions:


**Comprehensive Dataset Collection** : The wide range of categories covered, from object-level to indoor and outdoor 
scene datasets, ensures that researchers can easily find the most relevant datasets for their 3D computer vision tasks.

**Seamless Data Integration**: The custom dataloaders designed for each dataset provide a unified interface 
``gaustudio.datasets.make(name='...',source_path='...')`` that abstracts away the complexity of dataset-specific 
preprocessing. This allows researchers to focus on their core research without getting bogged down in the details of data 
loading and formatting.

**Colmap-format Preprocessing**: Colmap is a widely-used structure-from-motion and multi-view stereo pipeline.  We 
have pre-processed many datasets in this standardized format to reduce the time and effort required for data preprocessing, 
freeing up researchers to concentrate on their primary research objectives.

The GauStudio's unified dataset interface can be used not only for evaluating multi-view stereo (MVS) and novel-view 
synthesis (NVS) task, but also for **pre-training large-scale 3D vision models from diverse data sources.** This broad 
applicability further enhances the value of the GauStudio project for the 3D computer vision research community.


Object Datasets for Benchmarking
-------------------------------------

For non-colmap format dataset, such as synthetic dataset (nerf, refnerf, nero, NSVF), and reconstruction dataset 
(DTU, BlendedMVS, and MobileBrick), we provide preprocessed versions.
`https://pan.baidu.com/s/1QzoEDPr-SFjid-EMhc-Xng?pwd=8iwa <https://pan.baidu.com/s/1QzoEDPr-SFjid-EMhc-Xng?pwd=8iwa>`_

.. list-table::
   :header-rows: 1

   * - Name
     - Brief Description
     - Statistics
     - Download Link
     - How to Use
   * - `nerf_synthetic <https://github.com/bmild/nerf>`_
     - a synthetic dataset that exhibit complicated geometry and realistic non-Lambertian materials.
     - 8 scenes, 100 ~ 200 views per scene
     - `https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 <https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1>`_
     - gaustudio.datasets.make(name='nerf', source_path='...')
   * - `refnerf_synthetic <https://dorverbin.github.io/refnerf/>`_
     - a dataset of shiny objects rendered in Blender
     - 6 scenes,  100 ~ 200 views per scene
     - `https://storage.googleapis.com/gresearch/refraw360/ref.zip <https://storage.googleapis.com/gresearch/refraw360/ref.zip>`_
     - gaustudio.datasets.make(name='nerf', source_path='...')
   * - `nero_synthetic <https://liuyuan-pal.github.io/NeRO/>`_
     - a dataset of glossy objects rendered in Blender
     - 8 scenes, 128 views per scene
     - `https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EvNz_o6SuE1MsXeVyB0VoQ0B9zL8NZXjQQg0KknIh6RKjQ?e=MaonKe <https://connecthkuhk-my.sharepoint.com/>`_
     - gaustudio.datasets.make(name='nero', source_path='...')
   * - `NSVF_synthetic <https://lingjie0206.github.io/papers/NSVF/>`_
     - a synthetic dataset with more complex geometry and lighting effects
     - 8 scenes, 100 views per scene
     - `https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip <https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip>`_
     - gaustudio.datasets.make(name='nsvf', source_path='...')
   * - `DTU <https://roboimagedata.compute.dtu.dk/>`_
     - a multi-view stereo dataset, which is an order of magnitude larger in number of scenes and with a significant increase in diversity
     - 80 scenes, 49 ~ 64 views per scene
     - `https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view <https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view>`_
     - gaustudio.datasets.make(name='mvsnet', source_path='...')
   * - `BlendedMVS <https://github.com/YoYo000/BlendedMVS>`_
     - a large-scale dataset, to provide sufficient training ground truth for learning-based MVS
     - 113 scenes, 20 ~ 1,000 views per scene
     - `https://drive.google.com/open?id=1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb <https://drive.google.com/open?id=1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb>`_
     - gaustudio.datasets.make(name='mvsnet', source_path='...')
   * - `mobilebrick <https://code.active.vision/MobileBrick/>`_
     - a novel data capturing and 3D annotation pipeline to obtain precise 3D ground-truth shapes without relying on expensive 3D scanners
     - 18 scenes, 40~150 views per scene
     - `https://www.robots.ox.ac.uk/~victor/data/MobileBrick/MobileBrick_Mar23.zip <https://www.robots.ox.ac.uk/>`_
     - gaustudio.datasets.make(name='mobilebrick', source_path='...')
   * - deepvoxels
     - a dataset of high-quality 3D scans
     - 4 scenes, 400~500 views per scene
     - `https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH <https://drive.google.com/drive/folders/1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH>`_
     - gaustudio.datasets.make(name='deepvoxels', source_path='...')
   * - `MipNerf360 <https://jonbarron.info/mipnerf360/>`_
     - a dataset of object captured in indoor scene and outdoor scenes
     - 9 scenes, 100~300 views
     - `http://storage.googleapis.com/gresearch/refraw360/360_v2.zip <http://storage.googleapis.com/gresearch/refraw360/360_v2.zip>`_
     - gaustudio.datasets.make(name='colmap', source_path='...')
   * - `TanksAndTemples <https://github.com/isl-org/TanksAndTemples>`_
     - A benchmark for image-based 3D reconstruction
     - 21 scenes, 100~300 views
     - `https://www.tanksandtemples.org/download/ <https://www.tanksandtemples.org/download/>`_
     - gaustudio.datasets.make(name='colmap', source_path='...')
   * - nero_real
     - a dataset of glossy objects captured in real scene
     - 5 scenes, 100-130 views per scene
     - `https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EvNz_o6SuE1MsXeVyB0VoQ0B9zL8NZXjQQg0KknIh6RKjQ?e=MaonKe <https://connecthkuhk-my.sharepoint.com/>`_
     - gaustudio.datasets.make(name='colmap', source_path='...')
   * - `refnerf_real <https://dorverbin.github.io/refnerf/>`_
     - a dataset of  shiny objects captured in real scene
     - 3 scenes,  100 ~ 200 views per scene
     - `https://storage.googleapis.com/gresearch/refraw360/ref_real.zip <https://storage.googleapis.com/gresearch/refraw360/ref_real.zip>`_
     - gaustudio.datasets.make(name='colmap', source_path='...')
   * - `Standford-ORB <https://stanfordorb.github.io/>`_
     - a new real-world 3D Object inverse Rendering Benchmark.
     - 42 scenes, 70 views per scene
     - `https://downloads.cs.stanford.edu/viscam/StanfordORB/llff_colmap_LDR.tar.gz <https://downloads.cs.stanford.edu/viscam/StanfordORB/llff_colmap_LDR.tar.gz>`_
     - gaustudio.datasets.make(name='colmap', source_path='...')

Object Datasets for Training (Large Reconstruction Model, Gaussian Completion Model...)
--------------------------------------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Name
     - Brief Description
     - Statistics
     - Download Link
     - How to Use
   * - `MVImgNet <https://gaplab.cuhk.edu.cn/projects/MVImgNet/>`_
     - a large-scale dataset of multi-view images
     - 219,188 scenes, 25~30 views per scene
     - `https://docs.google.com/forms/d/e/1FAIpQLSfU9BkV1hY3r75n5rc37IvlzaK2VFYbdsvohqPGAjb2YWIbUg/viewform <https://docs.google.com/forms/d/e/1FAIpQLSfU9BkV1hY3r75n5rc37IvlzaK2VFYbdsvohqPGAjb2YWIbUg/viewform>`_
     -
   * - CO3D
     - a multi-view images dataset of common object categories
     - 18,619 scenes, 100 views per scene
     - `https://scontent-nrt1-2.xx.fbcdn.net/m1/v/t6/An_tlCbE1hnVIBR2LJJWNbGINO9Jj5_Rmu9KGNdrDm_PoQ4xY3WuRbDIIfdKeiiBcgb8vJ0.txt?ccb=10-5&oh=00_AfBYv11mvow85Rgx0BObQWmqyo2IemZrYKj3Vb2gGlRKXQ&oe=66372FF4&_nc_sid=ba4296 <https://scontent-nrt1-2.xx.fbcdn.net/m1/v/t6/An_tlCbE1hnVIBR2LJJWNbGINO9Jj5_Rmu9KGNdrDm_PoQ4xY3WuRbDIIfdKeiiBcgb8vJ0.txt?ccb=10-5&oh=00_AfBYv11mvow85Rgx0BObQWmqyo2IemZrYKj3Vb2gGlRKXQ&oe=66372FF4&_nc_sid=ba4296>`_
     -
   * - `WildRGB-D <https://wildrgbd.github.io/>`_
     - a RGB-D object dataset captured in the wild
     - 23,049 scenes
     - `https://huggingface.co/hongchi/wildrgbd <https://huggingface.co/hongchi/wildrgbd>`_
     -
   * - Objectron
     - a dataset of short object centric video clips with pose annotations
     - 15,000 scenes
     - `https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Download%20Data.ipynb <https://github.com/google-research-datasets/Objectron/blob/master/notebooks/Download%20Data.ipynb>`_
     -
   * - RTMV
     - A Ray-Traced Multi-View Synthetic Dataset
     - 2,000 scenes, 20 ~ 50 views per scene
     - `https://drive.google.com/drive/folders/1cUXxUp6g25WwzHnm_491zNJJ4T7R_fum <https://drive.google.com/drive/folders/1cUXxUp6g25WwzHnm_491zNJJ4T7R_fum>`_
     -
   * - `OmniObject3D <https://omniobject3d.github.io/>`_
     - a large vocabulary 3D object dataset with massive high-quality real-scanned 3D objects.
     - 6,000 scenes, 200 views per scene
     - `https://openxlab.org.cn/datasets/OpenXDLab/OmniObject3D-New <https://openxlab.org.cn/datasets/OpenXDLab/OmniObject3D-New>`_
     -
   * - `ABO <https://amazon-berkeley-objects.s3.amazonaws.com/index.html>`_
     - a large-scale dataset designed for material prediction and multi-view retrieval experiments
     - 7,953 scenes, 30 views per scene
     - `https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-benchmark-material.tar <https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-benchmark-material.tar>`_
     -
   * - `G-buffer Objaverse <https://aigc3d.github.io/gobjaverse/>`_
     - a synthetic dataset renderer on Objaverse
     - 270,000 scenes
     - `https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse <https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse>`_
     -

Outdoor Datasets
----------------------

.. list-table::
   :header-rows: 1

   * - **Name**
     - **Brief Description**
     - **Statistics**
     - **Download Link**
     - **How to Use**
   * - LLFF
     - The dataset contains three parts: Diffuse Synthetic 360◦, Realistic Synthetic 360◦ and Real LLFF. Diffuse Synthetic 360◦ consists of four Lambertian objects with simple geometry. Realistic Synthetic 360◦ consists of eight objects of complicated geometry. The real images of complex scenes consist of 8 forward-facing scenes captured with a cellphone at a size of 1008x756 pixels.
     - 24 scenes
     - `https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 <https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1>`_
     - 
   * - ibrnet
     - For synthetic data, we generate object-centric renderings of the 1,023 models in Google Scanned Objects. For real data, we use RealEstate10K, the Spaces dataset, and 102 real scenes from handheld cellphone captures (35 from LLFF and 67 from ourselves).
     - 67 scenes, 20-60 view per scene
     - `https://drive.google.com/drive/folders/1qfcPffMy8-rmZjbapLAtdrKwg3AV-NJe <https://drive.google.com/drive/folders/1qfcPffMy8-rmZjbapLAtdrKwg3AV-NJe>`_
     - 
   * - Shiny Dataset
     - Shiny dataset contains captured with a smartphone in a similar manner as Real Forward-Facing dataset. However, the scenes contain much more challenging view-dependent effects, such as the rainbow reflections on a CD, refraction through a liquid bottle or a magnifying glass, metallic and ceramic reflections, and sharp specular highlights on silverware, as well as detailed thin structures.
     - 8 scenes
     - `https://vistec-my.sharepoint.com/personal/pakkapon_p_s19_vistec_ac_th/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpakkapon%5Fp%5Fs19%5Fvistec%5Fac%5Fth%2FDocuments%2Fpublic%2FVLL%2FNeX%2Fshiny%5Fdatasets&ga=1 <https://vistec-my.sharepoint.com/personal/pakkapon_p_s19_vistec_ac_th/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fpakkapon%5Fp%5Fs19%5Fvistec%5Fac%5Fth%2FDocuments%2Fpublic%2FVLL%2FNeX%2Fshiny%5Fdatasets&ga=1>`_
     - 
   * - DeepView
     - DeepView captures a Spaces Dataset consists of 100 scenes of data for training and testing. The data was collected with a 16 camera rig. For each scene we captured between 3 and 10 different rig positions. The rig positions are all relatively close together, so one rig position can be used as the input to a model, with an image from a different rig position used as the target image.
     - 100 scenes
     - `https://github.com/augmentedperception/spaces_dataset/tree/master <https://github.com/augmentedperception/spaces_dataset/tree/master>`_
     - 
   * - Deblur-NeRF
     - Deblur-NeRF synthesizes 5 scenes using Blender with camera motion blur and defocus blur. It also contains  20 real world scenes with 10 scenes for each blur type for a qualitative study.
     - 25 scenes
     - `https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx <https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx>`_\ _
     - 
   * - LaMAR
     - LaMAR is a benchmark dataset for localization and mapping in AR. It captures three diverse and large-scale scenes recorded with head-mounted and hand-held AR devices.
     - 3 scenes
     - https://lamar.ethz.ch/
     - 
   * - UrbanScene3D
     - The UrbanScene3D provides 10 synthetic and 6 real-world scenes with CAD and reconstructed mesh models and the corresponding aerial images. It contains over 128k high-resolution images covering 16 scenes, including largescale real urban regions and synthetic cities with 136 km2 area in total.
     - 5 scenes
     - `https://www.dropbox.com/scl/fo/ajgcxsec1cojohdvn3dfd/h?rlkey=2tnehx3ixc3ue7nne4yvflhd0&dl=0https://github.com/Linxius/UrbanScene3D?tab=readme-ov-file#urbanscene3d-v1 <https://www.dropbox.com/scl/fo/ajgcxsec1cojohdvn3dfd/h?rlkey=2tnehx3ixc3ue7nne4yvflhd0&dl=0https://github.com/Linxius/UrbanScene3D?tab=readme-ov-file#urbanscene3d-v1>`_
     - 
   * - Mill-19
     - The Mill-19 dataset contains high resolution drone captured images for 2 scenes with given camera poses.
     - 2 scenes
     - `https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgzhttps://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz <https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgzhttps://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz>`_
     - 
   * - DL3DV-10K
     - The DL3DV-10k contains 10510 different scenes with consistent capture standards at 60 fps and 4K resolution. It has 96 complexity categories to cover real-world complexities
     - 10510 scenes
     - `https://github.com/DL3DV-10K/Dataset?tab=readme-ov-file#dataset-download <https://github.com/DL3DV-10K/Dataset?tab=readme-ov-file#dataset-download>`_
     - 
   * - MatrtixCity
     - The MatrixCity dataset contains  67k  aerial images and 452k street images with Ground Truth pose synthesised by UE5  from two city maps of total size 28km2.
     - 2 scenes
     - `https://huggingface.co/datasets/BoDai/MatrixCity/tree/main <https://huggingface.co/datasets/BoDai/MatrixCity/tree/main>`_
     - 
   * - Quad 6k
     - The Quad 6k dataset is a drone captured dataset containing 6514 images of the Arts Quad at Cornell University.
     - 1 scene
     - `https://vision.soic.indiana.edu/projects/disco/ <https://vision.soic.indiana.edu/projects/disco/>`_
     - 
   * - Waymo
     - The Perception dataset of Waymo open dataset is a street view dataset that contains high resolution sensor data for 2,030 scenes.
     - 2030 scenes
     - `https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_3_0 <https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_3_0>`_
     - 
   * - STPLS3d
     - STPLS3D is a drone captured dataset that provides point cloud and precise semantic and instance annotations.
     - 1 scene
     - `https://www.stpls3d.com/data <https://www.stpls3d.com/data>`_
     - 
   * - nuSences
     - The nusences dataset is a large-scale dataset for autonomous driving developed by the team at `Motional <https://www.motional.com/>`_. It contains 1000 scenes with 1.4M camera images.
     - 1000 scenes
     - `https://www.nuscenes.org/nuscenes#download <https://www.nuscenes.org/nuscenes#download>`_
     - 
   * - TartanAir
     - TartanAir is a challenging dataset for robot navigation task and more, collected in photo-realistic simulation environments with various light conditions, weather and moving objects. The data is multi-modal, with ground truth labels such as depth, segmentation, camera pose and LiDAR points.
     - 16 scenes
     - `https://theairlab.org/tartanair-dataset/ <https://theairlab.org/tartanair-dataset/>`_
     - 


In-the-wild Dataset
-------------------------

.. list-table::
   :header-rows: 1

   * - **Name**
     - **Brief Description**
     - **Statistics**
     - **Download Link**
     - **How to Use**
   * - NeROIC
     - The dataset used in `NeROIC <https://zfkuang.github.io/NeROIC/>`_\ , an object level multi view images dataset.
     - 4 Objects
     - `https://drive.google.com/drive/folders/1HzxaO9CcQOcUOp32xexVYFtsyKKULR7T <https://drive.google.com/drive/folders/1HzxaO9CcQOcUOp32xexVYFtsyKKULR7T>`_
     - 
   * - NeRF-OSR
     - The NeRF-OSR dataset contains eight sites shot at different timings using a DSLR camera and a 360° camera to capture the environment map. A colour chequerboard is also captured for colour calibration.
     - 8 scenes
     - https://nextcloud.mpi-klsb.mpg.de/index.php/s/mGXYKpD8raQ8nMk
     - 
   * - SAMURAI
     - The dataset used in `SAMURAI <https://markboss.me/publication/2022-samurai/>`_\ , each image may have different backgrounds.
     - 8 objects
     - `https://www.dropbox.com/sh/x3u2szvaqjtaykl/AACCZn05NciMa5bHhn60p9vja?dl=0 <https://www.dropbox.com/sh/x3u2szvaqjtaykl/AACCZn05NciMa5bHhn60p9vja?dl=0>`_
     - 
   * - PhotoTourism
     - The phototourism dataset used in NeRF in the wild consisted of 6 scenes with large changes in lighting and many dynamic objects.
     - 6 scenes
     - https://www.cs.ubc.ca/~kmyi/imw2020/data.html
     - 
   * - NeRD
     - A dataset used for inverse rendering, captures 3 synthesis and 5 real objects with complex material and lighting condition.
     - 8 objects
     - `https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition/blob/master/download_datasets.py <https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition/blob/master/download_datasets.py>`_
     - 
   * - NAVI
     - The NAVI dataset consists of both in-the-wild and multi-view image collections with high-quality aligned 3D shape ground-truths
     - 37 objects
     - `https://github.com/google/navi <https://github.com/google/navi>`_
     - 
   * - MegaDepth
     - The MegaDepth dataset is a dataset for single-view depth prediction that includes 196 different locations reconstructed from COLMAP SfM/MVS.
     - 196 scenes
     - `https://www.cs.cornell.edu/projects/megadepth/ <https://www.cs.cornell.edu/projects/megadepth/>`_
     - 
   * - BigTIME
     - The BigTime dataset includes > 200 timelapse image sequences collected from the Internet designed to do intrinsic image decomposition.
     - 145 indoor scenes50 outdoor scenes
     - `https://www.cs.cornell.edu/projects/bigtime/ <https://www.cs.cornell.edu/projects/bigtime/>`_
     - 
   * - sitcom3D
     - The sitcom3D is consisted of indoor images collected from TV shows.
     - 7 scenes
     - `https://github.com/ethanweber/sitcoms3D/blob/master/METADATA.md <https://github.com/ethanweber/sitcoms3D/blob/master/METADATA.md>`_
     - 
   * - Neural Scene Chronology
     - The dataset captures two commercial tourist areas, a graffiti mecca, and a museum. All contain elements that change significantly over time.
     - 4 scenes
     - `https://github.com/zju3dv/NeuSC <https://github.com/zju3dv/NeuSC>`_
     - 


Indoor Datasets
-----------------------------

.. list-table::
   :header-rows: 1

   * - **Name**
     - **Brief Description**
     - **Statistics**
     - **Download Link**
     - **How to Use**
   * - Deep Blending
     - 
     - 
     - 
     - 
   * - MuSHRoom
     - 
     - 
     - 
     - 
   * - EyefulTower
     - 
     - 
     - 
     - 
   * - Replica
     - 
     - 
     - 
     - 
   * - `Scalable-Neural-Indoor-Scene-Rendering <https://xchaowu.github.io/papers/scalable-nisr/>`_
     - 
     - 
     - `https://drive.google.com/drive/folders/11kZ9vu1BqKKqV4p5RRwlh-f_hQbEpaIp?usp=sharing <https://drive.google.com/drive/folders/11kZ9vu1BqKKqV4p5RRwlh-f_hQbEpaIp?usp=sharing>`_
     - 
   * - Neural RGB-D
     - 
     - 
     - 
     - 
   * - TUM-RGBD
     - 
     - 
     - 
     - 
   * - ScanNet++
     - 
     - 
     - 
     - 
   * - BS3D
     - 
     - 
     - 
     - 
   * - ScanNet
     - 
     - 
     - 
     - 
   * - RealEstate10K
     - 
     - 
     - 
     - 
   * - ArkitScenes
     - 
     - 
     - 
     - 
   * - 3RScan
     - 
     - 
     - 
     - 
   * - multiscan
     - 
     - 
     - 
     -
