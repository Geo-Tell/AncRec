## Installation
Install the required packages in [requirement.txt](requirement.txt)

Install the pointnet submodule:
```
cd external/pointnet2_ops_lib
pip install .
```

## Data
Configurate the paths in [path_config.py](configs/path_config.py) and [shapenet_path_configuration](utils/shapenet/__init__.py)
The data structure will be organised as follows:
```
data
├── scannet
│   ├── processed_data 
│   │   ├── scene_name/
│   │   │   ├── full_scan.npz           # Processed point cloud and point votes
│   │   │   ├── bbox.pkl                # GT bounding box labels
│   │   │   ├── object_shape_points.npz # GT surface points to supervise the generation of shape anchor of objects
│   │   │   ├── quad_svotes.npz         # GT surface points to supervise the generation of shape anchor of layout
│   │   │   ├── quad_votes.npz          # Quad votes
│   ├── processed_data2 
│   │   ├── scene_name/
│   │   │   ├── bbox.pkl                # GT bounding box labels
│   │   │   ├── new_votes.npy           # GT surface points to supervise the generation of shape anchor of objects
│   ├── posed_images                                # Images and correspondence with point clouds
│   │   ├── scene_name/
│   │   │   ├── ptids_images.pkl        # Correspondence between image pixel coordinates and point cloud indices
│   │   │   ├── index.npy               # Selected image indices from the video sequence of ScanNet data
│   │   │   ├── output_instance2imageid.pkl    # Images corresponding to each instance sorted by coverage of the projected instance points
│   │   │   ├── intrinsics.txt          # Camera intrinsics
│   │   │   ├── 00000.jpg               # RGB image
│   │   │   ├── 00000.png               # Depth image
│   │   │   ├── 00000.txt               # Camera extrinsics
│   │   │   ├── .......
│   ├── splits           # Data splits   
│   ├── scannet_planes   # Layout data
│  ├── scannet_rgb       # Cropped RGB images corresponding to each GT instance (see ./data_processing/make_gt_instance_images.py for details)
│  ├── scan2cad          # ScanNet to CAD Alignment data
├── ShapeNetv2_data   # Pre-processed ShapeNet data

dataset               # Pretrained BSPNet models and features
├── latentz.pkl
├── BSP_AE.model64.pth
├── all_vox256_img.hdf5
├── all_vox256_img.txt
```
1. Ask for the [ScanNet](http://www.scan-net.org/) dataset and download it to `data/scannet/scans`
2. Ask for the [Scan2CAD](https://github.com/skanti/Scan2CAD) dataset and download it to  `data/scannet/scan2cad`
3. Download the pre-processed scannet data from [RfD-Net](https://github.com/GAP-LAB-CUHK-SZ/RfDNet) and unzip it to `data/scannet/processed_data`, which consists of point cloud, votes and bounding box parameters (full_scan.npz and bbox.pkl).
4. Download the processed data [[link](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/EWQ0PiEY0qVImiWHH2ku7HEBCV8mjeIbj5Hcpa84adiA2Q?e=30iYKX)] and extract them to `data/ShapeNetv2_data`.
5. Download the sceneCAD layout dataset from the link provided by [PQ-Transformer](http://kaldir.vc.in.tum.de/scannet_planes) and unzip it to `data/scannet/scannet_planes`.
6. Generate GT object surface points to supervise the prediction of shape anchors.
    ```
    python data_preprocessing/sample_mesh_and_resize.py
    ```
7. Generate layout votes and GT layout surface points.
    ```
    python data_preprocessing/generate_layout_votes_and_anchors.py
    ```
8. To fit the problem of the discrepancy between ScanNet instance labels and Scan2CAD bounding box labels.
   ```
   python update_bbox_labels.py  
   ```
9. Extract camera extrinsic, intrinsics, RGB and depth images from the ScanNet sensor data,  while simultaneously extract the correspondences between image pixel coordinates and point cloud indices. 
Also build connection between images and instances in the scene to accelerate the training process. Images corresponding to each instance are sorted by comparing the coverage of the projected instance points.
The output data requires a storage of approximately 1T.
   ```
   python extract_images_and_point_correspondences.py  
   ```
10. Extract image regions corresponding to GT instances for pretraining.
     ```
      python make_gt_instance_images.py  
     ```
11. The pretrained BSPNet models and shape features can be downloaded from [here](https://drive.google.com/drive/folders/1LQ4Og0jClbx7mXDm6H_hIIUaN2Wc-Xx5?usp=drive_link)

## Demo
The pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1-uQ13sC6VUPVeHrdZ82ZjZXpC8AQwbkk?usp=drive_link). 
Put the pretrained model in the ./ckpts folder.

Generate scene mesh reconstruction results with the following script:
 ```bash
python demo.py --config configs/config_files/final_recon_with_rgb.yaml
 ```
Specify the scenes to generate in by modifying *vis_scan_names* in [final_recon_with_rgb.yaml](configs/config_files/final_recon_with_rgb.yaml). If you want to generate the results for all scenes in the test dataset, set vis_scan_names to all.

## Visualization
We use open3d for visualization. See [vis_single_scene.py](vis/vis_single_scene.py) for details.

## Evaluation
After generating the scene object meshes, we use the code in [dimr](https://github.com/ashawkey/dimr) for evaluation by comparing the predicted meshes and the GT meshes. Please downloade the GT meshes from [here](https://drive.google.com/file/d/1ArUgyoSfXuSP34Asf0HrZYbd28yPm0vQ/view?usp=sharing).


## Training procedure
Stage 1: train the **detection** module. 
 ```bash
python main.py --config configs/config_files/detection.yaml
 ```
Stage 2: Use the pretrained weight from stage 1 to save the intermediate detection outcomes of both the training and validation sets.
In this way, we can train the following RGB branch and reconstruction branch in a proposal-wise manner, which allows more flexible data augmentation compared to the scene-wise training.
 ```bash
python main.py --config configs/config_files/prepare_data.yaml --mode prepare_data_for_refine
 ```
Cconfigure the **split_name** item in [prepare_data.yaml](configs/config_files/prepare_data.yaml) to generate the data of train/val set respectively.

Stage 3: Use the cropped rgb images of each GT instance (data/scanent/scannet_rgb) to pretrain the backbone of the image encoder.
 ```bash
python main.py  --config configs/config_files/rgb_pretrain.yaml --mode pretrain_rgb
 ```

Stage 4: Train the RGB branch to refine the detection result.
 ```bash
python main.py --config configs/config_files/detection_refine.yaml --mode train_detection_refine
 ```

Stage 5: Output **RGB** features obtained from Stage 4 as part of the input for training the reconstruction branch.
 ```bash
python main.py --config configs/config_files/detection_refine_test.yaml --mode prepare_recon_data_w_rgb
 ```
Cconfigure the *split_name* in [detection_refine_test](configs/config_files/detection_refine_test) to generate the data of train/val set respectively.

Stage 6: Train the **reconstruction** module. 
 ```bash
python main.py --config configs/config_files/singleObject_align_w_rgb.yaml --mode train_recon
 ```
Please check the config files for more options.


## Acknowledgement
The code is built based on [RfD-Net](https://github.com/GAP-LAB-CUHK-SZ/RfDNet/tree/main), [PQ-Transformer](http://kaldir.vc.in.tum.de/scannet_planes), [dimr](https://github.com/ashawkey/dimr) and [BSP-Net](https://github.com/czq142857/BSP-NET-pytorch/tree/master). We thank all the authors for their excellent works.