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
│   ├── splits            # Data splits   
│   ├── scannet_planes    # Layout data
├│   ── scan2cad          # ScanNet to CAD Alignment data
├── ShapeNetv2_data   # Pre-processed ShapeNet data

dataset               # Pretrained BSPNet models and features
├── latentz.pkl
├── BSP_AE.model64.pth
├── all_vox256_img.hdf5
├── all_vox256_img.txt
```

1. Ask for the [ScanNet](http://www.scan-net.org/) dataset and download it to
   ```
   data/scannet/scans
   ```
2. Ask for the [Scan2CAD](https://github.com/skanti/Scan2CAD) dataset and download it to
   ```
   data/scannet/scan2cad
   ```
3. Download the pre-processed scannet data from [RfD-Net](https://github.com/GAP-LAB-CUHK-SZ/RfDNet), which consists of point cloud, votes and bounding box parameters (full_scan.npz and bbox.pkl).

4. Download the processed data [[link](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/EWQ0PiEY0qVImiWHH2ku7HEBCV8mjeIbj5Hcpa84adiA2Q?e=30iYKX)] and extract them to `data/ShapeNetv2_data` as below
    ```
    data/ShapeNetv2_data/point
    data/ShapeNetv2_data/pointcloud
    data/ShapeNetv2_data/voxel
    data/ShapeNetv2_data/watertight_scaled_simplified
    ```
5. Download the sceneCAD layout dataset from the link provided by [PQ-Transformer](http://kaldir.vc.in.tum.de/scannet_planes) and unzip it to data/scannet/scannet_planes.
6. Generate GT object surface points to supervise the prediction of shape anchors.
    ```
    python data_preprocessing/sample_mesh_and_resize.py
    ```
7. Generate layout votes and GT layout surface points.
    ```
    python data_preprocessing/generate_layout_votes_and_anchors.py
    ```
8. The pretrained BSPNet models and shape features can be downloaded from [here](https://drive.google.com/drive/folders/1LQ4Og0jClbx7mXDm6H_hIIUaN2Wc-Xx5?usp=drive_link)


## Demo
The pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1vUjG3tWISI_X6GIveMRW9YC3f85cXVzm?usp=drive_link). 
Put the pretrained model in the ./ckpts folder.

Generate scene mesh reconstruction results with the following script:
```
python demo.py ---config configs/config_files/final_recon.yaml
```
Specify the scenes to generate by modifying *vis_scan_names* in [final_recon.yaml](configs/config_files/final_recon.yaml). If you want to generate the results for all scenes in the test dataset, set *vis_scan_names* to all.

## Visualization
We use open3d for visualization. See [vis_single_scene.py](vis/vis_single_scene.py) for details.

## Evaluation
After generating the scene object meshes, we use the code in [dimr](https://github.com/ashawkey/dimr) for evaluation by comparing the predicted meshes and the GT meshes. Please downloade the GT meshes from [here](https://drive.google.com/file/d/1ArUgyoSfXuSP34Asf0HrZYbd28yPm0vQ/view?usp=sharing).
You can also test the results of object detection and layout estimation by 
 ```bash
python main.py --config configs/config_files/detection.yaml --mode test
 ```

## Training procedure
Stage 1: train the **detection** module. 
 ```bash
python main.py --config configs/config_files/detection.yaml
 ```
Stage 2: Output the detection results based on the pretrained-weights from stage 1. Extract the object-wise information for the further training.
 ```bash
python main.py --config configs/config_files/prepare_data.yaml --mode prepare_data  
 ```
Cconfigure the *split_name* in [prepare_data.yaml](configs/config_files/prepare_data.yaml) to generate the data of train/val set respectively.

Stage 3: train the **reconstruction** module. 
 ```bash
python main.py --config configs/config_files/singleObject.yaml --mode train_recon
 ```
Please check the config files for more options.


## Acknowledgement
The code is built based on [RfD-Net](https://github.com/GAP-LAB-CUHK-SZ/RfDNet/tree/main), [PQ-Transformer](http://kaldir.vc.in.tum.de/scannet_planes), [dimr](https://github.com/ashawkey/dimr) and [BSP-Net](https://github.com/czq142857/BSP-NET-pytorch/tree/master). We thank all the authors for their excellent works.
