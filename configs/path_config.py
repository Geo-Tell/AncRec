import os
from glob import glob
import pickle

import numpy as np
from utils.scannet.scannet_utils import read_label_mapping

scannet_path = '/data/dong/anchorrec/scannet'
scannet_processed_path = '/data/dong/anchorrec/processed_data'
scannet_processed_path2 = '/data/dong/anchorrec/processed_data2' # '/home/dmy/data/datasets/processed_data2'
data_root_path = '/home/dmy/data'
layout_path = '/data/dong/anchorrec/scannet_planes'
scan2cad_path = '/data/dong/anchorrec/scan2cad'
raw_scans_path = '/data/dong/anchorrec/scans'
posed_images_instance_path =  '/data/dong/anchorrec/posed_images_final_instance'
posed_images_path =  '/data/dong/anchorrec/posed_images_final'
split_path = '/data/dong/anchorrec/splits'
instance_gt_rgb_path = '/data/dong/anchorrec/scannet_rgb'
scans_dir = '/data/dong/anchorrec/scans'
SHAPENETCLASSES = ['void',
                   'table', 'jar', 'skateboard', 'car', 'bottle',
                   'tower', 'chair', 'bookshelf', 'camera', 'airplane',
                   'laptop', 'basket', 'sofa', 'knife', 'can',
                   'rifle', 'train', 'pillow', 'lamp', 'trash_bin',
                   'mailbox', 'watercraft', 'motorbike', 'dishwasher', 'bench',
                   'pistol', 'rocket', 'loudspeaker', 'file cabinet', 'bag',
                   'cabinet', 'bed', 'birdhouse', 'display', 'piano',
                   'earphone', 'telephone', 'stove', 'microphone', 'bus',
                   'mug', 'remote', 'bathtub', 'bowl', 'keyboard',
                   'guitar', 'washer', 'bicycle', 'faucet', 'printer',
                   'cap', 'clock', 'helmet', 'flowerpot', 'microwaves']

ScanNet_OBJ_CLASS_IDS = np.array([ 1,  7,  8, 13, 20, 31, 34, 43]) #table, chair, bookshelf, sofa, trash_bin, cabinet, display, bathtub
shapenetid2class = {class_id: i for i, class_id in enumerate(list(ScanNet_OBJ_CLASS_IDS))} #table, chair, bookshelf, sofa, trash_bin, cabinet, display, bathtub
ScanNet_OBJ_CLASS_NAMES =  ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
ScanNet_OBJ_CLASS2ID = {'04379243':0,'03001627':1,'02871439':2,'04256520':3,'02747177':4,'02933112':5,'03211117':6,'02808440':7}
ShapeNetIDMap = {'4379243': 'table', '3593526': 'jar', '4225987': 'skateboard', '2958343': 'car', '2876657': 'bottle', '4460130': 'tower', '3001627': 'chair', '2871439': 'bookshelf', '2942699': 'camera', '2691156': 'airplane', '3642806': 'laptop', '2801938': 'basket', '4256520': 'sofa', '3624134': 'knife', '2946921': 'can', '4090263': 'rifle', '4468005': 'train', '3938244': 'pillow', '3636649': 'lamp', '2747177': 'trash_bin', '3710193': 'mailbox', '4530566': 'watercraft', '3790512': 'motorbike', '3207941': 'dishwasher', '2828884': 'bench', '3948459': 'pistol', '4099429': 'rocket', '3691459': 'loudspeaker', '3337140': 'file cabinet', '2773838': 'bag', '2933112': 'cabinet', '2818832': 'bed', '2843684': 'birdhouse', '3211117': 'display', '3928116': 'piano', '3261776': 'earphone', '4401088': 'telephone', '4330267': 'stove', '3759954': 'microphone', '2924116': 'bus', '3797390': 'mug', '4074963': 'remote', '2808440': 'bathtub', '2880940': 'bowl', '3085013': 'keyboard', '3467517': 'guitar', '4554684': 'washer', '2834778': 'bicycle', '3325088': 'faucet', '4004475': 'printer', '2954340': 'cap', '3046257': 'clock', '3513137': 'helmet', '3991062': 'flowerpot', '3761084': 'microwaves'}
# trash bin: 2747177; '4379243': 'table'
class PathConfig(object):
    def __init__(self, dataset):
        if dataset == 'scannet':
            self.metadata_root = scannet_path
            self.split_files = [split_path + '/scannet/scannetv2_train.txt',
                                split_path + '/scannet/scannetv2_val.txt']
            all_scenes = []
            for split_file in self.split_files:
                all_scenes += list(np.loadtxt(split_file, dtype=str))

            scene_paths = np.sort(glob(os.path.join(self.metadata_root, 'scans', '*')))
            self.scene_paths = [scene_path for scene_path in scene_paths if os.path.basename(scene_path) in all_scenes]
            self.scan2cad_annotation_path = os.path.join(scan2cad_path, 'full_annotations.json')
            self.processed_data_path = os.path.join(self.metadata_root, 'processed_data')

            label_type = 'synsetoffset'
            self.raw_label_map_file = os.path.join(self.metadata_root, 'raw_lablemap_to_' + label_type + '.pkl')
            self.OBJ_CLASS_IDS = ScanNet_OBJ_CLASS_IDS

            if not os.path.exists(self.raw_label_map_file):
                LABEL_MAP_FILE = os.path.join(self.metadata_root, 'scannetv2-labels.combined.tsv')
                assert os.path.exists(LABEL_MAP_FILE)
                raw_label_map = read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to=label_type)
                label_map = {}
                for key, item in raw_label_map.items():
                    if item in ShapeNetIDMap:
                        label_map[key] = SHAPENETCLASSES.index(ShapeNetIDMap[item])
                    else:
                        label_map[key] = 0
                with open(self.raw_label_map_file, 'wb') as file:
                    pickle.dump(label_map, file)

            if not os.path.exists(self.processed_data_path):
                os.mkdir(self.processed_data_path)

if __name__ == '__main__':
    path_config = PathConfig('scannet')
    print(path_config.scene_paths)
