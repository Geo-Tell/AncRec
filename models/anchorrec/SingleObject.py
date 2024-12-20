import os
import pickle
import numpy as np
import torch.utils.data
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path, ShapeNetv2_imcomplete_path
from configs.path_config import scannet_processed_path
import cv2
import h5py
default_collate = torch.utils.data.dataloader.default_collate
MAX_NUM_OBJ = 64
MAX_NUM_QUAD = 32
MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673])


class SingleObject():
    def __init__(self, cfg, mode):
        super(SingleObject, self).__init__()
        # self.augment = False
        self.augment = mode == 'train'
        self.mode = mode
        self.input_path = cfg.config['data']['input_path']

        self.resample = cfg.config['data'].get('resample',False)
        self.mincount = cfg.config['data'].get('mincount',40)
        self.maxcount = cfg.config['data'].get('maxcount',250)
        self.use_rgb= cfg.config['data'].get('use_rgb',False)
        self.output_images = cfg.config['data'].get('output_images',False)
        self.vis_compare = cfg.config['data'].get('vis_compare',False)
        with open(cfg.config['data']['pretrained_features_pth'], 'rb') as pf:
            self.zs = pickle.load(pf)
        with open('datasets/all_vox256_img.txt') as f:
            self.data_name_lists = f.readlines()
        self.use_decoder = False
        if 'completion' in cfg.config['model'] or mode == 'test':
            self.use_decoder = True #Use BSPNet decoder for mesh generation
            data_dict = h5py.File('datasets/all_vox256_img.hdf5', 'r')
            if 'sample_vox_size' in cfg.config['model']['completion']:
                self.sample_vox_size = cfg.config['model']['completion']['sample_vox_size']
            else:
                self.sample_vox_size = 64
            if self.sample_vox_size == 16:
                self.load_point_batch_size = 16 * 16 * 16
            elif self.sample_vox_size == 32:
                self.load_point_batch_size = 16 * 16 * 16
            elif self.sample_vox_size == 64:
                self.load_point_batch_size = 16 * 16 * 16 * 4
            self.whole_data_points = np.array(data_dict['points_' + str(self.sample_vox_size)],dtype = np.float32)
            self.whole_data_values = np.array(data_dict['values_' + str(self.sample_vox_size)],dtype = np.float32)
            # self.whole_data_voxels = np.array(data_dict['voxels'][:])
            self.input_size = 64

        self.vis_scan_names = cfg.config['log'].get('vis_scan_names',None)
        if mode =='test':
            self.load_point_data_test(cfg)
        elif mode == 'vis':
            self.load_point_data_vis()
        else:
            self.vis_scan_names = None
            self.output_images = False
            self.load_point_data(cfg, mode)
        # self.load_point_data_small(cfg, mode) （# for debugging on small dataset)

    def load_point_data(self, cfg, mode):
        ## mode = train/val
        self.scan_names = []
        # self.objectness_label = []
        self.gt_shapenet_ids = []
        self.gt_shapenet_catids = []
        self.cls_labels = []
        # recon
        self.input_vote_features = []
        self.input_anchor_features = []
        self.input_points = []
        self.boxids = []

        input_path = cfg.config['data']['input_path']
        with open(os.path.join(input_path, '%s.pkl' % mode), 'rb') as f:
            data = pickle.load(f)

        input_point_list = data['anchor_sampled_pts_list'] # _xyzs
        # objectness_label_list = data['objectness_label']
        shapenet_catids_list = data['shapenet_ids_list']
        shapenet_ids_list = data['shapenet_catids_list']
        vote_features = data['vote_features']
        anchor_features = data['anchor_features']
        self.num_pts = len(input_point_list[0][0])

        scan_names = data['scan_names']
        scene_num = len(input_point_list)

        '''Select positive proposals '''
        for iter in range(scene_num):
            if self.vis_scan_names is not None and scan_names[iter] not in self.vis_scan_names:
                continue
            num_proposal = input_point_list[iter].shape[0]
            for nid in range(num_proposal):
                assert type(shapenet_ids_list[iter][nid]) == str
                self.gt_shapenet_ids.append(shapenet_ids_list[iter][nid])
                self.gt_shapenet_catids.append(shapenet_catids_list[iter][nid])
                self.input_points.append(input_point_list[iter][nid])
                self.input_vote_features.append(vote_features[iter][nid])
                self.input_anchor_features.append(anchor_features[iter][nid])
                self.scan_names.append(scan_names[iter])
                # self.boxids.append(nid)

        unique_shapenet_ids = list(set(self.gt_shapenet_ids))
        unique_shapenet_ids.sort()
        shapenet2cls = {shape_id: i for i, shape_id in enumerate(unique_shapenet_ids)}
        self.cls_labels = [shapenet2cls[shape_id] for shape_id in self.gt_shapenet_ids]

    def __getitem__(self, idx):
        ret_dict = {}
        ret_dict['scan_name'] = self.scan_names[idx]
        input_points = self.input_points[idx]
        if self.augment:
            drop_ratio = 0.2
            jitter_std = 0.01
            input_points = self.pointcloudScale(input_points)
            input_points = self.pointcloudJitter(input_points,std=jitter_std)
            input_points = self.pointcloudDropFilled(input_points , drop_ratio = drop_ratio, std=0.02, clip=0.04)

        ret_dict['input_points'] =input_points.astype(np.float32) #normalized points
        ret_dict['input_vote_features'] = self.input_vote_features[idx].astype(np.float32)
        ret_dict['input_anchor_features'] = self.input_anchor_features[idx].astype(np.float32)
        # ret_dict['box_id'] = self.boxids[idx]

        ret_dict['shape_label'] = self.cls_labels[idx]
        ret_dict['zs'] = self. zs[self.gt_shapenet_catids[idx] + '/' +self.gt_shapenet_ids[idx]][0].astype(np.float32)
        gt_mesh_path = os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path,
                                    self.gt_shapenet_catids[idx] + '/' +self.gt_shapenet_ids[idx] + '.off')
        ret_dict['mesh_pth'] = gt_mesh_path
        if self.use_decoder:
            id = self.data_name_lists.index(self.gt_shapenet_catids[idx] + '/' +self.gt_shapenet_ids[idx] + '\n')
            data_points = (self.whole_data_points[id] + 0.5) / 256 - 0.5
            data_points = np.concatenate([data_points, np.ones([self.load_point_batch_size, 1], np.float32)], axis=1)
            data_values = self.whole_data_values[id]
            # data_voxels = self.whole_data_voxels[id].astype(np.float32)
            ret_dict['data_points'] = data_points
            ret_dict['data_values'] = data_values
        return ret_dict

    def __len__(self):
        # assert len(self.points_for_completion_paths) == len(self.input_features_paths)
        return len(self.input_points)

    def pointcloudScale(self, points, lo=0.8, hi=1.25):
        scaler = np.random.uniform(lo, hi)
        points[:, 0:3] *= scaler
        return points

    def pointcloudJitter(self, points, std=0.01, clip=0.05):
        jittered_data = std * np.random.randn(points.shape[0], 3).clip(-clip, clip)
        points[:, 0:3] += jittered_data
        return points

    def pointcloudDropFilled(self, points, drop_ratio=0.3, std=0.01, clip=0.05):
        point_number = len(points)
        sampled_number = int(point_number * (1-drop_ratio))
        points = points[np.random.choice(np.arange(point_number), sampled_number),:]
        new_point_number = point_number - sampled_number
        jittered_data = std * np.random.randn(new_point_number, 3).clip(-clip, clip)
        new_points = points[np.random.choice(np.arange(sampled_number), new_point_number),:]
        new_points[:,:3] += jittered_data
        points = np.concatenate([points, new_points],axis=0)
        return points

    def pointcloudFlip(self, points, axis = 0):
        if axis not in [0,1]:
            axis = 0
        if np.random.random() > 0.5:
            points[:, axis] = -1 * points[:, axis]
        return points

def recursive_cat_to_numpy(data_list):
    '''Covert a list of dict to dict of numpy arrays.'''
    out_dict = {}
    for key, value in data_list[0].items():
        if isinstance(value, np.ndarray):
            out_dict = {**out_dict, key: np.concatenate([data[key][np.newaxis] for data in data_list], axis=0)}
        elif isinstance(value, dict):
            out_dict =  {**out_dict, **recursive_cat_to_numpy(value)}
        elif np.isscalar(value):
            out_dict = {**out_dict, key: np.concatenate([np.array([data[key]])[np.newaxis] for data in data_list], axis=0)}
        elif isinstance(value, list):
            out_dict = {**out_dict, key: np.concatenate([np.array(data[key])[np.newaxis] for data in data_list], axis=0)}
    return out_dict

def collate_fn(batch):
    '''
    data collater
    :param batch:
    :return:
    '''
    collated_batch = {}
    for key in batch[0]:
        if key not in ['shapenet_catids', 'shapenet_ids']:
           collated_batch[key] = default_collate([elem[key] for elem in batch])
        else:
            collated_batch[key] = [elem[key] for elem in batch]

    return collated_batch

class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        # self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, 'flag'):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)






