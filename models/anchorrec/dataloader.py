import os
import pickle
import random
from glob import glob
import numpy as np
from collections import Counter
from utils.read_and_write import read_json
import torch.utils.data
from torch.utils.data import DataLoader,WeightedRandomSampler
# from external import binvox_rw
from models.datasets import ScanNet
from net_utils.libs import rotz
from net_utils.transforms import SubsamplePoints, SubsamplePoints2
from utils import pc_util
from utils.scannet.scannet_planes import get_quads, rotate_quad
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
from configs.path_config import scannet_processed_path, scannet_processed_path2, scannet_path, data_root_path, posed_images_path, posed_images_instance_path
from configs.path_config import ScanNet_OBJ_CLASS_IDS
from net_utils.common_utils import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from net_utils import augmentor_utils
from net_utils.libs import flip_axis_to_camera
from net_utils.box_util import get_3d_box
from configs.scannet_config import ScannetConfig

dataset_config = ScannetConfig()
default_collate = torch.utils.data.dataloader.default_collate
MAX_NUM_OBJ = 64
MAX_NUM_QUAD = 32
MAX_INSTANCE_PT_NUMBER = 8000
MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673])
class AnchorRec_ScanNet(ScanNet):
    def __init__(self, cfg, mode):
        super(AnchorRec_ScanNet, self).__init__(cfg, mode)
        self.num_points = cfg.config['data']['num_point']
        self.use_color = cfg.config['data']['use_color_detection'] or cfg.config['data']['use_color_completion']
        self.use_height = not cfg.config['data']['no_height']
        self.num_quad_proposal = cfg.config['data']['num_quad_proposal']
        self.num_proposal = cfg.config['data']['num_target']
        self.augment = mode == 'train'
        # self.shapenet_path = cfg.config['data']['shapenet_path']
        # self.points_unpackbits = cfg.config['data']['points_unpackbits']
        # self.n_points_object = cfg.config['data']['points_subsample']
        # self.points_transform = SubsamplePoints2(cfg.config['data']['points_subsample'], mode)
        self.phase = cfg.config[self.mode]['phase']
        self.noise_traslation_std =  cfg.config['data']['noise_traslation_std']
        self.world_scale_range =  cfg.config['data']['world_scale_range']
        self.posed_images_path = posed_images_path #cfg.config['data'].get('posed_images_path', None)
        self.posed_images_path2 = posed_images_instance_path #cfg.config['data'].get('posed_images_path2', None)
        # if self.mode == 'test':
        #     self.phase = cfg.config['test']['test_phase']
        # else:
        #     self.phase = cfg.config[self.mode]['phase']
        # if type(self.phase) == list and len(self.phase) == 1:
        #     self.phase = self.phase[0]
        if self.phase == 'quad_detection' or self.phase == 'joint_detection':
            self.use_quad = True
        else:
            self.use_quad = False
        if 'test_quad' in  cfg.config[self.mode] and  cfg.config[self.mode]['test_quad'] == True:
            self.use_quad = True
        if 'test_phase' in cfg.config[self.mode] and  cfg.config[self.mode]['test_phase']== ['quad_detection', 'joint_detection']:
            self.use_quad = True
        if self.phase == 'prepare_data':
            self.mode = 'val'
            self.augment = False
            # with open('./datasets/latentz.pkl', 'rb') as f:
        #     self.zs_dataset = pickle.load(f)

    def __getitem__(self, idx):
        use_quad = False
        data_path = self.split[idx]

        scan_name = data_path['bbox'].split('/')[-2]
        self.scene_root_path = os.path.join(scannet_processed_path, scan_name)
        scan_data = np.load(os.path.join(self.scene_root_path, data_path['scan'].split('/')[-1]))
        point_cloud = scan_data['mesh_vertices']
        point_instance_labels = scan_data['instance_labels']

        # use new vote
        point_votes_ = np.load(os.path.join(scannet_processed_path2, scan_name,'new_votes.npy'))
        point_votes = np.zeros((len(point_votes_),10))
        point_votes[:,:4] = point_votes_
        point_votes[:,4:7] = point_votes_[:,1:4]
        point_votes[:,7:10] = point_votes_[:,1:4]
        # point_votes = scan_data['point_votes']
        point_vote_mask = point_votes[:,0].copy()
        if point_vote_mask.sum()==0:
            print('no valid vote%s'%scan_name)
        # point_instance_labels = scan_data['instance_labels']

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]  # do not use color for now
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        with open(os.path.join(scannet_processed_path2, scan_name,'bbox.pkl'), 'rb') as file:
            box_info = pickle.load(file)

        surface_points = np.load(os.path.join(self.scene_root_path, 'object_shape_points.npz'))['points']
        # surface_points = np.load(os.path.join(data_root_path,data_path['surface'].replace('object_shape_points.npz', 'osp24.npz')))['points']
        surface_points = surface_points.reshape(-1, 3)

        # object_instance_ids_all = np.zeros((len(point_cloud)),np.uint8)
        boxes3D = []
        classes = []
        shapenet_catids = []
        shapenet_ids = []
        # shapenet_instance_ids_all = []
        # shapenet_instance_inds = [] #补零问题：按场景保存
        points_data = []
        object_instance_ids = []
        target_bbox_instance_pts = np.zeros((MAX_NUM_OBJ, MAX_INSTANCE_PT_NUMBER, 3))
        target_bbox_instance_pt_numbers = np.zeros((MAX_NUM_OBJ,)).astype(np.int64) #每个instance最多xx个点

        # mesh_zs = []
        box_id = 0
        for item in box_info:
            object_instance_ids.append(item['instance_id'])
            boxes3D.append(item['box3D'])
            classes.append(item['cls_id'])
            shapenet_catids.append(item['shapenet_catid'])
            shapenet_ids.append(item['shapenet_id'])
            if self.phase == 'prepare_data' and self.mode != 'test':
                shapenet_instance_inds = item['shapenet_instance_inds']
                # shapenet_instance_ids_all.append(item['shapenet_instance_inds'])
                # object_instance_ids_all[item['shapenet_instance_inds']] = box_id + 1
                pt_number = len(shapenet_instance_inds)
                if pt_number > MAX_INSTANCE_PT_NUMBER:
                    shapenet_instance_inds = shapenet_instance_inds[
                        np.random.choice(pt_number, MAX_INSTANCE_PT_NUMBER, replace=False)]
                target_bbox_instance_pts[box_id, :pt_number, :] = point_cloud[shapenet_instance_inds, :3]
                target_bbox_instance_pt_numbers[box_id] = pt_number
                # points_data.append(self.get_shapenet_points(box_id, self.points_transform))
            box_id += 1
            # mesh_zs.append(self.zs_dataset[shapenet_catid + '/' + shapenet_id])
        boxes3D = np.array(boxes3D)
        # mesh_zs = np.array(mesh_zs)

        '''quad relevant'''
        quad_surface_points = np.zeros((1, 3))  # 1024 x nquads,3
        quad_votes = np.zeros((self.num_points, 3))  # npc,3
        rectangles = np.zeros((1, 6))
        if self.use_quad and data_path.__contains__('quad_surface'):
            if os.path.exists(os.path.join(self.scene_root_path, data_path['quad_surface'].split('/')[-1])):
                quad_surface_points = np.load(os.path.join(self.scene_root_path, data_path['quad_surface'].split('/')[-1]))[
                    'quad_svotes'].transpose(1, 0, 2)
                quad_votes = np.load(os.path.join(self.scene_root_path, data_path['quad_vote'].split('/')[-1]))['quad_votes']

                rectangles, total_quad_num, horizontal_quads = get_quads(scan_name)
                quad_shape_num = quad_surface_points.shape[0]
                quad_surface_points = quad_surface_points.reshape(-1, 3)
                quad_num = rectangles.shape[0]
                use_quad = True

        '''Augment'''
        if self.augment:
            point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_point = augmentor_utils.random_flip_along_x(
                point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_points)
            point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_point = augmentor_utils.random_flip_along_y(
                point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_points)
            point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_point = augmentor_utils.global_rotation_original(
                point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_points)
            point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_point = augmentor_utils.global_scaling(
                point_cloud, point_votes, surface_points, boxes3D, rectangles, quad_votes, quad_surface_points, self.world_scale_range)
            point_cloud, surface_points, boxes3D, rectangles, quad_surface_point = augmentor_utils.random_translation_along_x(
                point_cloud, surface_points, boxes3D, rectangles, quad_surface_point, self.noise_traslation_std)
            point_cloud, surface_points, boxes3D, rectangles, quad_surface_point = augmentor_utils.random_translation_along_y(
                point_cloud, surface_points, boxes3D, rectangles, quad_surface_point, self.noise_traslation_std)
            # point_cloud, surface_points, boxes3D, rectangles, quad_surface_point = augmentor_utils.random_translation_along_x(
            #     point_cloud, surface_points, boxes3D, rectangles, quad_surface_point)


        class_ind = [self.dataset_config.shapenetid2class[x] for x in classes]

        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_surface_points = np.zeros((MAX_NUM_OBJ, 1024, 3))  # surface point num
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        object_instance_labels = np.zeros((MAX_NUM_OBJ,))
        num_gt_objects = np.zeros((self.num_proposal))
        num_gt_objects += boxes3D.shape[0]
        # target_bbox_instance_ids = -1 * np.ones((MAX_NUM_OBJ,MAX_INSTANCE_PT_NUMBER)).astype(np.int64) #每个instance最多xx个点

        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:boxes3D.shape[0]] = class_ind
        size_residuals[0:boxes3D.shape[0], :] = boxes3D[:, 3:6] - self.dataset_config.mean_size_arr[class_ind, :]
        target_bboxes_mask[0:boxes3D.shape[0]] = 1
        target_bboxes[0:boxes3D.shape[0], :] = boxes3D[:, 0:6]
        surface_points = surface_points.reshape(boxes3D.shape[0], -1, 3)
        target_surface_points[0:boxes3D.shape[0], :, :] = surface_points
        object_instance_labels[0:boxes3D.shape[0]] = object_instance_ids

        obj_angle_class, obj_angle_residuals = self.dataset_config.angle2class(boxes3D[:, 6])
        angle_classes[0:boxes3D.shape[0]] = obj_angle_class
        angle_residuals[0:boxes3D.shape[0]] = obj_angle_residuals
        original_pc_num = len(point_cloud)
        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_vote_mask[choices]
        point_votes = point_votes[choices, 1:]
        point_instance_labels = point_instance_labels[choices]
        # object_instance_ids_all = shapenet_instance_ids[choices]
        ##todo goal: 已知原始点序号，获得在下采样点中的序号
        ## 原始点对应的下采样点序号， 若无则为-1

        ret_dict = {}
        '''For quad Detection'''
        target_quad_surface_points = np.zeros((MAX_NUM_OBJ, 1024, 3))  # surface point num
        target_quad_centers = np.zeros((MAX_NUM_QUAD, 3))
        target_normal_vectors = np.zeros((MAX_NUM_QUAD, 3))
        target_quad_sizes = np.zeros((MAX_NUM_QUAD, 2))
        target_quad_mask = np.zeros((MAX_NUM_QUAD))
        target_horizontal_quads = np.zeros((4, 4, 3))
        num_total_quads = np.zeros((self.num_quad_proposal))
        num_gt_quads = np.zeros((self.num_quad_proposal))
        if use_quad and rectangles.shape[0] > 0:
            quad_votes = quad_votes[choices]
            quad_surface_points = quad_surface_points.reshape(quad_shape_num, -1, 3)
            target_quad_surface_points[0:quad_num, :, :] = quad_surface_points
            target_quad_mask[0:quad_num] = 1
            target_quad_centers[0:rectangles.shape[0], :] = rectangles[:, 0:3]
            target_normal_vectors[0:rectangles.shape[0], :] = rectangles[:, 3:6]
            target_quad_sizes[0:rectangles.shape[0], :] = rectangles[:, 6:8]
            num_gt_quads += rectangles.shape[0]
            num_total_quads += total_quad_num
            if len(horizontal_quads) > 0:
                target_horizontal_quads[0:len(horizontal_quads), :] = horizontal_quads[:, :, :]

        if self.use_quad:
            ret_dict['use_quad'] = int(use_quad) #use_quad.astype(np.int16)
            ret_dict['num_total_quads'] = num_total_quads.astype(np.int64)
            ret_dict['num_gt_quads'] = num_gt_quads.astype(np.int64)
            ret_dict['gt_quad_centers'] = target_quad_centers.astype(np.float32)
            ret_dict['gt_quad_sizes'] = target_quad_sizes.astype(np.float32)
            ret_dict['gt_normal_vectors'] = target_normal_vectors.astype(np.float32)
            ret_dict['horizontal_quads'] = target_horizontal_quads.astype(np.float32)
            ret_dict['quad_surface_points'] = target_quad_surface_points.astype(np.float32)
            ret_dict['target_quad_mask'] = target_quad_mask.astype(np.float32)
            ret_dict['quad_vote_label'] = quad_votes.astype(np.float32)

        '''For Object Detection'''
        ret_dict['scan_name'] = scan_name
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['choices'] = choices
        ret_dict['floor_height'] = np.array(floor_height).astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:boxes3D.shape[0]] = class_ind
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['surface_points'] = target_surface_points.astype(np.float32)
        ret_dict['num_gt_objects'] = num_gt_objects.astype(np.int64)
        ret_dict['point_instance_labels'] = point_instance_labels.astype(np.int64)
        ret_dict['shapenet_catids'] = shapenet_catids
        ret_dict['shapenet_ids'] = shapenet_ids
        ret_dict['object_instance_labels'] = object_instance_labels.astype(np.float32)

        if self.phase == 'prepare_data' or self.phase =='final_recon_with_rgb'  or self.phase =='detection_with_rgb':
            if self.mode != 'test':
                ret_dict['instance_pts'] = target_bbox_instance_pts.astype(np.float32)
                ret_dict['instance_pts_numbers'] = target_bbox_instance_pt_numbers.astype(np.int64)
                # ret_dict['shapenet_instance_labels'] = object_instance_ids_all
                # start = time()
                with open(os.path.join(self.posed_images_path2, scan_name, 'output_instance2imageid.pkl'), 'rb') as f:
                    instance2imageid = pickle.load(f)
                ret_dict['instance2imageid'] =  instance2imageid
            # print('load instance2imageid time: %s'%(time()-start))
            # start = time()
            with open(os.path.join(self.posed_images_path, scan_name, 'ptids_images.pkl'), 'rb') as f:
                ptids_images = pickle.load(f)
            ret_dict['ptids_images'] = ptids_images
            # print('load ptids_images time: %s'%(time()-start))

        return ret_dict


    def get_shapenet_points(self, boxid, transform=None):
        '''Load points and corresponding occ values.'''
        points_dict = np.load(os.path.join(self.scene_root_path, 'obj_points_for_completion_%s.npz' % boxid))
        points = points_dict['points']
        occupancies = points_dict['occ']
        sdf = points_dict['sdf']
        # Break symmetry if given in float16:
        if points.dtype == np.float16 and self.mode == 'train':
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)
        if self.points_unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]  # 整数转化为二进制数
        occupancies = occupancies.astype(np.float32)
        sdf = sdf.astype(np.float32)
        data = {'points': points, 'occ': occupancies, 'sdf': sdf}
        if transform is not None:
            data = transform(data)
        return data

def get_class_frequencies(cfg):
    mode = 'train'
    split_file = os.path.join(cfg.config['data']['split'], 'scannetv2_' + mode + '.json') #
    input_path = cfg.config['data']['input_path']
    split = read_json(split_file)
    cls = []
    for pth in split:
        scan_name = pth['bbox'].split('/')[-2]
        input_scene_pth = os.path.join(input_path, mode,'data', scan_name)
        proposal_ids = np.load(os.path.join(input_scene_pth, 'proposal_ids.npy'))
        with open(os.path.join('/home/dmy/data', pth['bbox']), 'rb') as file:
            box_info = pickle.load(file)
        box_num = len(glob(os.path.join(input_scene_pth, 'object_input_features_*.npy')))
        for boxid in range(box_num):
            gtid = proposal_ids[boxid][1]
            cls.append(list(ScanNet_OBJ_CLASS_IDS).index(box_info[gtid]['cls_id']))
    return cls

def get_shape_frequencies(cfg,mode):
    split_file = os.path.join(cfg.config['data']['split'], 'scannetv2_' + mode + '.json')  #
    input_path = cfg.config['data']['input_path']
    split = read_json(split_file)
    shapenet_ids = []
    frequecies = []

    for pth in split:
        with open(os.path.join(data_root_path, pth['bbox']), 'rb') as file:
            box_info = pickle.load(file)
        scan_name = pth['bbox'].split('/')[-2]
        input_scene_pth = os.path.join(input_path, mode, 'data', scan_name)
        proposal_ids = np.load(os.path.join(input_scene_pth, 'proposal_ids.npy'))

        box_num = len(glob(os.path.join(input_scene_pth, 'object_input_features_*.npy')))
        for boxid in range(box_num):
            gtid = proposal_ids[boxid][1]
            shapenet_id = box_info[gtid]['shapenet_id']
            shapenet_ids.append(shapenet_id)

    Shapeid_counts = Counter(shapenet_ids)
    mean_freq = round(sum(Shapeid_counts.values())/len(Shapeid_counts))
    for sid in shapenet_ids:
        frequecies.append(mean_freq/(len(shapenet_ids) * Shapeid_counts[sid]))

    return np.array(frequecies)


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
        if key not in ['shapenet_catids', 'shapenet_ids','instance2imageid','ptids_images']:
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

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

def ISCNet_dataloader(cfg, mode='train'):
    dataset = AnchorRec_ScanNet(cfg, mode)
    sampler = None
    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=cfg.config[mode]['batch_size'],
                            # shuffle= mode =='train' ,
                            collate_fn=collate_fn,
                            sampler = None,
                            # sampler=WeightedRandomSampler(sampling_p, len(dataset),replacement=True),
                            drop_last=(mode=='train'),
                            generator= g,
                            worker_init_fn = worker_init_fn)

    return dataset, dataloader, sampler


def check_instance_label_by_shapenet():
    from utils.pc_util import write_ply_color, write_ply, write_oriented_bbox
    from tqdm import tqdm
    scan_names = ["scene0616_00"]
    for scan_name in tqdm(scan_names):
        box_path = os.path.join(scannet_processed_path2, scan_name, 'bbox.pkl')
        if not os.path.exists(box_path) :
            print("%s path doesn\'t exist"%scan_name)
            continue
        with open( box_path, 'rb') as file:
            box_info = pickle.load(file)
        data_path = os.path.join(scannet_processed_path,scan_name,'full_scan.npz')
        scan_data = np.load(data_path)
        point_cloud = scan_data['mesh_vertices'][:, :3]
        boxid = 0
        for item in box_info:
            valid_ids = item['shapenet_instance_inds']
            instance_pts = point_cloud[valid_ids]
            box3D = item['box3D']
            box3D[3:6] += 0.2
            write_oriented_bbox(np.expand_dims(box3D,0),'temp/%s_box%s_box.ply'%(scan_name, boxid))
            write_ply_color(instance_pts,np.array([255,0,0]), 'temp/%s_box%s_instance_pts.ply'%(scan_name, boxid))
            if not os.path.exists('temp/%s.ply'%scan_name):
                write_ply(point_cloud[np.random.choice(len(point_cloud), 50000)],'temp/%s.ply'%scan_name)
            boxid += 1

def delete_repeated_boxes():
    from tqdm import tqdm
    from utils.pc_util import write_ply_color, write_ply, write_oriented_bbox
    scan_names = os.listdir(scannet_processed_path2)
    max_num_pts = 0
    num_pt_list = []
    for scan_name in tqdm(scan_names):
        box_path = os.path.join(scannet_processed_path2, scan_name, 'bbox.pkl')
        if not os.path.exists(box_path) :
            print("%s path doesn\'t exist"%scan_name)
            continue
        with open( box_path, 'rb') as file:
            box_info = pickle.load(file)

        for boxid1, item1 in enumerate(box_info):
            valid_ids1 = item1['shapenet_instance_inds']
            num_pts = len(valid_ids1)
            num_pt_list.append(num_pts)
            if num_pts > max_num_pts:
                max_num_pts = num_pts
            # for boxid2, item2 in enumerate(box_info):
            #     if boxid2 == boxid1:
            #         continue
            #     valid_ids2 = item2['shapenet_instance_inds']
            #     intersect_ratio = (np.in1d(valid_ids1, valid_ids2)).sum() / len(valid_ids1)
            #     if intersect_ratio > 0.8:
            #         box3D_1 = item1['box3D']
            #         box3D_2 = item2['box3D']
            #         print("%s: box %s intersect box %s with ratio of %s"%(scan_name, boxid1, boxid2,intersect_ratio))
            #         write_oriented_bbox(np.expand_dims(box3D_1 , 0), 'temp/%s_box%s.ply' % (scan_name, boxid1))
            #         write_oriented_bbox(np.expand_dims(box3D_2 , 0), 'temp/%s_box%s.ply' % (scan_name, boxid2))
    print('max_num_pts: %s'%max_num_pts)


def parse_groundtruths_from_raw(scan_name):
    # scan_data = np.load(os.path.join(scannet_processed_path, scan_name,"full_scan.npz"))
    with open(os.path.join(scannet_processed_path2, scan_name, 'bbox.pkl'), 'rb') as file:
        box_info = pickle.load(file)
    boxes3D = []
    cls_ids = []
    # mesh_zs = []
    box_id = 0
    for item in box_info:
        boxes3D.append(item['box3D'])
        cls_ids.append(item['cls_id'])
        box_id += 1
        # mesh_zs.append(self.zs_dataset[shapenet_catid + '/' + shapenet_id])
    boxes3D = np.array(boxes3D)
    sem_cls_labels = [dataset_config.shapenetid2class[x] for x in cls_ids]
    box_centers = boxes3D[:,:3]
    box_sizes = boxes3D[:,3:6]
    box_heading_angles = boxes3D[:,-1]
    sem_cls_labels = np.array(sem_cls_labels)
    gt_center_upright_camera = flip_axis_to_camera(box_centers)
    num_targets = len(boxes3D)
    gt_corners_3d_upright_camera = np.zeros((num_targets, 8, 3))
    box_label_mask = np.ones(num_targets).astype(np.float32) #.astype(np.bool)
    for nid in range(num_targets):
        corners_3d_upright_camera = get_3d_box(box_sizes[nid], - box_heading_angles[nid],
                                               gt_center_upright_camera[nid])
        gt_corners_3d_upright_camera[nid] = corners_3d_upright_camera

    return {'sem_cls_label': sem_cls_labels,
            'gt_corners_3d_upright_camera': gt_corners_3d_upright_camera,
            'heading_angles': box_heading_angles,
            'box_sizes': box_sizes,
            'box_centers': box_centers,
            'box_label_mask': box_label_mask}


# if __name__ == '__main__':
    # delete_repeated_boxes()
    # check_instance_label_by_shapenet()
    # make_instance_label_by_shapenet('/home/lht/dmy/data/processed_data2')