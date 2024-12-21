import os
import pickle
import numpy as np
import torch.utils.data
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path, ShapeNetv2_imcomplete_path
from configs.path_config import scannet_path, ScanNet_OBJ_CLASS2ID, ScanNet_OBJ_CLASS_NAMES, posed_images_path, scannet_processed_path,scannet_processed_path2, instance_gt_rgb_path
from utils.pc_util import write_ply
import cv2
import h5py
import albumentations as A
from utils.scannet.extract_posed_images import vis_intersection,crop_image,\
    vis_projected_points, get_random_range, rescale_pts2d, adjust_image_range
from net_utils.libs import softmax2
default_collate = torch.utils.data.dataloader.default_collate

MAX_NUM_OBJ = 64
MAX_NUM_QUAD = 32
MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673])
MAX_INSTANCE_PT_NUMBER = 2000


'''Proposal-wise dataset for training the RGB branch. Organize proposal-wise  so that proposals from different scenes can be collated to a single batch'''
class AnchorRec_ScanNet_For_Refine_single():
    def __init__(self, cfg, mode):
        super(AnchorRec_ScanNet_For_Refine_single, self).__init__()
        self.mode = mode
        self.augment = True
        self.cfg = cfg
        self.min_pts2d_number = cfg.config['data']['min_pts2d_number']
        self.min_pts2d_range = cfg.config['data']['min_pts2d_range']
        self.image_size = cfg.config['data'].get('image_size', 400)
        self.shift_range = cfg.config['data'].get('shift_range', 20)
        self.rescale_ratio_min = cfg.config['data'].get('rescale_ratio_min',0.7)
        self.rescale_ratio_max = cfg.config['data'].get('rescale_ratio_max',1.3)
        self.input_path = cfg.config['data']['input_path']
        self.dataset_config = cfg.dataset_config
        self.num_proposal = cfg.config['data']['num_proposal']
        self.vis_scan_names = cfg.config['log'].get('vis_scan_names',None)
        self.load_point_data(cfg, mode)

    def load_point_data(self, cfg, mode):
        self.scan_names = []
        self.geo_params = []
        self.sem_params = []
        # self.anchor_sampled_pts_list = []
        self.input_image_paths = []
        self.pts2d_range_list = []
        # self.valid_masks = []
        self.vote_features = []
        self.anchor_features = []
        # self.aggregated_vote_xyz = []
        # self.anchors = []
        # self.original_pc_inds_list = []
        self.gt_sem_cls = []
        self.objectness_label = []

        input_path = cfg.config['data']['input_path']
        with open(os.path.join(input_path, '%s.pkl'%mode), 'rb') as f:
        # with open(os.path.join(input_path, 'val.pkl'), 'rb') as f:
            data = pickle.load(f)
        scan_names_list = data['scan_names']
        geo_params_list = data['geo_params']
        sem_params_list = data['sem_params'] # / self.weight
        input_image_paths_list = data['input_image_paths']
        pts2d_range_list = data['input_pts2d_range_list']
        valid_masks_list = data['valid_mask']
        vote_features_list = data['vote_features']
        anchor_features_list = data['anchor_features']
        # aggregated_vote_xyz_list = data['aggregated_vote_xyz']
        shapenet_catids_list = data['shapenet_ids_list']
        objectness_label_list = data['objectness_label']

        data_num = len(scan_names_list)
        nproposal = geo_params_list[0].shape[0]
        for iter in range(data_num):
            sem_params = sem_params_list[iter]

            objectness_scores = sem_params[:, :2]
            objectness_probs = softmax2(objectness_scores, dim=1)[:, 1]
            if self.num_proposal < 256:
                ''' Only maintain part of the proposals with higher objectness'''
                objectness_probs_ = objectness_probs.copy()
                objectness_probs_.sort()
                objectness_prob_thresh = objectness_probs_[-self.num_proposal]
                mask = objectness_probs >= objectness_prob_thresh
                if mask.sum() > self.num_proposal:
                    false_proposal_num = mask.sum()
                    iid = np.where(objectness_probs == objectness_prob_thresh)[0]
                    mask[iid[:false_proposal_num - self.num_proposal]] = False
            for nid in range(nproposal):
                if not mask[nid] or not valid_masks_list[iter][nid] or shapenet_catids_list[iter][nid]==0:
                    continue
                self.scan_names.append(scan_names_list[iter])
                self.sem_params.append(sem_params_list[iter][nid])
                self.input_image_paths.append(input_image_paths_list[iter][nid])
                self.pts2d_range_list.append(pts2d_range_list[iter][nid])
                self.vote_features.append(vote_features_list[iter][nid])
                self.anchor_features.append(anchor_features_list[iter][nid])
                self.objectness_label.append(objectness_label_list[iter][nid])
                self.gt_sem_cls.append(ScanNet_OBJ_CLASS2ID [shapenet_catids_list[iter][nid]])

    def __getitem__(self, idx):
        # 序号相关
        ret_dict = {}
        ret_dict['scan_name'] = self.scan_names[idx]
        ret_dict['sem_params'] = self.sem_params[idx].astype(np.float32)
        ret_dict['vote_features'] = self.vote_features[idx].astype(np.float32)
        ret_dict['anchor_features'] = self.anchor_features[idx].astype(np.float32)
        ret_dict['objectness_label'] = self.objectness_label[idx]
        ret_dict['gt_sem_cls'] = int(self.gt_sem_cls[idx]) #.astype(np.int64)
        # input_num_proposal = self.geo_params[idx].shape[0]
        input_image_path = posed_images_path+'/'+'/'.join(self.input_image_paths[idx].split('/')[-2:]) if len(self.input_image_paths[idx])>0 else ''
        pts2d_range = self.pts2d_range_list[idx]
        rescale_ratio = 1
        shift_range = 0
        if self.augment:
            rescale_ratio = np.random.uniform(self.rescale_ratio_min,self.rescale_ratio_max)
            shift_range = np.random.randint(self.shift_range)

        valid_mask = True
        objectness_label = self.objectness_label[idx]

        image = cv2.imread(input_image_path)
        if image is None: #In case of file corruption
            image = np.zeros((self.image_size, self.image_size, 3)).astype(np.float32)
            valid_mask = False
            objectness_label  = False
        new_range = adjust_image_range(image, pts2d_range, margin=10, shift_range=shift_range,
                                     scale_ratio=rescale_ratio)
        minx, miny, maxx, maxy = new_range
        '''Filter out proposals with too small image regions'''
        if (maxx - minx) < self.min_pts2d_range or (maxy - miny) < self.min_pts2d_range:
            image = np.zeros((self.image_size, self.image_size, 3)).astype(np.float32)
            valid_mask = False
            objectness_label  = False
        else:
            cropped_image = image[miny: maxy, minx: maxx]  # image coordinate system is different with point corrdinate system
            cropped_image = cv2.resize(cropped_image, (self.image_size, self.image_size))
            if self.augment:  # Image augmentation
                transform = A.Compose([
                    # A.RandomCrop(width=256, height=256),
                    # A.HorizontalFlip(p=0.5),
                    # A.transforms.Equalize(),
                    A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0., p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.MotionBlur(p=0.4),
                    A.Defocus(p=0.2),
                    # A.crops.transforms.RandomCrop(int(w * rescale_ratio), int(h * rescale_ratio), p=1) #若rescale, pts2d也要相应变化
                ])
                cropped_image = transform(image=cropped_image)['image']
            cropped_image = (cropped_image - MEAN_COLOR_RGB) / 255.0
            image = cropped_image
        ret_dict['img'] = image.astype(np.float32)
        ret_dict['valid_mask'] = valid_mask #.astype(np.int64)
        ret_dict['objectness_label'] = objectness_label #.astype(np.int64)
        return ret_dict

    def __len__(self):
        # assert len(self.points_for_completion_paths) == len(self.input_features_paths)
        return len(self.scan_names)

'''Scene-wise dataset for testing the RGB branch'''
class AnchorRec_ScanNet_For_Refine():
    def __init__(self, cfg, mode, augment = True):
        super(AnchorRec_ScanNet_For_Refine, self).__init__()
        self.augment = mode == 'train' if augment else False
        self.mode = mode
        self.cfg = cfg
        self.min_pts2d_number = cfg.config['data']['min_pts2d_number']
        self.min_pts2d_range = cfg.config['data']['min_pts2d_range']
        self.image_size = cfg.config['data'].get('image_size', 400)
        self.shift_range = cfg.config['data'].get('shift_range', 20)
        self.rescale_ratio_min = cfg.config['data'].get('rescale_ratio_min',0.7)
        self.rescale_ratio_max = cfg.config['data'].get('rescale_ratio_max',1.3)
        self.input_path = cfg.config['data']['input_path']
        self.dataset_config = cfg.dataset_config
        self.num_proposal = cfg.config['data']['num_proposal']
        if mode=='test' or 'val':
            self.num_proposal = 256
        # self.w_sem_vote = np.load('out/detection_w_rgb/no_rgb/1/w_sem_vote.npy')
        # self.w_sem_anchor = np.load('out/detection_w_rgb/no_rgb/1/w_sem_anchor.npy')
        # self.weight = self.w_sem_anchor + self.w_sem_vote
        self.vis_scan_names = cfg.config['log'].get('vis_scan_names',None)
        if self.vis_scan_names is not None and len(self.vis_scan_names)>0:
            self.load_point_data_for_vis(cfg, mode)
        else:
            self.load_point_data(cfg, mode)
        # self.load_point_data_small(cfg, mode) （# for debugging on small dataset)

    def load_point_data(self, cfg, mode):
        self.scan_names = []
        self.geo_params = []
        self.sem_params = []
        self.anchor_sampled_pts_list = []
        self.input_image_paths = []
        self.pts2d_range_list = []
        self.valid_masks = []
        self.vote_features = []
        self.anchor_features = []
        self.aggregated_vote_xyz = []
        self.anchors = []
        self.original_pc_inds_list = []

        input_path = cfg.config['data']['input_path']
        with open(os.path.join(input_path, '%s.pkl'%mode), 'rb') as f:
        # with open(os.path.join(input_path, 'val.pkl'), 'rb') as f:
            data = pickle.load(f)
        self.scan_names = data['scan_names']
        self.geo_params = data['geo_params']
        self.sem_params = data['sem_params'] # / self.weight
        self.anchor_sampled_pts_list = data['anchor_sampled_pts_list']
        self.input_image_paths = data['input_image_paths']
        self.pts2d_range_list = data['input_pts2d_range_list']
        self.valid_masks = data['valid_mask']
        self.vote_features = data['vote_features']
        self.anchor_features = data['anchor_features']
        self.aggregated_vote_xyz = data['aggregated_vote_xyz']
        self.anchors = data['anchors']
        self.original_pc_inds_list = data['original_pc_inds_list']
        # self.objectness_label_list = data['objectness_label']

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        ret_dict_gts = load_scannet_data_by_scan_name(scan_name, self.cfg.dataset_config, num_proposal=self.num_proposal)
        del ret_dict_gts['point_cloud']
        objectness_scores = self.sem_params[idx][:,:2]
        objectness_probs = softmax2(objectness_scores, dim=1)[:, 1]
        mask = objectness_probs >= 0
        if self.num_proposal < 256:
            objectness_probs_ = objectness_probs.copy()
            objectness_probs_.sort()
            objectness_prob_thresh = objectness_probs_[-self.num_proposal]
            mask = objectness_probs >= objectness_prob_thresh
            if mask.sum() >self.num_proposal:
                false_proposal_num = mask.sum()
                iid = np.where(objectness_probs == objectness_prob_thresh)[0]
                mask[iid[:false_proposal_num-self.num_proposal]]= False
        ret_dict = {}
        ret_dict['scan_name'] = scan_name
        ret_dict['geo_params'] = self.geo_params[idx][mask]
        ret_dict['sem_params'] = self.sem_params[idx][mask]
        ret_dict['aggregated_vote_xyz'] = self.aggregated_vote_xyz[idx][mask]
        ret_dict['objectness_probs'] = objectness_probs[mask]
        ret_dict['anchor_sampled_xyzs'] = self.anchor_sampled_pts_list[idx][mask]
        ret_dict['vote_features'] = self.vote_features[idx][mask]
        ret_dict['anchor_features'] = self.anchor_features[idx][mask]
        ret_dict['anchors'] = self.anchors[idx][mask]
        batch_valid_masks = self.valid_masks[idx][mask]
        input_image_paths = [pth for i, pth in enumerate(self.input_image_paths[idx]) if mask[i]]
        pts2d_ranges = self.pts2d_range_list[idx][mask]
        rescale_ratio = 1
        shift_range = 0
        if self.augment:
            rescale_ratio = np.random.uniform(self.rescale_ratio_min,self.rescale_ratio_max)
            shift_range = np.random.randint(self.shift_range)

        batch_images = []
        # anchor_sampled_xyzs = np.zeros((self.num_proposal, 500, 3))
        for i in range(self.num_proposal):
            if  len(input_image_paths[i])>0 and type(input_image_paths[i])==str:
                input_image_path = posed_images_path+'/'+'/'.join(input_image_paths[i].split('/')[-2:])
            else:
                input_image_path =''
            pts2d_range = pts2d_ranges[i]
            if len(input_image_path)>0:
                image = cv2.imread(input_image_path)
                if image is None: #文件损坏： scene0608_01/00047.jpg； 00001.jpg
                    batch_images.append(np.zeros((self.image_size, self.image_size, 3)).astype(np.float32))
                    batch_valid_masks[i] = 0
                    continue
                new_range = adjust_image_range(image, pts2d_range, margin=10, shift_range=shift_range,
                                             scale_ratio=rescale_ratio)
                minx, miny, maxx, maxy = new_range
                if (maxx - minx) < self.min_pts2d_range or (maxy - miny) < self.min_pts2d_range:
                    batch_images.append(np.zeros((self.image_size, self.image_size, 3)).astype(np.float32))
                    batch_valid_masks[i] = 0
                else:
                    ret_dict['valid_mask'] = 1
                    cropped_image = image[miny: maxy, minx: maxx]  # + 1
                    cropped_image = cv2.resize(cropped_image, (self.image_size, self.image_size))
                    if self.augment:  # 更大的增强
                        transform = A.Compose([
                            # A.RandomCrop(width=256, height=256),
                            # A.HorizontalFlip(p=0.5),
                            # A.transforms.Equalize(),
                            A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0., p=1),
                            A.RandomBrightnessContrast(p=1),
                            A.MotionBlur(p=0.4),
                            A.Defocus(p=0.2),
                            # A.crops.transforms.RandomCrop(int(w * rescale_ratio), int(h * rescale_ratio), p=1) #若rescale, pts2d也要相应变化
                        ])
                        cropped_image = transform(image=cropped_image)['image']
                    cropped_image = (cropped_image - MEAN_COLOR_RGB) / 255.0
                    batch_images.append(cropped_image.astype(np.float32))
            else:
                batch_images.append(np.zeros((self.image_size, self.image_size, 3)).astype(np.float32))
                batch_valid_masks[i] = 0

        ret_dict['img'] = np.concatenate([np.expand_dims(ii,0) for ii in batch_images])
        ret_dict['valid_mask'] = batch_valid_masks
        ret_dict.update(ret_dict_gts)
        return ret_dict

    def __len__(self):
        # assert len(self.points_for_completion_paths) == len(self.input_features_paths)
        return len(self.scan_names)

'''Proposal-wise dataset for training the reconstruction branch'''
class SingleObject_recon_new():
    def __init__(self, cfg, mode):
        super(SingleObject_recon_new, self).__init__()
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
        self.input_rgb_features = []
        self.input_points = []
        self.boxids = []
        self.rgb_features = []
        # images
        if self.output_images:
            self.input_image_paths = []
            self.pts2d_range_list = []
            self.valid_masks = []
            self.anchor_list = []
        input_path = cfg.config['data']['input_path']
        input_rgb_path = cfg.config['data'].get('rgb_path', None)
        if input_rgb_path ==None:
            self.use_rgb = False
            with open(os.path.join(input_path, '%s.pkl' % mode), 'rb') as f:
                data = pickle.load(f)
        else:
            with open(os.path.join(input_rgb_path, '%s.pkl' % mode), 'rb') as f:
                rgb_feat = pickle.load(f)
            with open(os.path.join(input_path, '%s.pkl' % mode), 'rb') as f:
                data = pickle.load(f)
            data = {**data, **rgb_feat}


        input_point_list = data['anchor_sampled_pts_list'] # _xyzs
        valid_mask_list = data['valid_mask'] #_list
        if self.use_rgb:
            rgb_feat_data = data['rgb_features']
        if self.output_images:
            input_image_paths = data['input_image_paths']
            pts2d_range_list = data['input_pts2d_range_list']
            anchor_list = data['anchors']

        objectness_label_list = data['objectness_label']
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
                mask = objectness_label_list[iter][nid]
                if self.use_rgb:
                    mask = mask * valid_mask_list[iter][nid]
                if mask:
                    assert type(shapenet_ids_list[iter][nid]) == str
                    self.gt_shapenet_ids.append(shapenet_ids_list[iter][nid])
                    self.gt_shapenet_catids.append(shapenet_catids_list[iter][nid])
                    self.input_points.append(input_point_list[iter][nid])
                    self.input_vote_features.append(vote_features[iter][nid])
                    self.input_anchor_features.append(anchor_features[iter][nid])
                    self.scan_names.append(scan_names[iter])
                    self.boxids.append(nid)
                    if self.use_rgb:
                        self.input_rgb_features.append(rgb_feat_data[iter][nid])
                    if self.output_images:
                        # proposal id 和 nid转换关系:  nid表示在pred_mask_ids中的序号； pred_mask_ids是包含的序号
                        proposal_id = pred_mask_ids[nid]
                        self.input_image_paths.append(input_image_paths[iter][proposal_id])
                        self.pts2d_range_list.append(pts2d_range_list[iter][proposal_id])
                        self.anchor_list.append(anchor_list[iter][proposal_id])

        unique_shapenet_ids = list(set(self.gt_shapenet_ids))
        unique_shapenet_ids.sort()
        shapenet2cls = {shape_id: i for i, shape_id in enumerate(unique_shapenet_ids)}
        self.cls_labels = [shapenet2cls[shape_id] for shape_id in self.gt_shapenet_ids]


    def load_point_data_test(self, cfg):
        '''bbox已经过滤过了'''
        self.scan_names = []
        # recon
        self.input_vote_features = []
        self.input_anchor_features = []
        self.input_rgb_features = []
        self.input_points = []
        self.bbox_parameters = []
        self.boxids = []
        self.rgb_features = []
        self.anchors = []
        input_path = cfg.config['data']['input_path']
        input_rgb_path = cfg.config['data'].get('rgb_path', None)
        # conf_thresh = cfg.config['data'].get('conf_thresh', 0.05)
        if input_rgb_path ==None:
            self.use_rgb = False
        if self.use_rgb:
            with open(os.path.join(input_rgb_path, 'test.pkl'), 'rb') as f:
                data = pickle.load(f)
        else:
            with open(os.path.join(input_path, 'test.pkl'), 'rb') as f: #test_no_rgb.pkl
                data = pickle.load(f)
        input_point_list = data['anchor_sampled_xyzs'] if 'anchor_sampled_xyzs' in data else data['anchor_sampled_pts_list']
        vote_features = data['vote_features']
        anchor_features = data['anchor_features']
        anchors = data['anchors']
        parsed_predictions_list = data['parsed_predictions']
        if self.use_rgb:
            rgb_feat_data = data['rgb_features']
            valid_mask_list = data['valid_mask_list']
        self.num_pts = len(input_point_list[0][0])

        scan_names = data['scan_names']
        scene_num = len(input_point_list)
        for iter in range(scene_num):
            if self.vis_scan_names is not None and scan_names[iter] not in self.vis_scan_names:
                continue
            num_proposal = input_point_list[iter].shape[0]
            parsed_prediction = parsed_predictions_list[iter]
            pred_centers = parsed_prediction['pred_centers']
            pred_sizes = parsed_prediction['pred_sizes']
            pred_headings = parsed_prediction['pred_headings']
            sem_cls_probs = parsed_prediction['sem_cls_probs']
            obj_probs = parsed_prediction['obj_prob']
            # pred_mask = parsed_prediction['pred_mask']
            bbox_params = np.concatenate([pred_centers, pred_sizes, pred_headings[:, np.newaxis], sem_cls_probs, obj_probs[:, np.newaxis]], -1)
            for nid in range(num_proposal):
                # if obj_probs[nid] < 0.05:
                #     continue
                if self.use_rgb and not valid_mask_list[iter][nid]:
                    continue
                if obj_probs[nid] < 0.05 or obj_probs[nid] >= 0.5:
                    continue
                # if not self.use_rgb and not pred_mask[nid]:
                #     continue
                self.input_points.append(input_point_list[iter][nid])
                self.input_vote_features.append(vote_features[iter][nid])
                self.input_anchor_features.append(anchor_features[iter][nid])
                self.scan_names.append(scan_names[iter])
                self.anchors.append(anchors[iter][nid])
                self.boxids.append(nid)
                self.bbox_parameters.append(bbox_params[nid])
                if self.use_rgb:
                    self.input_rgb_features.append(rgb_feat_data[iter][nid])

    def load_point_data_vis(self):
        ## mode = train/val
        self.scan_names = []
        # self.objectness_label = []
        self.gt_shapenet_ids = []
        self.gt_shapenet_catids = []
        self.cls_labels = []
        # recon
        self.input_vote_features = []
        self.input_anchor_features = []
        self.input_rgb_features = []
        self.input_points = []
        self.boxids = []
        # images
        self.input_image_paths = []
        self.pts2d_range_list = []
        self.valid_masks = []
        self.anchor_list = []
        if self.vis_compare:
            with open("out/only_detection/refine/2/test/best_45.58/cls_improved_by_rgb.txt","r") as f:
                improved_list = f.readlines()
            vis_instances = {}
            for item in improved_list:
                vis_scene_name = item.split(':')[0]
                vis_id = item.split(':')[1].strip()
                if vis_scene_name not in vis_instances:
                    vis_instances[vis_scene_name] = []
                vis_instances[vis_scene_name].append(int(vis_id))
            vis_scene_names = list(vis_instances.keys())
        with open('out/only_detection/refine/2/for_recon/test_with_correspondence_new.pkl', 'rb') as f:
            data = pickle.load(f)
        input_point_list = data['anchor_sampled_xyzs'] #  _pts_list
        valid_mask_list = data['valid_mask_list'] #_list
        pred_mask_list = data['sampled_ids']
        if self.use_rgb:
            rgb_feat_data = data['rgb_features']
            if self.output_images:
                with open('out/only_detection/prepare_data/test.pkl', 'rb') as f:
                    data2 = pickle.load(f)
                input_image_paths = data2['input_image_paths']
                pts2d_range_list = data2['input_pts2d_range_list']
                anchor_list = data2['anchors']
        objectness_label_list = data['objectness_label']
        shapenet_catids_list = data['shapenet_catids_list']
        shapenet_ids_list = data['shapenet_ids_list']
        vote_features = data['vote_features']
        anchor_features = data['anchor_features']
        self.num_pts = len(input_point_list[0][0])

        scan_names = data['scan_names']
        scene_num = len(input_point_list)
        for iter in range(scene_num):
            if self.vis_scan_names is not None and scan_names[iter] not in self.vis_scan_names:
                continue
            # if self.vis_compare and scan_names[iter] not in vis_scene_names:
            #     continue
            num_proposal = input_point_list[iter].shape[0]
            pred_mask_ids = np.where(pred_mask_list[iter])[0]
            for nid in range(num_proposal):
            # vis_instance_ids = vis_instances[scan_names[iter]] #proposal_ids
            # for proposal_id in vis_instance_ids:
            #     nid = np.where(pred_mask_ids == proposal_id)[0][0]
                mask = objectness_label_list[iter][nid]
                if self.use_rgb:
                    mask = mask * valid_mask_list [iter][nid]
                if mask:
                    assert type(shapenet_ids_list[iter][nid]) == str
                    self.gt_shapenet_ids.append(shapenet_ids_list[iter][nid])
                    self.gt_shapenet_catids.append(shapenet_catids_list[iter][nid])
                    self.input_points.append(input_point_list[iter][nid])
                    self.input_vote_features.append(vote_features[iter][nid])
                    self.input_anchor_features.append(anchor_features[iter][nid])
                    self.scan_names.append(scan_names[iter])
                    self.boxids.append(nid)
                    if self.use_rgb:
                        self.input_rgb_features.append(rgb_feat_data[iter][nid])
                        if self.output_images:
                            # proposal id 和 nid转换关系:  nid表示在pred_mask_ids中的序号； pred_mask_ids是包含的序号
                            proposal_id = pred_mask_ids[nid]
                            self.input_image_paths.append(input_image_paths[iter][proposal_id])
                            self.pts2d_range_list.append(pts2d_range_list[iter][proposal_id])
                            self.anchor_list.append(anchor_list[iter][proposal_id])

        #shapenet id缩略版编号
        unique_shapenet_ids = list(set(self.gt_shapenet_ids))
        unique_shapenet_ids.sort()
        shapenet2cls = {shape_id: i for i, shape_id in enumerate(unique_shapenet_ids)}
        self.cls_labels = [shapenet2cls[shape_id] for shape_id in self.gt_shapenet_ids]


    def __getitem__(self, idx):
        ret_dict = {}
        ret_dict['scan_name'] = self.scan_names[idx]

        input_points = self.input_points[idx]
        pts_from_gt = False
        if len(input_points)==0:
            assert self.augment and self.mode != 'test'
            pts_from_gt = True
            view_id = np.random.choice(8,1)[0]
            input_points = np.load(os.path.join(ShapeNetv2_imcomplete_path,self.gt_shapenet_catids[idx],
                                   self.gt_shapenet_ids[idx],'view_%s.npy'%view_id))
        if self.augment:
            if pts_from_gt:
                input_points = input_points[np.random.choice(len(input_points), self.num_pts)]
                # drop_ratio = 0.9
                jitter_std = 0.03
            else:
                # drop_ratio = 0.3
                jitter_std = 0.01
            drop_ratio = 0.2
            # jitter_std = 0.01
            input_points = self.pointcloudScale(input_points)
            input_points  = self.pointcloudJitter(input_points,std=jitter_std)
            input_points  = self.pointcloudDropFilled(input_points , drop_ratio = drop_ratio, std=0.02, clip=0.04)

        # semantic相关
        ret_dict['input_points'] =input_points.astype(np.float32) #normalized points
        ret_dict['input_vote_features'] = self.input_vote_features[idx].astype(np.float32)
        ret_dict['input_anchor_features'] = self.input_anchor_features[idx].astype(np.float32)
        ret_dict['box_id'] = self.boxids[idx]
        if self.use_rgb:
            # ret_dict['anchors'] = self.anchor_list[idx].astype(np.float32)
            ret_dict['input_rgb_features'] = self.input_rgb_features[idx].astype(np.float32)
            if self.output_images:
                pts2d_range = self.pts2d_range_list[idx]
                img_path = self.input_image_paths[idx]
                img_path = posed_images_path+'/'+'/'.join(img_path.split('/')[-2:])
                assert os.path.exists(img_path)
                image = cv2.imread(img_path)
                new_range = adjust_image_range(image, pts2d_range, margin=10, shift_range=0.,
                                             scale_ratio=1.)
                minx, miny, maxx, maxy =  new_range
                cropped_image = image[miny: maxy, minx: maxx]  # + 1
                cropped_image = cv2.resize(cropped_image, (256,256))
                ret_dict['img'] = cropped_image

        if self.mode =='test':
            ret_dict['bbox_params'] = self.bbox_parameters[idx].astype(np.float32)
            ret_dict['anchors'] = self.anchors[idx].astype(np.float32)

            return ret_dict

        ret_dict['shape_label'] = self.cls_labels[idx]
        if len(self.gt_shapenet_catids[idx])>0:
            ret_dict['zs'] = self. zs[self.gt_shapenet_catids[idx] + '/' +self.gt_shapenet_ids[idx]][0].astype(np.float32)
        else:
            ret_dict['zs'] = np.zeros((1,256)).astype(np.float32)
            ret_dict['mesh_pth'] = ""
            ret_dict['data_points'] = np.zeros((4096,4)).astype(np.float32)
            ret_dict['data_values'] = np.zeros((4096,1)).astype(np.float32)
            return ret_dict
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

'''using instance corresponding cropped RGB images for pretraining'''
class RGB_GT():
    def __init__(self, cfg, mode):
        super(RGB_GT, self).__init__()
        # self.augment = False
        self.augment = mode == 'train'
        self.mode = mode
        self.input_path =  instance_gt_rgb_path
        self.image_size = cfg.config['data']['image_size']
        with open(os.path.join(self.input_path,"%s.txt"%mode),'r') as f:
            self.split = f.readlines()
        # self.load_data()

    def load_data(self):
        self.imgs = []
        self.cls_ids = []
        # scan_names.sort()
        for filename in self.split:
            filename = filename.strip()
            abs_path = self.input_path +'/' + filename
            cls_id = int(filename.split('/')[-1].split('_')[-1].split('.')[0])
            self.cls_ids.append(cls_id)
            img = cv2.imread(abs_path)
            img_ = cv2.resize(img, (self.image_size, self.image_size))
            self.imgs.append(img_)


    def __getitem__(self, idx):
        ret_dict = {}
        # 序号相关
        filename = self.split[idx]
        filename = filename.strip()
        abs_path = self.input_path + '/' + filename
        cls_id = int(filename.split('/')[-1].split('_')[-1].split('.')[0])
        img = cv2.imread(abs_path)
        if img is None:
            filename = self.split[idx+1]
            filename = filename.strip()
            abs_path = self.input_path + '/' + filename
            cls_id = int(filename.split('/')[-1].split('_')[-1].split('.')[0])
            img = cv2.imread(abs_path)
        h, w,_ = img.shape
        # img = self.imgs[idx]
        # cls_id = self.cls_ids[idx]
        if self.augment: #更大的增强
            transform = A.Compose([
                A.Rotate(p=0.5),
                A.crops.transforms.RandomCrop(height= int(np.random.uniform(0.5,1.,1)* h) , width=int(np.random.uniform(0.5,1.,1)* w ), p=0.5),
                # A.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0., p=1),
                A.RandomBrightnessContrast (brightness_limit=(-0.2, 0.2), p=0.5),
                # (limit=0.2, p=0.5),
                A.MotionBlur(p=0.3),
                # A.Defocus(p=0.2),
                A.dropout.coarse_dropout.CoarseDropout(max_holes=4, max_height=30, max_width=30,min_holes=1, min_height=5, min_width=5,fill_value=0, p=0.5) #模拟遮挡
            ])
            img = transform(image=img)['image']
        img = cv2.resize(img, (self.image_size, self.image_size))
        '''depth-aware CNN normalization method '''
        img = (img - MEAN_COLOR_RGB)/ 255.0
        ret_dict['img'] = img.astype(np.float32)
        ret_dict['cls_id'] = cls_id
        return ret_dict

    def __len__(self):
        return len(self.split)


def denormalize_image(img):
    if torch.is_tensor(img):
       img = img.cpu().numpy()
    img = img * 255.0 +  MEAN_COLOR_RGB
    return img

def denormalize_image2(img):
    if torch.is_tensor(img):
       img = img.cpu().numpy()
    img = img  +  MEAN_COLOR_RGB
    return img


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

def check_data(pth, output_pth):
    posed_images_path = '/home/dmy/data/datasets/scannet/posed_images_final'
    with open(os.path.join(pth,'data.pkl'), 'rb') as f:
        data = pickle.load(f)
    for iter, shapenet_id in enumerate(data.keys()):
        shape_count = len(data[shapenet_id]['input_features'])
        selected = np.arange(shape_count)
        for i in selected:
            scan_name = data[shapenet_id]['scan_names'][i] # N,2
            if scan_name != 'scene0092_03':
                continue

            iou_with_gt = data[shapenet_id]['iou_with_gt'][i]
            sampling_number_per_anchor = data[shapenet_id]['sampling_number_per_anchor'][i]
            bad_anchor_number = (sampling_number_per_anchor < 1).sum()
            '''根据与gt的iou + bad anchor number(周围采样点数量为0的shape anchor) 删除错误预测的实例'''
            if iou_with_gt < 0.2 or bad_anchor_number > 45:
                continue

            projected_pt2d = np.array(data[shapenet_id]['projected_pts2d'][i])  # N,2
            img_id = data[shapenet_id]['img_ids'][i]  # N,2
            box_id = data[shapenet_id]['box_ids'][i] # N,2

            img_path = os.path.join(posed_images_path, scan_name, str(img_id).zfill(5) + '.jpg')  # glob(obj_%s*
            print(os.path.join(pth, scan_name, 'obj_%s.jpg' % box_id))
            image = cv2.imread(img_path)
            vis_img = image.copy()
            vis_projected_points(projected_pt2d, vis_img, os.path.join(output_pth,'iter_%s_id_%s_(%s_boxid%s)' % (iter,i, scan_name, box_id)), pointwidth=3)

            #
            pc_out_pth = os.path.join(output_pth, '%s_pc.ply' % scan_name)
            if not os.path.exists(pc_out_pth):
                scene_root_path = os.path.join(scannet_path, 'processed_data', scan_name)
                point_cloud = np.load(os.path.join(scene_root_path, 'full_scan.npz'))['mesh_vertices'][:, :3]
                write_ply(point_cloud, pc_out_pth)
            write_ply(data[shapenet_id]['obj_xyzs'][i],os.path.join(output_pth,'iter_%s_id_%s_(%s_boxid%s).ply' % (iter,i, scan_name, box_id)))

def load_scannet_data_by_scan_name(scan_name, dataset_config, num_proposal=256, collate_batch = True):
    ret_dict={}
    scan_data = np.load(os.path.join(scannet_processed_path, scan_name, "full_scan.npz"))
    point_cloud = scan_data['mesh_vertices'][:, :3]
    ret_dict['point_cloud'] = point_cloud
    ##read gt bbox data
    with open(os.path.join(scannet_processed_path2, scan_name, 'bbox.pkl'), 'rb') as file:
        box_info = pickle.load(file)
    boxes3D = []
    classes = []
    shapenet_catids = []
    shapenet_ids = []
    # shapenet_instance_ids_all = []
    box_id = 0
    # target_bbox_instance_ids = -1 * np.ones((self.num_proposal,MAX_INSTANCE_PT_NUMBER)).astype(np.int64) #每个instance最多xx个点
    if collate_batch:
        target_bbox_instance_pt_numbers = np.zeros((num_proposal,)).astype(np.int64)  # 每个instance最多xx个点
        target_bbox_instance_pts = np.zeros((num_proposal, MAX_INSTANCE_PT_NUMBER, 3))
    else:
        target_bbox_instance_pts = []

    for gtid, item in enumerate(box_info):
        boxes3D.append(item['box3D'])
        classes.append(item['cls_id'])
        shapenet_catids.append(item['shapenet_catid'])
        shapenet_ids.append(item['shapenet_id'])

        shapenet_instance_inds = item['shapenet_instance_inds']
        pt_number = len(shapenet_instance_inds)
        if pt_number > MAX_INSTANCE_PT_NUMBER:
            shapenet_instance_inds = shapenet_instance_inds[
                np.random.choice(pt_number, MAX_INSTANCE_PT_NUMBER, replace=False)]
        if collate_batch:
            target_bbox_instance_pts[gtid, :pt_number, :] = point_cloud[shapenet_instance_inds]
            target_bbox_instance_pt_numbers[gtid] = pt_number
        else:
            target_bbox_instance_pts.append(point_cloud[shapenet_instance_inds])
        box_id += 1
        # mesh_zs.append(self.zs_dataset[shapenet_catid + '/' + shapenet_id])
    boxes3D = np.array(boxes3D)
    class_ind = [dataset_config.shapenetid2class[x] for x in classes]
    obj_angle_class, obj_angle_residuals = dataset_config.angle2class(boxes3D[:, 6])


    if collate_batch:
        size_classes = np.zeros((num_proposal,))
        size_residuals = np.zeros((num_proposal, 3))
        target_bboxes_mask = np.zeros((num_proposal))
        target_bboxes = np.zeros((num_proposal, 6))
        angle_classes = np.zeros((num_proposal,))
        angle_residuals = np.zeros((num_proposal,))
        target_bboxes_semcls = np.zeros((num_proposal))
        num_gt_objects = np.zeros((num_proposal))
        num_gt_objects += boxes3D.shape[0]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:boxes3D.shape[0]] = class_ind
        size_residuals[0:boxes3D.shape[0], :] = boxes3D[:, 3:6] - dataset_config.mean_size_arr[class_ind, :]
        target_bboxes_mask[0:boxes3D.shape[0]] = 1
        target_bboxes[0:boxes3D.shape[0], :] = boxes3D[:, 0:6]
        angle_classes[0:boxes3D.shape[0]] = obj_angle_class
        angle_residuals[0:boxes3D.shape[0]] = obj_angle_residuals
        target_bboxes_semcls[0:boxes3D.shape[0]] = class_ind
    else:
        target_bboxes = boxes3D
        size_classes = class_ind
        size_residuals = boxes3D[:, 3:6] - dataset_config.mean_size_arr[class_ind, :]
        angle_classes = obj_angle_class
        angle_residuals = obj_angle_residuals
        target_bboxes_semcls = class_ind

    ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
    ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
    ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
    ret_dict['size_class_label'] = size_classes.astype(np.int64)
    ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
    ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
    ret_dict['shapenet_catids'] = shapenet_catids
    ret_dict['shapenet_ids'] = shapenet_ids
    if collate_batch:
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['num_gt_objects'] = num_gt_objects.astype(np.int64)
        ret_dict['instance_pts_numbers'] = target_bbox_instance_pt_numbers.astype(np.int64)
    ret_dict['instance_pts'] = target_bbox_instance_pts.astype(np.float32)
    return ret_dict




