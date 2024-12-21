from models.registers import MODULES
import torch
from torch import nn
import numpy as np
import os
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import  QueryFromAnchors_per_proposal
from net_utils.nn_distance import nn_distance_self
MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673])
from configs.path_config import posed_images_path

@MODULES.register_module
class InstanceFeatSampling(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(InstanceFeatSampling, self).__init__()
        self.optim_spec = optim_spec
        model_config = cfg.config['model']['GenerateInstanceFeat']
        # self.sample_num = cfg.config['model']['skip_propagation']['sample_num']
        # self.output_points_num = model_config['output_points_num']
        self.sample_radius = model_config['sample_radius']
        self.nsample = model_config['nsample']
        self.min_num_pts = model_config['min_num_pts']
        self.shift_range = model_config['shift_range']
        self.rescale_range = model_config['rescale_range']
        self.image_size = model_config['image_size']
        self.min_pts2d_range = model_config['min_pts2d_range']
    def forward(self, data, anchors, augment=True, return_images = True):
        input_point_cloud = data['point_clouds'][:,:,:3]
        point_clouds_sampled_ids = data['choices'].detach().cpu().numpy()
        if return_images:
            point_instance_labels = data['point_instance_labels'].detach().cpu().numpy()
            instance2imageid = data['instance2imageid']
            ptids_images = data['ptids_images']
        batch_size, nproposal, nanchor,_ = anchors.shape
        dist, _ = nn_distance_self(anchors.reshape(-1, nanchor, 3), square_root=True)
        '''For each proposal respectively, using the median distance between the anchors as the search radius of anchor-based sampling'''
        dist, _ = dist.median(dim=-1)# BXK
        dist = dist.reshape(batch_size, nproposal)

        valid_mask = torch.ones(batch_size * nproposal) # To filter out proposals by anchor-based point/image searching
        output_image_paths = []
        output_pts2d_range_list = []

        anchor_sampled_pts_list = []
        original_pc_inds_list = []
        for bid in range(batch_size):
            '''load data'''
            scan_name = data['scan_name'][bid]
            for nid in range(nproposal):
                # start = time()
                unique_sampled_point_indices, sampled_xyzs = QueryFromAnchors_per_proposal(
                    input_point_cloud[bid], anchors[bid][nid], dist[bid][nid], return_xyz=True, output_points_num = 500)
                unique_sampled_point_indices = unique_sampled_point_indices.detach().cpu().numpy()
                sampled_xyzs = sampled_xyzs.cpu().numpy()
                original_pc_inds = point_clouds_sampled_ids[bid][unique_sampled_point_indices]
                anchor_sampled_pts_list.append(sampled_xyzs.astype(np.float32))
                original_pc_inds_list.append(original_pc_inds)
                if not return_images:
                    continue

                # print('sampling time: %s'%(time()-start))
                # start = time()
                if len(unique_sampled_point_indices)<self.min_num_pts:
                    valid_mask[(bid-1)*nproposal + nid] = 0
                    continue

                '''
                Search for the best image for each proposal according to the number of valid projection points
                To accelerate the training process, we narrow the search space of best image ids by our pre-computed "instance2imgid" data, so that we can get image ids by point labels
                In the testing stage, we feedforward by the generate function and the "instance2imgid" data is not used'
                '''
                sampled_pts_inst_labels = point_instance_labels[bid][unique_sampled_point_indices]
                unique_sapmled_pts_instance_labels = np.unique(sampled_pts_inst_labels)
                instance_pts_numbers = [len(np.where(sampled_pts_inst_labels == instance_ind)[0]) for instance_ind in unique_sapmled_pts_instance_labels ]
                best_intance_id = unique_sapmled_pts_instance_labels[np.argmax(np.array(instance_pts_numbers))]
                # max_instance_pts_num = instance_pts_numbers[best_intance_id]
                if best_intance_id not in instance2imageid[bid]:
                    valid_mask[(bid-1)*nproposal + nid] = 0
                    #output_images.append(cropped_image_default)
                    continue

                best_num_pts_in_image = 0
                best_img_id = -1
                MIN_NUM_PTS_THRESH = 200 if augment else 400  #Filter out proposals with not enough projected points
                ''' Add randomness by random permutation'''
                range_ids = np.random.permutation(np.arange(5)) if augment else np.arange(5)
                for iii in range_ids:
                    img_id = instance2imageid[bid][best_intance_id][iii].astype(int)
                    ptids_image = ptids_images[bid][img_id]  # [max_count_id]
                    ptids_image_flattened = ptids_image.flatten()
                    ids_pc_mask = np.in1d(original_pc_inds, ptids_image_flattened)  # pc_inds允许重复
                    num_pts_in_image = ids_pc_mask.sum()
                    if num_pts_in_image > best_num_pts_in_image:
                        best_num_pts_in_image = num_pts_in_image
                        best_img_id = img_id
                        best_ptids_image = ptids_image
                        # best_ids_pc_mask = ids_pc_mask
                    if num_pts_in_image > MIN_NUM_PTS_THRESH:
                        break
                if best_num_pts_in_image < self.min_num_pts:
                    valid_mask[(bid-1)*nproposal + nid] = 0
                    #output_images.append(cropped_image_default)
                    continue
                '''Output point ranges for image cropping'''
                ptids_image = best_ptids_image
                img_id = best_img_id
                ptids_image_flattened = ptids_image.flatten()
                on_image_ids = np.where(np.in1d(ptids_image_flattened, original_pc_inds))[
                    0]  # np.where(ptids_image == original_pc_inds)
                # on_image_ids  = ptids_image_flattened[ids]
                ids_image_x = on_image_ids % ptids_image.shape[1]
                ids_image_y = np.floor(on_image_ids /ptids_image.shape[1])
                output_pts2d = np.concatenate([ids_image_x, ids_image_y]).astype(np.int_).reshape(2,-1).transpose()  # N,2
                img_path = os.path.join(posed_images_path, scan_name,
                                        str(img_id).zfill(5) + '.jpg')  # glob(obj_%s*
                output_image_paths.append(img_path)
                mins = np.min(output_pts2d, 0).astype(np.float32)
                maxs = np.max(output_pts2d, 0).astype(np.float32)
                output_pts2d_range_list.append(np.concatenate([mins,maxs]).astype(np.float32))

        anchor_sampled_pts_list = np.array(anchor_sampled_pts_list)
        # original_pc_inds_list = np.array(original_pc_inds_list)
        if return_images:
            output_pts2d_range_list = np.array(output_pts2d_range_list).astype(np.float32)
            return anchor_sampled_pts_list, output_image_paths, output_pts2d_range_list, valid_mask, original_pc_inds_list

        return anchor_sampled_pts_list,  original_pc_inds_list

    def generate(self, data, anchors):
        ''' Only used when use_rgb = True'''
        input_point_cloud = data['point_clouds'][:, :, :3]
        point_clouds_sampled_ids = data['choices'].detach().cpu().numpy()
        ptids_images = data['ptids_images']
        batch_size, nproposal, nanchor, _ = anchors.shape
        '''For each proposal respectively, using the median distance between the anchors as the search radius of anchor-based sampling'''
        dist, _ = nn_distance_self(anchors.reshape(-1, nanchor, 3), square_root=True)  # BXK, 18
        dist, _ = dist.median(dim=-1)  # BXK
        dist = dist.reshape(batch_size, nproposal)
        valid_mask = torch.ones(batch_size * nproposal)
        output_image_paths = []
        output_pts2d_range_list = []
        anchor_sampled_pts_list = []
        original_pc_inds_list = []
        for bid in range(batch_size):
            '''load data'''
            scan_name = data['scan_name'][bid]
            for nid in range(nproposal):
                '''Search for the best image for each proposal according to the number of valid projection points'''
                # start = time()
                unique_sampled_point_indices, sampled_xyzs = QueryFromAnchors_per_proposal(
                    input_point_cloud[bid], anchors[bid][nid], dist[bid][nid], return_xyz=True, output_points_num=500)
                unique_sampled_point_indices = unique_sampled_point_indices.detach().cpu().numpy()
                sampled_xyzs = sampled_xyzs.cpu().numpy()
                # unique_sampled_point_indices = np.unique(sampled_point_indices[bid][nid])
                # print('sampling time: %s'%(time()-start))
                # start = time()
                anchor_sampled_pts_list.append(sampled_xyzs.astype(np.float32))
                original_pc_inds = point_clouds_sampled_ids[bid][unique_sampled_point_indices]
                original_pc_inds_list.append(original_pc_inds)

                if len(unique_sampled_point_indices) < self.min_num_pts:
                    valid_mask[(bid - 1) * nproposal + nid] = 0
                    # output_images.append(cropped_image_default)
                    continue
                '''Find the best corresponding images according to instance id and instance2imageid'''
                best_num_pts_in_image = 0
                best_img_id = -1

                for img_id, ptids_image in enumerate(ptids_images[bid]):  # [max_count_id]
                    ptids_image_flattened = ptids_image.flatten()
                    ptids_image_flattened = ptids_image_flattened[ptids_image_flattened > -1]
                    ids_pc_mask = np.in1d(original_pc_inds, ptids_image_flattened)  # pc_inds允许重复
                    num_pts_in_image = ids_pc_mask.sum()
                    if num_pts_in_image > best_num_pts_in_image:
                        best_num_pts_in_image = num_pts_in_image
                        best_img_id = img_id
                        best_ptids_image = ptids_image

                if best_num_pts_in_image <  self.min_num_pts:
                    valid_mask[(bid - 1) * nproposal + nid] = 0
                    continue

                '''cropped image'''
                ptids_image = best_ptids_image
                img_id = best_img_id
                ptids_image_flattened = ptids_image.flatten()
                on_image_ids = np.where(np.in1d(ptids_image_flattened, original_pc_inds))[
                    0]
                ids_image_x = on_image_ids % ptids_image.shape[1]
                ids_image_y = np.floor(on_image_ids / ptids_image.shape[1])
                output_pts2d = np.concatenate([ids_image_x, ids_image_y]).reshape(2, -1).transpose() .astype(np.int_) # N,2
                img_path = os.path.join(posed_images_path, scan_name,
                                        str(img_id).zfill(5) + '.jpg')  # glob(obj_%s*
                output_image_paths.append(img_path)
                mins = np.min(output_pts2d, 0).astype(np.float32)
                maxs = np.max(output_pts2d, 0).astype(np.float32)
                output_pts2d_range_list.append(np.concatenate([mins, maxs]).astype(np.float32))

        output_pts2d_range_list = np.array(output_pts2d_range_list).astype(np.float32)
        anchor_sampled_pts_list = np.array(anchor_sampled_pts_list)
        return anchor_sampled_pts_list, output_image_paths, output_pts2d_range_list, valid_mask, original_pc_inds_list

