import torch
import numpy as np
import os
import pickle
from models.training import BaseTrainer
from net_utils import visualization as vis
from utils import pc_util
from models.loss import compute_objectness_loss
from net_utils.libs import flip_axis_to_depth,  softmax
from utils.read_and_write import read_json
from configs.path_config import split_path
class Trainer(BaseTrainer):
    '''
    Trainer object for total3d.
    '''

    def eval_step(self, data):
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        # loss = self.compute_loss(data)
        '''load input and ground-truth data'''
        data = self.to_device(data)
        '''network forwarding'''
        self.net.module.eval()
        with torch.no_grad():
            est_data = self.net(data)
        loss = self.net.module.loss(est_data, data)
        output_loss = {}
        for key, value in loss.items():
            if torch.is_tensor(value):
                output_loss[key] = loss[key].item()
            elif type(value) == float:
                output_loss[key] = loss[key]
            elif type(value) == int:
                output_loss[key] = loss[key]
        return output_loss, est_data

    def eval_step_for_mAP(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        self.net.module.eval()
        est_data = self.net.module.generate(data)

        '''computer losses'''
        loss = self.net.module.loss(est_data, data)
        # eval_metrics = self.evaluate_step(est_data, data)

        output_loss = {}
        for key, value in loss.items():
            if torch.is_tensor(value):
                output_loss[key] = loss[key].item()
            elif type(value) == float:
                output_loss[key] = loss[key]
            elif type(value) == int:
                output_loss[key] = loss[key]
        # output_loss = {}
        return output_loss, est_data


    def visualize_det_step_ori(self, epoch, phase, iter, data):
        ''' Performs a visualization step.
        '''
        '''load input and ground-truth data'''
        det_dict = {}
        det_dict['scan_idx'] = data['scan_idx']
        data = self.to_device(data)

        with torch.no_grad():
            '''network forwarding'''
            est_data = self.net({**data, 'export_shape':True})
            for k in est_data[0].keys():
                if ('sa' not in k) and ('fp' not in k):
                    det_dict[k] = est_data[0][k]

            det_vis_path = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%d_det.pkl' % (epoch, phase, iter))
            with open(det_vis_path, 'wb') as f:
                pickle.dump(det_dict, f)

    def visualize_detection_step(self, phase, iter, gt_data, our_data, inference_switch=False):
        est_data = our_data[0]
        parsed_predictions = our_data[7]
        # eval_dict = our_data[2]
        # meshes = our_data[6]
        batchid = 0
        DUMP_CONF_THRESH = self.cfg.config['log']['dump_threshold']  # Dump boxes with obj prob larger than that.
        scene_name = gt_data['scan_name'][batchid]
        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s' % (phase, iter, scene_name))
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        # INPUT
        point_clouds = gt_data['point_clouds'].cpu().numpy()
        pc = point_clouds[batchid, :, :]
        seed_xyz = est_data['seed_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
        aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().cpu().numpy()

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, 'pc.ply'))
        pc_util.write_ply(seed_xyz[batchid, :, :], os.path.join(dump_dir, 'seed_pc.ply'))
        if 'vote_xyz' in est_data:
            pc_util.write_ply(est_data['vote_xyz'][batchid, :, :],
                              os.path.join(dump_dir, 'vgen_pc.ply'))
            pc_util.write_ply(aggregated_vote_xyz[batchid, :, :],
                              os.path.join(dump_dir, 'aggregated_vote_pc.ply'))
        # pc_util.write_ply(box_params[:, 0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply' % (batchid)))
        # if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
        #     pc_util.write_ply(box_params[objectness_prob > DUMP_CONF_THRESH, 0:3],
        #                       os.path.join(dump_dir, '%06d_confident_proposal_pc.ply' % (batchid)))
        # NETWORK OUTPUTS

        if self.cfg.config['use_iou_loss']:
            _, object_label, object_assignment, _, _ = \
                compute_objectness_iouloss(est_data, gt_data, self.cfg.dataset_config,
                                           self.cfg.config['model']['detection']['iou_threshold'])
        else:
            _, object_label, _, object_assignment, _, _ = \
                compute_objectness_loss(est_data, gt_data)

        indd = torch.nonzero(object_label[batchid]).squeeze(1)
        pc_util.write_ply(est_data['shape_points'][-1][batchid][indd].detach().cpu().numpy().reshape(-1, 3),
                          os.path.join(dump_dir, 'object_svote.ply'))
        pc_util.write_ply(est_data['shape_points'][-1][batchid].detach().cpu().numpy().reshape(-1, 3),
                          os.path.join(dump_dir, 'object_svote_whole.ply'))
        pc_util.write_ply(
            gt_data['surface_points'][batchid][object_assignment[batchid][indd]].detach().cpu().numpy().reshape(-1, 3),
            os.path.join(dump_dir, 'object_gt_svote.ply'))  # surface_points[object_assignment[batchid][indd]]

        '''Predict boxes'''
        pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
        objectness_prob = parsed_predictions['obj_prob'][batchid]
        pred_sem_cls = parsed_predictions['pred_sem_cls'][batchid].long().cpu().numpy()
        box_corners_cam = pred_corners_3d_upright_camera[batchid]
        box_corners_depth = flip_axis_to_depth(box_corners_cam)
        centroid = (np.max(box_corners_depth, axis=1) + np.min(box_corners_depth, axis=1)) / 2.
        forward_vector = box_corners_depth[:, 1] - box_corners_depth[:, 2]
        left_vector = box_corners_depth[:, 0] - box_corners_depth[:, 1]
        up_vector = box_corners_depth[:, 6] - box_corners_depth[:, 2]
        orientation = np.arctan2(forward_vector[:, 1], forward_vector[:, 0])
        forward_size = np.linalg.norm(forward_vector, axis=1)
        left_size = np.linalg.norm(left_vector, axis=1)
        up_size = np.linalg.norm(up_vector, axis=1)
        sizes = np.vstack([forward_size, left_size, up_size]).T
        box_params = np.hstack([centroid, sizes, orientation[:, np.newaxis], pred_sem_cls[:, np.newaxis]])

        # OTHERS
        pred_mask = parsed_predictions['pred_mask']  # B,num_proposal

        # Dump predicted bounding boxes
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            num_proposal = box_params.shape[0]
            if len(box_params) > 0:
                # pc_util.write_colored_oriented_bbox(box_params[objectness_prob > DUMP_CONF_THRESH, :],
                #                                     os.path.join(dump_dir, 'pred_confident_bbox.ply'))
                # pc_util.write_colored_oriented_bbox(box_params[pred_mask[batchid, :] == 1, :],
                #                                     os.path.join(dump_dir, 'pred_nms_bbox.ply'))
                pc_util.write_colored_oriented_bbox(
                    box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batchid, :] == 1), :],
                    os.path.join(dump_dir, 'pred_confident_nms_bbox.ply'))

        # Return if it is at inference time. No dumping of groundtruths
        if inference_switch:
            return

        # LABELS
        gt_center = gt_data['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_mask = gt_data['box_label_mask'].cpu().numpy()  # B,K2
        gt_heading_class = gt_data['heading_class_label'].cpu().numpy()  # B,K2
        gt_heading_residual = gt_data['heading_residual_label'].cpu().numpy()  # B,K2
        gt_size_class = gt_data['size_class_label'].cpu().numpy()  # B,K2
        gt_size_residual = gt_data['size_residual_label'].cpu().numpy()  # B,K2,3
        gt_sem_cls = gt_data['sem_cls_label'].cpu().numpy()  # B,K2,3
        # pc_util.write_ply(gt_center[batchid, :, 0:3], os.path.join(dump_dir, 'gt_centroid_pc.ply'))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[batchid, j] == 0: continue
            obb = self.cfg.dataset_config.param2obb(gt_center[batchid, j, 0:3], gt_heading_class[batchid, j],
                                                    gt_heading_residual[batchid, j],
                                                    gt_size_class[batchid, j], gt_size_residual[batchid, j])
            obb = np.insert(obb, len(obb), gt_sem_cls[batchid, j])
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_colored_oriented_bbox(obbs, os.path.join(dump_dir, 'gt_bbox.ply'))

    def visualize_quad(self, phase, iter, gt_data, our_data):

        DUMP_CONF_THRESH = self.cfg.config['log']['quad_dump_threshold']
        end_points = our_data[0]
        parsed_predictions = our_data[1]
        batchid = 0
        scene_name = gt_data['scan_name'][batchid]
        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s' % (phase, iter, scene_name))
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        # INPUT
        point_clouds = gt_data['point_clouds'].cpu().numpy()
        pc = point_clouds[batchid, :, :]

        # LABELS
        # Dump GT bounding boxes
        obbs = our_data[2]['gt_quad_box_params'][batchid]
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, 'gt_quad.ply'))

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, 'pc.ply'))
        quad_base_xyz = end_points['aggregated_vote_quad_xyz'][batchid]

        # Dump predicted bounding boxes
        if self.cfg.config['use_iou_loss']:
            iou_threshold = self.cfg.config['model']['quad_detection']['iou_threshold']
            quad_scores_loss, quad_label, quad_assignment, quad_accuracy, quad_recall = \
                compute_quad_score_iouloss(end_points, gt_data, self.cfg.dataset_config, iou_threshold)
        else:
            quad_scores_loss, quad_label, quad_mask, quad_assignment, quad_accuracy, quad_recall \
                = compute_quad_score_loss(end_points, gt_data)

        indd = torch.nonzero(quad_label[batchid]).squeeze(1)

        pc_util.write_ply(end_points['quad_surface_points'][-1][batchid][indd].detach().cpu().numpy().reshape(-1, 3),
                          os.path.join(dump_dir, 'quad_svote.ply'))
        pc_util.write_ply(end_points['vote_quad_xyz'][batchid, :, :],
                          os.path.join(dump_dir, 'quad_vgen.ply'))
        pc_util.write_ply(
            gt_data['quad_surface_points'][batchid][quad_assignment[batchid][indd]].detach().cpu().numpy().reshape(-1,
                                                                                                                   3),
            os.path.join(dump_dir, 'quad_gt_svote.ply'))  # surface_points[object_assignment[batchid][indd]]
        # pc_util.write_ply(gt_data['quad_surface_points'][batchid].detach().cpu().numpy().reshape(-1, 3),
        #                   os.path.join(dump_dir, 'gtvote_whole.ply'))

        pc_util.write_ply_color(quad_base_xyz, quad_label[batchid].cpu().numpy(),
                                os.path.join(dump_dir, 'gt_positive_quad_base_xyz.ply'))

        quad_prob = parsed_predictions['pred_quad_obj_prob'][batchid]
        pred_mask = parsed_predictions['pred_quad_mask']
        if np.sum(quad_prob > DUMP_CONF_THRESH) > 0:
            obbs = parsed_predictions['pred_oriented_quad_bboxes']
            # pc_util.write_oriented_bbox(obbs[batchid, quad_prob > DUMP_CONF_THRESH, :],
            #                             os.path.join(dump_dir, 'pred_confident_quad.ply'))
            objectness_pred_label = np.zeros(quad_prob.shape)
            objectness_pred_label[quad_prob > DUMP_CONF_THRESH] = 1
            # pc_util.write_ply_color(quad_base_xyz[batchid], objectness_pred_label,
            #                         os.path.join(dump_dir, 'pred_confident_quad_base_xyz.ply'))
            if len(obbs[batchid, np.logical_and(quad_prob > DUMP_CONF_THRESH, pred_mask[batchid, :] == 1), :]) > 0:
                pc_util.write_oriented_bbox(
                    obbs[batchid, np.logical_and(quad_prob > DUMP_CONF_THRESH, pred_mask[batchid, :] == 1), :],
                    os.path.join(dump_dir, 'pred_confident_nms_quad.ply'))

                # pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, 'pred_quad.ply'))

    def visualize_step(self, epoch, phase, iter, gt_data):
        ''' Performs a visualization step.
        '''
        with torch.no_grad():
            est_data = self.net.module.generate(gt_data)

        split_file = os.path.join(split_path + '/fullscan/'+ self.cfg.config['data']['split'], 'scannetv2_' + phase + '.json')
        scene_name = read_json(split_file)[gt_data['scan_idx'][0]]['scan'].split('/')[3]
        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s' % (phase, iter, scene_name))
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        # INPUT
        batch_id = 0
        point_clouds = gt_data['point_clouds'].cpu().numpy()
        pc = point_clouds[batch_id, :, :]
        pc_util.write_ply(pc, os.path.join(dump_dir, '%01d_pc.ply' % (batch_id)))
        np.save(os.path.join(dump_dir, '%01d_pc.npy' % (batch_id)),pc)

        eval_dict = est_data['eval_dict']
        parsed_predictions = est_data['parsed_predictions']
        DUMP_CONF_THRESH = self.cfg.config['log']['dump_threshold']  # Dump boxes with obj prob larger than that.
        '''Predict boxes'''
        pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
        objectness_prob = parsed_predictions['obj_prob'][batch_id]

        if 'vote_xyz' in est_data:
            vote_xyz = est_data['vote_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
            pc_util.write_ply(vote_xyz[batch_id], 'vote_xyz.ply')

        box_corners_cam = pred_corners_3d_upright_camera[batch_id]
        box_corners_depth = flip_axis_to_depth(box_corners_cam)
        centroid = (np.max(box_corners_depth, axis=1) + np.min(box_corners_depth, axis=1)) / 2.
        forward_vector = box_corners_depth[:,1] - box_corners_depth[:,2]
        left_vector = box_corners_depth[:,0] - box_corners_depth[:,1]
        up_vector = box_corners_depth[:,6] - box_corners_depth[:,2]
        orientation = np.arctan2(forward_vector[:,1], forward_vector[:,0])
        forward_size = np.linalg.norm(forward_vector, axis=1)
        left_size = np.linalg.norm(left_vector, axis=1)
        up_size = np.linalg.norm(up_vector, axis=1)
        sizes = np.vstack([forward_size, left_size, up_size]).T
        box_params = np.hstack([centroid, sizes, orientation[:,np.newaxis]])
        pred_mask = eval_dict['pred_mask']  # B,num_proposal
        # save_path = os.path.join(dump_dir, '%01d_pred_confident_nms_bbox.npz' % (batch_id))
        np.save(os.path.join(dump_dir,'bbox.npy'),box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :])
        pc_util.write_oriented_bbox(
                        box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
                        os.path.join(dump_dir, '%01d_pred_confident_nms_bbox.ply' % (batch_id)))

        if 'anchors' in est_data:
            anchors = est_data['anchors'][-1][batch_id].cpu().numpy()
            pc_util.write_ply(anchors[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1),:,:].reshape(-1,3), os.path.join(dump_dir, 'anchor.ply'))
            # np.save(os.path.join(dump_dir, 'anchors'),anchors)

        gt_center = gt_data['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_mask = gt_data['box_label_mask'].cpu().numpy()  # B,K2
        gt_heading_class = gt_data['heading_class_label'].cpu().numpy()  # B,K2
        gt_heading_residual = gt_data['heading_residual_label'].cpu().numpy()  # B,K2
        gt_size_class = gt_data['size_class_label'].cpu().numpy()  # B,K2
        gt_size_residual = gt_data['size_residual_label'].cpu().numpy()  # B,K2,3

        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[batch_id, j] == 0: continue
            obb = self.cfg.dataset_config.param2obb(gt_center[batch_id, j, 0:3], gt_heading_class[batch_id, j], gt_heading_residual[batch_id, j],
                                   gt_size_class[batch_id, j], gt_size_residual[batch_id, j])
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%01d_gt_bbox.ply' % (batch_id)))


    def to_device(self, data):
        device = self.device
        for key in data:
            if key not in ['object_voxels', 'shapenet_catids', 'shapenet_ids','scan_name','use_quad','mesh_pth','scene_boxid']\
                and  type(data[key]).__name__== 'Tensor':
                data[key] = data[key].to(device)
        return data

    def compute_loss(self, data):
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)
        # print("computeLoss:%s"%self.cfg.LOCAL_RANK )
        '''computer losses'''
        loss = self.net.module.loss(est_data, data)
        return loss
