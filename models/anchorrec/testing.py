# Tester for Total3D
# author: ynie
# date: April, 2020
import pickle

from models.testing import BaseTester
from .training import Trainer
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
import os
import torch
import numpy as np
from net_utils.libs import softmax
from utils import pc_util
from models.loss import compute_objectness_loss,compute_quad_score_loss
from utils.read_and_write import read_json
from net_utils.libs import flip_axis_to_depth, flip_axis_to_camera
from net_utils.box_util import get_vis_box
from net_utils.quad_ap_helper import get_quad_bbox_for_visualize,get_gt_quad_bbox_for_visualize
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
import trimesh

class Tester(BaseTester, Trainer):
    '''
    Tester object for ISCNet.
    '''

    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        if est_data[4] is not None:
            eval_dict = est_data[4]
        else:
            eval_dict, parsed_predictions = parse_predictions(est_data[0], gt_data, self.cfg.eval_config)
            eval_dict = assembly_pred_map_cls(eval_dict, parsed_predictions, self.cfg.eval_config)

        parsed_gts = parse_groundtruths(gt_data, self.cfg.eval_config)
        batch_gt_map_cls = assembly_gt_map_cls(parsed_gts)
        eval_dict['batch_gt_map_cls'] = batch_gt_map_cls
        return eval_dict

    def evaluate_step(self, est_data, data):
        eval_metrics = {}

        cls_iou_stat = est_data[6]
        if cls_iou_stat is not None:
            cls_iou_stat_out = {}
            for cls, iou in zip(cls_iou_stat['cls'], cls_iou_stat['iou']):
                if str(cls) + '_voxel_iou' not in cls_iou_stat_out:
                    cls_iou_stat_out[str(cls) + '_voxel_iou'] = []
                cls_iou_stat_out[str(cls) + '_voxel_iou'].append(iou)

            eval_metrics = {**eval_metrics, **cls_iou_stat_out}

        return eval_metrics

    def test_step(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
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
        # loss['total'] = loss['total'].item()

        # loss = {**output_loss, **eval_metrics}
        return output_loss, est_data
    def visualize_step(self, gt_data, est_data, dump_thresh = 0.5):
        bid = 0
        parsed_predictions = est_data['parsed_predictions']
        pred_mask = parsed_predictions['pred_mask']
        # above_floor_mask = parsed_predictions['above_floor_mask']
        # eval_dict = est_data['eval_dict']
        scan_name = gt_data['scan_name'][bid]
        pred_sem_cls = parsed_predictions['pred_sem_cls'] #.cpu().numpy()
        sem_cls_probs = np.array(
            [parsed_predictions['sem_cls_probs'][bid][id][pred_cls] for id, pred_cls in enumerate(pred_sem_cls[bid])])
        obj_probs = parsed_predictions['obj_prob']
        vis_box_params = np.hstack([parsed_predictions['pred_centers'][bid],
                                    parsed_predictions['pred_sizes'][bid],
                                    parsed_predictions['pred_headings'][bid, :, np.newaxis],
                                    pred_sem_cls[bid, :,np.newaxis],
                                    sem_cls_probs[:,np.newaxis],
                                    obj_probs[bid, :,np.newaxis],
                                    pred_mask[bid, :,np.newaxis]])
        vis_path = self.cfg.config['log']['vis_path']
        np.save(os.path.join(vis_path, '%s_vis_bbox.npy' % scan_name), vis_box_params)
        # mask = pred_mask[bid] * (obj_probs[bid] > dump_thresh)
        # filtered_bboxes = vis_box_params[:,:7][mask.astype(np.bool)]
        # pc_util.write_oriented_bbox( filtered_bboxes , os.path.join(vis_path, '%s.ply' % scan_name))

    def visualize_step_ori(self, phase, iter, gt_data, our_data, eval_dict, inference_switch=False):
        ''' Performs a visualization step.
        '''
        split_file = os.path.join(self.cfg.config['data']['split'], 'scannetv2_' + phase + '.json')
        scene_name = read_json(split_file)[gt_data['scan_idx']]['scan'].split('/')[3]
        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s'%(phase, iter, scene_name))
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        output_file = dict()
        # INPUT
        batch_id = 0
        point_clouds = gt_data['point_clouds'].cpu().numpy()
        pc = point_clouds[batch_id, :, :]
        # pc_util.write_ply(pc, os.path.join(dump_dir, '%01d_pc.ply' % (batch_id)))
        output_file['pc'] = pc

        
        eval_dict = our_data['eval_dict']
        parsed_predictions= our_data['parsed_predictions']
        DUMP_CONF_THRESH = self.cfg.config['generation']['dump_threshold']  # Dump boxes with obj prob larger than that.
        '''Predict boxes'''
        pred_corners_3d_upright_camera =parsed_predictions['pred_corners_3d_upright_camera']
        objectness_prob =parsed_predictions['obj_prob'][batch_id]
        pred_sem_cls =parsed_predictions['pred_sem_cls'][batch_id]
        # NETWORK OUTPUTS
        # seed_xyz = est_data['seed_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
        if 'vote_xyz' in our_data:
            # aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().cpu().numpy()
            vote_xyz = our_data['vote_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
            pc_util.write_ply(vote_xyz[batch_id], 'vote_xyz.ply')
            # vote_mask = our_data['vote_masks'].bool().detach().cpu().numpy()
            # pc_util.write_ply(vote_xyz[batch_id][vote_mask[batch_id]],'supervised_vote_xyz.ply')

            # aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().cpu().numpy()
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
        pred_mask = eval_dict['pred_mask'][batch_id, :]  # B,num_proposal
        output_file['pred_bbox_params'] = box_params
        output_file['pred_nms_mask'] = pred_mask
        output_file['pred_sem_cls'] = pred_sem_cls
        output_file['objectness_prob'] = objectness_prob

        # save_path = os.path.join(dump_dir, '%01d_pred_confident_nms_bbox.npz' % (batch_id))
        # np.savez(save_path,
        #          obbs=box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :])
        #          # proposal_map=BATCH_PROPOSAL_IDs)
        # pc_util.write_oriented_bbox(
        #                 box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
        #                 os.path.join(dump_dir, '%01d_pred_confident_nms_bbox.ply' % (batch_id)))
        
        if 'anchors' in our_data:
            anchors = our_data['anchors'][-1][batch_id].cpu().numpy()
            output_file['anchors'] = anchors
            # pc_util.write_ply(anchors[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1),:,:].reshape(-1,3), os.path.join(dump_dir, 'anchor.ply'))
            # np.save(os.path.join(dump_dir, 'anchors'),anchors)

        '''Predict meshes'''
        if 'meshes' in our_data:
            meshes = our_data['meshes' ]
            object_points = our_data['object_surface_points']
            pred_sem_cls = parsed_predictions['pred_sem_cls'][batch_id].cpu().numpy()
            BATCH_PROPOSAL_IDs = our_data['BATCH_PROPOSAL_IDs'][0].cpu().numpy()
            # np.save(os.path.join(dump_dir, 'batch_proposal_ids'),BATCH_PROPOSAL_IDs)
            index = 0
            for mesh_data, map_data in zip(meshes, BATCH_PROPOSAL_IDs):
                str_nums = (map_data[0], map_data[1], pred_sem_cls[map_data[0]])
                object_mesh = os.path.join(dump_dir, 'proposal_%d_target_%d_class_%d_mesh.obj' % str_nums)
                mesh_data.export(object_mesh)
                pc_util.write_ply(object_points[index],os.path.join(dump_dir, 'proposal_%d_objpts.ply'% map_data[0]))
                pc_util.write_ply(our_data[0]['grouped_object_points'] [index],os.path.join(dump_dir, 'proposal_%d_original_pts.ply'% map_data[0]))
                pc_util.write_ply(our_data[0]['filtered_anchors'] [index],os.path.join(dump_dir, 'proposal_%d_anchors.ply'% map_data[0]))
                index += 1

        # pc_util.write_ply(seed_xyz[batch_id, :, :], os.path.join(dump_dir, '%01d_seed_pc.ply' % (batch_id)))
        # if 'vote_xyz' in est_data:
        #     pc_util.write_ply(est_data['vote_xyz'][batch_id, :, :],
        #                       os.path.join(dump_dir, '%01d_vgen_pc.ply' % (batch_id)))
        #     pc_util.write_ply(aggregated_vote_xyz[batch_id, :, :],
        #                       os.path.join(dump_dir, '%01d_aggregated_vote_pc.ply' % (batch_id)))
        # pc_util.write_ply(box_params[:, 0:3], os.path.join(dump_dir, '%01d_proposal_pc.ply' % (batch_id)))
        # if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
        #     pc_util.write_ply(box_params[objectness_prob > DUMP_CONF_THRESH, 0:3],
        #                       os.path.join(dump_dir, '%01d_confident_proposal_pc.ply' % (batch_id)))

        # Dump predicted bounding boxes
        # if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
        #     num_proposal = box_params.shape[0]
        #     if len(box_params) > 0:
        #         pc_util.write_oriented_bbox(box_params[objectness_prob > DUMP_CONF_THRESH, :],
        #                                     os.path.join(dump_dir, '%01d_pred_confident_bbox.ply' % (batch_id)))
        #         pc_util.write_oriented_bbox(
        #             box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
        #             os.path.join(dump_dir, '%01d_pred_confident_nms_bbox.ply' % (batch_id)))
        #         pc_util.write_oriented_bbox(box_params[pred_mask[batch_id, :] == 1, :],
        #                                     os.path.join(dump_dir, '%01d_pred_nms_bbox.ply' % (batch_id)))
        #         pc_util.write_oriented_bbox(box_params, os.path.join(dump_dir, '%01d_pred_bbox.ply' % (batch_id)))
        #
        #         save_path = os.path.join(dump_dir, '%01d_pred_confident_nms_bbox.npz' % (batch_id))
        #         np.savez(save_path, obbs=box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
        #                  proposal_map = BATCH_PROPOSAL_IDs)

        # Return if it is at inference time. No dumping of groundtruths
        if inference_switch:
            return

        # objectness_loss, objectness_label,  _ , _, _  = \
        #     compute_objectness_loss(our_data, gt_data, self.cfg.dataset_config)         #objectness_mask,

        # LABELS
        gt_center = gt_data['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_mask = gt_data['box_label_mask'].cpu().numpy()  # B,K2
        gt_heading_class = gt_data['heading_class_label'].cpu().numpy()  # B,K2
        gt_heading_residual = gt_data['heading_residual_label'].cpu().numpy()  # B,K2
        gt_size_class = gt_data['size_class_label'].cpu().numpy()  # B,K2
        gt_size_residual = gt_data['size_residual_label'].cpu().numpy()  # B,K2,3
        # objectness_label = objectness_label.detach().cpu().numpy()  # (B,K,)
        # objectness_mask = objectness_mask.detach().cpu().numpy()  # (B,K,)

        # if np.sum(objectness_label[batch_id, :]) > 0:
        #     pc_util.write_ply(box_params[objectness_label[batch_id, :] > 0, 0:3],
        #                       os.path.join(dump_dir, '%01d_gt_positive_proposal_pc.ply' % (batch_id)))
        # if np.sum(objectness_mask[batch_id, :]) > 0:
        #     pc_util.write_ply(box_params[objectness_mask[batch_id, :] > 0, 0:3],
        #                       os.path.join(dump_dir, '%01d_gt_mask_proposal_pc.ply' % (batch_id)))
        # pc_util.write_ply(gt_center[batch_id, :, 0:3], os.path.join(dump_dir, '%01d_gt_centroid_pc.ply' % (batch_id)))
        # pc_util.write_ply_color(box_params[:, 0:3], objectness_label[batch_id, :],
        #                         os.path.join(dump_dir, '%01d_proposal_pc_objectness_label.ply' % (batch_id)))

        # Dump GT bounding boxes
        obbs = []
        gt_classes = []
        for j in range(gt_center.shape[1]):
            if gt_mask[batch_id, j] == 0: continue
            obb = self.cfg.dataset_config.param2obb(gt_center[batch_id, j, 0:3], gt_heading_class[batch_id, j], gt_heading_residual[batch_id, j],
                                   gt_size_class[batch_id, j], gt_size_residual[batch_id, j])
            gt_classes.append(gt_data['sem_cls_label'][batch_id, j].item())
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            output_file['gt_bboxes'] = obbs
            output_file['gt_classes'] = np.array(gt_classes)
            # pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%01d_gt_bbox.ply' % (batch_id)))

        with open(os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s.pkl'%(phase, iter, scene_name)),'wb') as pf:
            pickle.dump(output_file, pf)

        '''gt meshes'''
        if 'meshes' in our_data:
            shapenet_catids = gt_data['shapenet_catids'][batch_id]
            shapenet_ids = gt_data['shapenet_ids'][batch_id]
            for i in range(len(shapenet_catids)):
                mesh = trimesh.load(os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path,
                                                 shapenet_catids[i], shapenet_ids[i] + '.off'), process=False)
                mesh.export(os.path.join(dump_dir, 'target_%d_mesh.obj' % i))

        # OPTIONALL, also dump prediction and gt details
        # if 'batch_pred_map_cls' in eval_dict:
        #     fout = open(os.path.join(dump_dir, '%01d_pred_map_cls.txt' % (batch_id)), 'w')
        #     for t in eval_dict['batch_pred_map_cls'][batch_id]:
        #         fout.write(str(t[0]) + ' ')
        #         fout.write(",".join([str(x) for x in list(t[1].flatten())]))
        #         fout.write(' ' + str(t[2]))
        #         fout.write('\n')
        #     fout.close()
        # if 'batch_gt_map_cls' in eval_dict:
        #     fout = open(os.path.join(dump_dir, '%01d_gt_map_cls.txt' % (batch_id)), 'w')
        #     for t in eval_dict['batch_gt_map_cls'][batch_id]:
        #         fout.write(str(t[0]) + ' ')
        #         fout.write(",".join([str(x) for x in list(t[1].flatten())]))
        #         fout.write('\n')
        #     fout.close()

    def visualize_mesh(self, phase, iter, gt_data, our_data, eval_dict, inference_switch=False):
        split_file = os.path.join(self.cfg.config['data']['split'], 'scannetv2_' + phase + '.json')
        scene_name = read_json(split_file)[gt_data['scan_idx']]['scan'].split('/')[3]

        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s'%(phase, iter, scene_name))
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        batch_id = 0

        '''Predict meshes'''
        pred_sem_cls = parsed_predictions['pred_sem_cls'][batch_id].cpu().numpy()

        if our_data[5] is not None:
            meshes = our_data[5]
            BATCH_PROPOSAL_IDs = our_data[3][0].cpu().numpy()
            for mesh_data, map_data in zip(meshes, BATCH_PROPOSAL_IDs):
                str_nums = (map_data[0], map_data[1], pred_sem_cls[map_data[0]])
                object_mesh = os.path.join(dump_dir, 'proposal_%d_target_%d_class_%d_mesh.obj' % str_nums)
                mesh_data.export(object_mesh)

        '''gt meshes'''
        shapenet_catids = gt_data['shapenet_catids'][batch_id]
        shapenet_ids= gt_data['shapenet_ids'][batch_id]
        for i in range(len(shapenet_catids)):
            mesh = trimesh.load(os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path,
                                             shapenet_catids[i],shapenet_ids[i] + '.off'),process=False)
            mesh.export(os.path.join(dump_dir, 'target_%d_mesh.obj' % i))

    def visualize_quad(self, phase, iter, gt_data, our_data):
        DUMP_CONF_THRESH = self.cfg.config['log']['quad_dump_threshold']
        parsed_predictions = our_data['parsed_predictions']
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
        obbs = our_data['eval_dict']['gt_quad_box_params'][batchid]
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, 'gt_quad.ply'))

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, 'pc.ply'))
        # quad_base_xyz = our_data['aggregated_vote_quad_xyz'][batchid]

        # Dump predicted bounding boxes
        if self.cfg.config['use_iou_loss']:
            iou_threshold = self.cfg.config['model']['quad_detection']['iou_threshold']
            quad_scores_loss, quad_label, quad_assignment, quad_accuracy, quad_recall = \
                compute_quad_score_iouloss(our_data, gt_data, self.cfg.dataset_config, iou_threshold)
        else:
            quad_scores_loss, quad_label, quad_mask, quad_assignment, quad_accuracy, quad_recall \
                = compute_quad_score_loss(our_data, gt_data)

        indd = torch.nonzero(quad_label[batchid]).squeeze(1)

        pc_util.write_ply(our_data['quad_surface_points'][-1][batchid][indd].detach().cpu().numpy().reshape(-1, 3),
                          os.path.join(dump_dir, 'quad_anchors.ply'))
        pc_util.write_ply(our_data['vote_quad_xyz'][batchid, :, :],
                          os.path.join(dump_dir, 'quad_votes.ply'))
        pc_util.write_ply(
            gt_data['quad_surface_points'][batchid][quad_assignment[batchid][indd]].detach().cpu().numpy().reshape(-1,
                                                                                                                   3),
            os.path.join(dump_dir, 'quad_gt_anchors.ply'))  # surface_points[object_assignment[batchid][indd]]

        # pc_util.write_ply_color(quad_base_xyz, quad_label[batchid].cpu().numpy(),
        #                         os.path.join(dump_dir, 'gt_positive_quad_base_xyz.ply'))

        quad_prob = parsed_predictions['pred_quad_obj_prob'][batchid]
        pred_mask = parsed_predictions['pred_quad_mask']
        if np.sum(quad_prob > DUMP_CONF_THRESH) > 0:
            obbs = parsed_predictions['pred_oriented_quad_bboxes']
            # pc_util.write_oriented_bbox(obbs[batchid, quad_prob > DUMP_CONF_THRESH, :],
            #                             os.path.join(dump_dir, 'pred_confident_quad.ply'))
            # objectness_pred_label = np.zeros(quad_prob.shape)
            # objectness_pred_label[quad_prob > DUMP_CONF_THRESH] = 1
            # pc_util.write_ply_color(quad_base_xyz[batchid], objectness_pred_label,
            #                         os.path.join(dump_dir, 'pred_confident_quad_base_xyz.ply'))
            if len(obbs[batchid, np.logical_and(quad_prob > DUMP_CONF_THRESH, pred_mask[batchid, :] == 1), :]) > 0:
                pc_util.write_oriented_bbox(
                    obbs[batchid, np.logical_and(quad_prob > DUMP_CONF_THRESH, pred_mask[batchid, :] == 1), :],
                    os.path.join(dump_dir, 'pred_confident_nms_quad.ply'))


    def save_results_for_visualization(self, iter, gt_data, est_data, loss):
        scene_name = gt_data['scan_name'][0]
        end_points = est_data[0]
        # eval_dict = est_data[4]
        parsed_predictions = est_data[7]
        # INPUT
        point_clouds = gt_data['point_clouds'][0].cpu().numpy()

        '''quad relevant'''
        if gt_data['use_quad'][0] and self.cfg.config['test']['test_phase'] =='quad_detection':
            quad_obbs, processed_obbs, floors, ceilings, max_height = get_quad_bbox_for_visualize(end_points, parsed_predictions, 0)
            gt_quad_obbs, gt_floors, gt_ceilings = get_gt_quad_bbox_for_visualize(gt_data, 0)
            np.savez(os.path.join(self.cfg.save_path, 'test%s_%s_quad.npz' % (iter, scene_name)),
                     floors=floors,
                     wall_height=max_height,
                     quad_obbs=quad_obbs,
                     gt_floors=gt_floors,
                     gt_quad_obbs=gt_quad_obbs,
                     gt_ceilings=gt_ceilings,
                     quad_anchors=end_points['quad_surface_points'][-1][0].cpu().numpy(),
                     #obj_loss=loss['quad_total']
                     )


        '''anchor and vote relevant'''
        votes = end_points['vote_xyz'][0].cpu().numpy()
        sp =  end_points['anchors'][-1][0].cpu().numpy()
        seed_points = end_points['seed_xyz'][0].cpu().numpy()#N,3
        #vote_clusters = end_points['vote_cluser_xyz'][0].cpu().numpy()
        numgt = int(torch.sum(gt_data['box_label_mask']).item())
        gt_spoints = gt_data['surface_points'][0][:numgt, ...].cpu().numpy() #Np, 1024,3
        #vote_cluster_inds = end_points['vote_cluster_inds'][0].cpu().numpy() #Np,K
        #  np.take(seed_xyz,idx.astype(int),axis=0).shape

        '''mask relevant'''
        pred_mask = eval_dict['pred_mask'][0]
        above_floor_mask = parsed_predictions['above_floor_mask'][0]
        objectness_label = end_points['objectness_label'][0].cpu().numpy()
        objectness_prob = est_data[7]['obj_prob'][0]


        '''bbox relevant'''
        #GT
        gt_center = gt_data['center_label'][0].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_mask = gt_data['box_label_mask'][0].cpu().numpy()  # B,K2
        gt_heading_class = gt_data['heading_class_label'][0].cpu().numpy()  # B,K2
        gt_heading_residual = gt_data['heading_residual_label'][0].cpu().numpy()  # B,K2
        gt_size_class = gt_data['size_class_label'][0].cpu().numpy()  # B,K2
        gt_size_residual = gt_data['size_residual_label'][0].cpu().numpy()  # B,K2,3
        gt_class = gt_data['sem_cls_label'][0].cpu().numpy()  # B,K2,3

        gt_obbs = []
        for j in range(gt_center.shape[0]):
            if gt_mask[ j] == 0: continue
            obb = self.cfg.dataset_config.param2obb(gt_center[j, 0:3], gt_heading_class[j], gt_heading_residual[j],
                                   gt_size_class[j], gt_size_residual[j])
            obb = np.hstack([obb,np.expand_dims(gt_class[j],0)])
            gt_obbs.append(obb)
        if len(gt_obbs ) > 0:
            gt_obbs = np.vstack(tuple(gt_obbs))  # (num_gt_objects, 7)


        # PREDICT
        pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
        pred_sem_cls = parsed_predictions['pred_sem_cls'].detach().cpu().numpy()
        box_corners_cam = pred_corners_3d_upright_camera[0]
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
        box_params = np.hstack([centroid, sizes, orientation[:, np.newaxis],pred_sem_cls.transpose()])


        np.savez(os.path.join(self.cfg.save_path,'test%s_%s.npz'%(iter,scene_name)),
                 pc = point_clouds,
                 votes = votes,
                 anchors = sp,
                 seeds = seed_points,
                 gt_surface_points = gt_spoints,
                 #vote_clusters = vote_clusters,
                 #vote_cluster_inds = vote_cluster_inds,
                 pred_mask = pred_mask,
                 above_floor_mask = above_floor_mask,
                 gt_obbs = gt_obbs,
                 output_obbs = box_params,
                 obj_label=objectness_label,
                 objectness_prob = objectness_prob,
                 obj_loss = loss['object_total']
                 )

    def save_results(self, iter, data, est_data, dir, save_gt=True):
        bid = 0
        scene_name = data['scan_name'][bid]
        dump_dir = os.path.join(dir, scene_name) #
        # dump_dir = os.path.join(dir, '%s'%(scene_name)) #test%s_
        # dump_dir = os.path.join(cfg.config['log']['vis_path'], str(scene_name))
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        '''输出条件全部依赖于proposal_ids'''
        # BATCH_PROPOSAL_IDs = est_data['BATCH_PROPOSAL_IDs'][bid].detach().cpu().numpy()
        sample_ids = est_data['sampled_ids']
        parsed_predictions = est_data['parsed_predictions']
        pred_sem_cls = parsed_predictions['pred_sem_cls'][bid]
        np.save(os.path.join(dump_dir,'proposal_ids.npy'), sample_ids)
        # anchors0 = est_data['anchors'][-2].detach().cpu().numpy()
        anchors = est_data['anchors'][-1].detach().cpu().numpy()
        # anchor_votes0 = est_data['anchor_vote_xyz'][-2].detach().cpu().numpy()
        # anchor_votes = est_data['anchor_vote_xyz'][-1].detach().cpu().numpy()
        pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera'][bid,sample_ids]
        pred_vis_box = get_vis_box(pred_corners_3d_upright_camera)
        np.save(os.path.join(dump_dir, 'bbox.npy'),pred_corners_3d_upright_camera)
        np.save(os.path.join(dump_dir, 'vis_bbox.npy'), pred_vis_box)
        # np.save(os.path.join(dump_dir,'above_floor_mask.npy'), parsed_predictions['above_floor_mask'][bid, sample_ids])
        np.save(os.path.join(dump_dir,'sem_cls_probs.npy'), parsed_predictions['sem_cls_probs'][bid, sample_ids])
        np.save(os.path.join(dump_dir,'obj_probs.npy'), parsed_predictions['obj_prob'][bid, sample_ids])
        np.save(os.path.join(dump_dir,'anchors.npy'), anchors[bid,sample_ids])
        # np.save(os.path.join(dump_dir,'anchors0.npy'), anchors0[bid,sample_ids])
        # np.save(os.path.join(dump_dir,'anchor_votes.npy'), anchor_votes[bid,sample_ids])
        # np.save(os.path.join(dump_dir,'anchor_votes0.npy'), anchor_votes0[bid,sample_ids])
        # np.save(os.path.join(dump_dir,'voxel_size.npy'), est_data['voxel_size'])
        np.save(os.path.join(dump_dir,'pc.npy'), data['point_clouds'][bid].detach().cpu().numpy())

        if save_gt:
            parsed_gts = est_data['parsed_gts']
            num_gts = data['num_gt_objects'][bid][0]
            sem_cls_labels = parsed_gts['sem_cls_label'][bid, :num_gts]
            gt_corners_3d_upright_camera = parsed_gts['gt_corners_3d_upright_camera'][bid, :num_gts]
            gt_vis_box = get_vis_box(gt_corners_3d_upright_camera)
            np.save(os.path.join(dump_dir, 'gt_bbox.npy'), gt_corners_3d_upright_camera)
            np.save(os.path.join(dump_dir, 'gt_vis_bbox.npy'), gt_vis_box)


        '''save quads'''
        if 'quad_center' in est_data:
            quad_obbs, processed_obbs, floors, ceilings, max_height = get_quad_bbox_for_visualize(est_data, parsed_predictions, 0)
            np.savez(os.path.join(dump_dir, 'quad.npz'),
                     floors=floors,
                     wall_height=max_height,
                     quad_obbs=quad_obbs,
                     quad_anchors = est_data['quad_surface_points'][-1][0].cpu().numpy(),
                     )

        '''输出条件全部依赖于proposal_ids'''
        if 'meshes' not in est_data:
            return
        else:
            meshes = est_data['meshes']
            '''pred_meshes and anchor/object points'''
            # tmesh_path =  os.path.join(dump_dir,'trimesh') # for testing
            # if not os.path.exists(tmesh_path):
            #     os.makedirs(tmesh_path)
            for mesh_data, sample_id in zip(meshes, sample_ids):
                str_nums = (sample_id, pred_sem_cls[sample_id])
                mesh_data.export(os.path.join(dump_dir, 'proposal_%d_class_%d_mesh.ply' % str_nums))
            # for mesh_data, map_data in zip(meshes, sample_ids):
                # str_nums = (map_data[1], map_data[0], pred_sem_cls[map_data[0]])
                # mesh_data.export(os.path.join(dump_dir, 'target_%d_proposal_%d_class_%d_mesh.ply' % str_nums))
                # tmesh = mesh_data.to_trimesh()
                # tmesh.export(os.path.join(tmesh_path, 'target_%d_proposal_%d_class_%d_mesh.ply' % str_nums))

            if save_gt:
                '''gt meshes'''
                shapenet_catids = data['shapenet_catids'][bid]
                shapenet_ids = data['shapenet_ids'][bid]
                for i in range(len(shapenet_catids)):
                    mesh = trimesh.load(os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path,
                                                    shapenet_catids[i], shapenet_ids[i] + '.off'), process=False)
                    mesh.export(os.path.join(dump_dir, 'target_%d_class_%d_mesh.obj' % (i, sem_cls_labels[i])))

            # object points
            # np.save(os.path.join(dump_dir, 'normalized_object_pts.npy'),est_data['object_surface_points'].detach().cpu().numpy())
            # np.save(os.path.join(dump_dir, 'object_pts.npy'),est_data['grouped_object_points'].detach().cpu().numpy())
            # np.save(os.path.join(dump_dir, 'anchors.npy'),est_data['filtered_anchors'].detach().cpu().numpy())





    def save_quad_results(self, iter, gt_data, est_data): #, dump_dir):
        scene_name = gt_data['scan_name'][0]
        parsed_predictions = est_data['parsed_predictions']
        # INPUT
        pc = gt_data['point_clouds'][0].cpu().numpy()



        '''quad relevant'''
        quad_obbs, processed_obbs, floors, ceilings, max_height = get_quad_bbox_for_visualize(est_data, parsed_predictions, 0)
        gt_quad_obbs, gt_floors, gt_ceilings = get_gt_quad_bbox_for_visualize(gt_data, 0)

        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], 'test%s_%s' % (iter, scene_name))
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        pc_util.write_ply(pc[:,:3], os.path.join(dump_dir, 'pc.ply'))
        pc_util.write_oriented_bbox(quad_obbs, os.path.join(dump_dir, 'pred_quads.ply'))
        pc_util.write_oriented_bbox(gt_quad_obbs, os.path.join(dump_dir, 'gt_quads.ply'))
        pc_util.write_oriented_bbox(processed_obbs, os.path.join(dump_dir, 'processed_pred_quads.ply'))


        np.savez(os.path.join(self.cfg.save_path, 'test%s_%s_quad.npz' % (iter, scene_name)),
                 pc = pc,
                 floors=floors,
                 wall_height=max_height,
                 quad_obbs=quad_obbs,
                 gt_floors=gt_floors,
                 gt_quad_obbs=gt_quad_obbs,
                 gt_ceilings=gt_ceilings,
                 # gt_quad_surface_points = gt_data['quad_surface_points'][batchid][quad_assignment[batchid][indd]],
                 quad_anchors = est_data['quad_surface_points'][-1][0].cpu().numpy(),
                 # obj_loss=loss['quad_total']
                 )


    def save_bbox(self, gt_data, our_data):
        bid = 0
        # normalized_xyz, xyz, proposal_surface_points, BATCH_PROPOSAL_IDs, parsed_predictions, proposal_features, non_empty_mask
        non_empty_masks = our_data[6][bid]
        normalized_box_xyz = our_data[0][bid]
        # box_xyz = our_data[1][bid]
        shape_xyz = our_data[2][bid]
        proposal_ids = our_data[3][bid]
        # pred_corners_3d_upright_camera = our_data[4]['pred_corners_3d_upright_camera'][bid] #5
        object_input_features = our_data[5][bid] #6

        # NETWORK OUTPUTS
        normalized_box_xyz = normalized_box_xyz[non_empty_masks]
        shape_xyz = shape_xyz[non_empty_masks]
        proposal_ids = proposal_ids[non_empty_masks]
        object_input_features = object_input_features[:, non_empty_masks].t()
        assert(len(normalized_box_xyz) == len(object_input_features))

        box_num = len(proposal_ids)
        for boxid in range(box_num):
            self.input_features_paths.append(
                {'pth': os.path.join(input_scene_pth, 'object_input_features.npy'), 'boxid': boxid})
            self.input_object_points_paths.append(
                {'pth': os.path.join(input_scene_pth, 'normalized_object_points.npy'), 'boxid': boxid})
            # self.input_object_points_paths.append({'pth':os.path.join(input_scene_pth,'normalized_object_points_use_anchor.npy'), 'boxid': boxid})

            gtid = proposal_ids[boxid][1]
            # self.points_for_completion_paths.append(os.path.join(gt_scene_pth, 'obj_points_for_completion_%s.npz'%gtid))

            shapenet_catid = box_info[gtid]['shapenet_catid']
            shapenet_id = box_info[gtid]['shapenet_id']
            self.gt_mesh_paths.append(shapenet_catid + '/' + shapenet_id)
            self.proposal_idss.append(proposal_ids)
            if shapenet_catid + '/' + shapenet_id not in self.shape_count:
                self.shape_count[shapenet_catid + '/' + shapenet_id] = 1
            else:
                self.shape_count[shapenet_catid + '/' + shapenet_id] += 1


        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s' % gt_data['scan_name'][bid])
        np.save(os.path.join(dump_dir, 'proposal_ids'), proposal_ids.cpu().numpy())
        # np.save(os.path.join(dump_dir, 'proposal_ids2'), proposal_ids2.cpu().numpy())
        np.save(os.path.join(dump_dir, 'normalized_object_points'), normalized_box_xyz.detach().cpu().numpy())
        # np.save(os.path.join(dump_dir, 'normalized_object_points_use_anchor'), box_xyz_use_anchor.detach().cpu().numpy())
        np.save(os.path.join(dump_dir, 'object_input_features'),
                object_input_features.detach().cpu().numpy())
        np.save(os.path.join(dump_dir, 'shape_points'), shape_xyz.detach().cpu().numpy())

        # box_params = get_vis_box(pred_corners_3d_upright_camera[proposal_ids[:,0].cpu().numpy()])
        # pc_util.write_oriented_bbox(box_params, os.path.join(dump_dir, 'pred_confident_nms_quad.ply'))
        # pc_util.write_ply(pc[bid,:,:3].cpu().numpy(), os.path.join(dump_dir, 'pc.ply'))
        # for i in range(box_xyz.shape[0]):
        #     suffix = '_' + str(i)
        #     pc_util.write_ply(normalized_box_xyz[i].detach().cpu().numpy(), os.path.join(dump_dir, 'object_points' + suffix + '.ply'))
            # pc_util.write_ply(box_xyz_use_anchor[i].detach().cpu().numpy(), os.path.join(dump_dir, 'object_points_use_anchor' + suffix + '.ply'))
            # pc_util.write_ply(normalized_box_xyz[i].detach().cpu().numpy(), os.path.join(dump_dir, 'normalized_object_points' + suffix + '.ply'))
            # pc_util.write_ply(shape_xyz[i].detach().cpu().numpy(), os.path.join(dump_dir, 'shape_points' + suffix + '.ply'))

        # import trimesh
        # from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path
        # batch_id = 0
        # shapenet_catids = gt_data['shapenet_catids'][batch_id]
        # shapenet_ids = gt_data['shapenet_ids'][batch_id]
        # for i in range(len(shapenet_catids)):
        #     mesh = trimesh.load(os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path,
        #                                      shapenet_catids[i], shapenet_ids[i] + '.off'), process=False)
        #     mesh.export(os.path.join(dump_dir, 'target_%d_mesh.obj' % i))


