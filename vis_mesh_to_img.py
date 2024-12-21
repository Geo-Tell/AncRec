import os
import time
import numpy as np
import copy
from glob import glob
import open3d as o3d
from net_utils.ap_helper import get_pred_bbox_numpy
from net_utils.libs import flip_axis_to_depth
import pickle
import argparse

palette_inst = np.array([
    [0,245,255], #blue 0 table
    [127,255,212],#Aquamarine1 (blue-green) 1 , chair
    [255,215,0], #gold 2 , bookshelf
    [255,130,171],#paleVioletdRed2 3, sofa
    [255,187,255],#plum1 (purple) 4 , trash_bin
    [255,99,71], #tomato 5, cabinet
    [255,165,0], #orange , display
    [192,255,62], #olivedrab1 (light green) bathtub
    [255,228,181] # moccasin (light orange)
])

def get_pose(angle_x=0, angle_y=0, angle_z=0):
    # angle_x = np.random.uniform() * 2 * np.pi
    # angle_y = np.random.uniform() * 2 * np.pi
    # angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose

def vis_detection(pc, bboxes,sem_cls,output_pth):
    # with Display(size=(60, 60)) as disp:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    # pcd.point_size = o3d.utility.DoubleVector([0.5 for i in range(len(pc))])
    pcd.paint_uniform_color([0.2,0.2,0.2])
    # pc_instance = {"name": "pc", "geometry":pcd}
    vis.add_geometry(pcd)
    # mat = o3d.visualization.rendering.MaterialRecord()
    # mat.shader = "unlitLine"
    # mat.line_width = 10

    boxid = 0
    for i,bbox in enumerate(bboxes):
        # center = bbox[0:3]
        center = bbox[:3]
        orientation = bbox[6]
        sizes = bbox[3:6]
        cls_id = sem_cls[i]
        color = palette_inst[int(cls_id)].astype(float)/255.0
        quaternion = np.array([-np.sin(orientation/2).tolist(), 0.0,0.0,  np.cos(orientation/2).tolist()])
        R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_quaternion(quaternion)
        o3d_box =  o3d.geometry.OrientedBoundingBox(center, R, sizes)
        o3d_box.color = color
        boxid += 1
        # instance = {'name':str(boxid), 'geometry': o3d_box,"material": mat}
        vis.add_geometry(o3d_box)

    vis.get_render_option().line_width = 9
    vis.get_render_option().point_size = 0.1

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_pth, True)
    # time.sleep(0.00001)
    # vis.destroy_window()
    # time.sleep(0)
    # vis.clear_geometries()


def vis_mesh_to_img(pth, outpth):
    if not os.path.exists(outpth):
        os.makedirs(outpth)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    meshes = os.listdir(pth)
    # with Display(size=(60, 60)) as disp:
    for mesh_file in glob(os.path.join(pth, '*pred*.ply')):
        mesh_file_ = mesh_file.split('/')[-1]
        print(mesh_file_)
        mesh = o3d.io.read_triangle_mesh(os.path.join(pth, mesh_file_))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array([192, 255, 62]).astype(float)/255.0)
        pose = get_pose(angle_x= -np.pi/6., angle_y= 5* np.pi/6., angle_z=0)
        mesh_t = copy.deepcopy(mesh).transform(pose)
        vis.add_geometry(mesh_t)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(outpth, mesh_file_.split('.')[0]+'.jpg'), True)
    time.sleep(0.00001)
    # vis.destroy_window()
    # time.sleep(0)
    vis.clear_geometries()

    # for mesh_file in glob(os.path.join(pth, '*gt_z.ply')):
    #     mesh_file_ = mesh_file.split('/')[-1]
    #     print(mesh_file_)
    #     mesh = o3d.io.read_triangle_mesh(os.path.join(pth, mesh_file_))
    #     mesh.compute_vertex_normals()
    #     mesh.paint_uniform_color(np.array([192, 255, 62]).astype(float)/255.0)
    #     pose = get_pose(angle_x= -np.pi/6., angle_y= 5* np.pi/6., angle_z=0)
    #     mesh_t = copy.deepcopy(mesh).transform(pose)
    #     vis.add_geometry(mesh_t)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     vis.capture_screen_image(os.path.join(outpth, mesh_file_.split('.')[0]+'.jpg'), True)
    #     time.sleep(0.00001)
    #     # vis.destroy_window()
    #     # time.sleep(0)
    #     vis.clear_geometries()

def decode_bbox_params(base_xyz, geo_params, sem_params, gt_data,objectness_thresh=0.05):
    num_heading_bin = 12
    num_size_cluster = 8
    end_points = {}
    center = base_xyz + geo_params[:,:3]  # (batch_size x num_proposal, 3)
    end_points['center'] = center

    heading_scores = geo_params[:,3:3 + num_heading_bin]
    heading_residuals_normalized = geo_params[:,3 + num_heading_bin:3 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized
    # B x num_proposal x num_heading_bin (should be -1 to 1)

    size_scores = geo_params[:,3 + num_heading_bin * 2:3 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = geo_params[:,3 + num_heading_bin * 2 + num_size_cluster:
                                             3 + num_heading_bin * 2 + num_size_cluster * 4].reshape(-1, num_size_cluster, 3)  # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    end_points['objectness_scores'] = sem_params[:,:2]
    end_points['sem_cls_scores'] = sem_params[:,2:]
    

    parsed_predictions = get_pred_bbox_numpy(end_points, cfg, nms=True, gt_data=gt_data)
    pred_corners_3d_upright_camera = parsed_predictions['pred_corners']
    objectness_prob = parsed_predictions['obj_prob']
    pred_sem_cls = parsed_predictions['pred_sem_cls']
    pred_nms_mask = parsed_predictions['pred_mask']

    # aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().cpu().numpy()
    box_corners_cam = pred_corners_3d_upright_camera
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

    pred_bboxes_all = np.hstack([centroid, sizes, orientation[:, np.newaxis]])
    mask = np.logical_and(objectness_prob > objectness_thresh, pred_nms_mask == 1)
    pred_bboxes = pred_bboxes_all[mask]
    pred_classes = pred_sem_cls[mask]
    return pred_bboxes, pred_classes


'''test.pkl （prased_predictions) -> vis_bbox'''
def save_detection_vis(data, outpth, eval=True, vis=True, use_valid_mask = False):
    from net_utils.ap_helper import assembly_pred_map_cls2, assembly_gt_map_cls2
    from net_utils.ap_helper import APCalculator
    from models.anchorrec.dataloader import parse_groundtruths_from_raw
    parsed_predictions_list = data['parsed_predictions']
    valid_mask_list = data['valid_mask_list']
    scan_names = data['scan_names']
    if eval:
        from configs.scannet_config import ScannetConfig
        dataset_config = ScannetConfig()
        ap_calculator_list = [APCalculator(iou_thresh, dataset_config.class2type, False)
                              for iou_thresh in [0.5,0.25]]
    for iter, scan_name in enumerate(scan_names):
        parsed_predictions = parsed_predictions_list[iter]
        valid_mask = valid_mask_list[iter]
        if vis:
            pred_sem_cls = parsed_predictions['pred_sem_cls'].cpu().numpy()
            sem_cls_probs = np.array([parsed_predictions['sem_cls_probs'][id][pred_cls] for id, pred_cls in enumerate(pred_sem_cls)])
            obj_probs = parsed_predictions['obj_prob']
            vis_box_params = np.hstack([parsed_predictions['pred_centers'],
                                        parsed_predictions['pred_sizes'],
                                        parsed_predictions['pred_headings'][:, np.newaxis],
                                        pred_sem_cls[:, np.newaxis],
                                        sem_cls_probs[:, np.newaxis],
                                        obj_probs[:, np.newaxis],
                                        valid_mask[:, np.newaxis]])
            # img_paths =
            np.save(os.path.join(outpth,'%s_vis_bbox.npy'%scan_name), vis_box_params)
        if eval:

            parsed_gts = parse_groundtruths_from_raw(scan_name)
            batch_pred_map_cls = assembly_pred_map_cls2(parsed_predictions, conf_thresh=0.05)
            batch_gt_map_cls = assembly_gt_map_cls2(parsed_gts)
            for ap_calculator  in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    if eval:
        for i, ap_calculator in enumerate(ap_calculator_list):
            print(('-' * 10 + 'iou_thresh: %f' + '-' * 10) % (ap_calculator.ap_thresh))
            metrics_dict = ap_calculator.compute_metrics()
            for key in metrics_dict:
                print('eval %s: %f' % (key, metrics_dict[key]))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--config', type=str, default='configs/config_files/ISCNet.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo/inputs/scene0549_00.off', help='Please specify the demo path.')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', default=18888)
    return parser.parse_args()

if __name__ == '__main__':
    with open('out/rgb_pretrain/2_resnet34_with_val/refine/2/gen_data_for_recon/test.pkl', 'rb') as pf:
        data = pickle.load(pf)
    output_pth = 'out/rgb_pretrain/2_resnet34_with_val/refine/2/gen_data_for_recon/detect_results'
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)
    save_detection_vis(data, output_pth, eval=True, vis=False) #, eval=False)
    # save_detection_vis(data, output_pth) #, eval=False)

    # import argparse
    # from configs.config_utils import CONFIG
    # from configs.config_utils import mount_external_config
    # args = parse_args()
    # cfg = CONFIG(args.config)
    # cfg.update_config(args.__dict__)
    # cfg = mount_external_config(cfg)
    # with open('/home/lht/dmy/CLEAN2/out/detection_w_rgb/no_rgb/prepare_data/val2.pkl', 'rb') as pf:
    #     data = pickle.load(pf)
    # vis_prepare_data(data, cfg, output_pth='/home/lht/dmy/CLEAN2/out/detection_w_rgb/no_rgb/prepare_data/vis_imgs')


    # input_pth = 'out/CAGroupBackbone/1_0.464/test/2/visualization'
    # output_pth = 'out/CAGroupBackbone/1_0.464/test/2/vis_img'
    # if not os.path.exists(output_pth):
    #     os.makedirs(output_pth)
    # import pickle
    # dump_thresh = 0.5
    # for filename in os.listdir(input_pth):
    #     filename_ = filename.split('.')[0]
    #     scene_name = '_'.join(filename_.split('_')[-2:])
    #     with open(os.path.join(input_pth, filename), 'rb') as pf:
    #         data = pickle.load(pf)
    #     pc = data['pc']
    #     pred_bboxes_all = data['pred_bbox_params']
    #     pred_mask = data['pred_nms_mask']
    #     pred_sem_cls= data['pred_sem_cls']
    #     objectness_prob = data['objectness_prob']
    #     mask = np.logical_and(objectness_prob > dump_thresh, pred_mask == 1)
    #     pred_bboxes = pred_bboxes_all[mask]
    #     pred_classes = pred_sem_cls[mask]
    #     gt_bboxes = data['gt_bboxes']
    #     gt_classes = data['gt_classes']
    #
    #     vis_detection(pc,pred_bboxes, pred_classes,os.path.join(output_pth, scene_name+'_pred.jpg'))
    #     vis_detection(pc, gt_bboxes, gt_classes, os.path.join(output_pth, scene_name+'_gt.jpg'))



