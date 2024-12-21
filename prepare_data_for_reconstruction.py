import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
from models.anchorrec.SingleObjectDatasetWithRGB import AnchorRec_ScanNet_For_Refine, collate_fn, my_worker_init_fn
from torch.utils.data import DataLoader
from net_utils.ap_helper import APCalculator
from models.loss import compute_objectness_loss_new
def decode_scores_geo(net, base_xyz, end_points, bsize, nproposal, num_heading_bin=12, num_size_cluster=8):
    center = base_xyz + net[:, :, :3]  # (batch_size x num_proposal, 3)
    end_points['center'] = center

    heading_scores = net[:, :, 3:3 + num_heading_bin]
    heading_residuals_normalized = net[:, :, 3 + num_heading_bin:3 + num_heading_bin * 2]
    end_points['heading_scores'] = heading_scores  # Bxnum_proposalxnum_heading_bin
    end_points['heading_residuals_normalized'] = heading_residuals_normalized
    # B x num_proposal x num_heading_bin (should be -1 to 1)

    size_scores = net[:, :, 3 + num_heading_bin * 2:3 + num_heading_bin * 2 + num_size_cluster]
    size_residuals_normalized = net[:, :,
                                3 + num_heading_bin * 2 + num_size_cluster:3 + num_heading_bin * 2 + num_size_cluster * 4].view(
        bsize, nproposal, num_size_cluster, 3)  # Bxnum_proposalxnum_size_clusterx3
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized

    return end_points


'''
prepare_data_for_refine: output obtained bbox parameters by pretrained models(geo_params, geo_params, anchor points, sampled points, bbox features).
For training set, output objectness_label and the corresponding shapenet_ids,shapenet_cat_ids labels for supervision in the next refinement step
'''
def prepare_data_for_refine(cfg, tester, test_loader):
    mode = cfg.config['data'].get('split_name','test')
    #输出：List of Nproposals
    tester.net.train(False)
    from configs.scannet_config import ScannetConfig
    dataset_config = ScannetConfig()
    results = {}
    results['scan_names']=[]
    results['sem_params']=[]
    results['geo_params']=[]
    results['anchor_sampled_pts_list'] = []
    results['input_image_paths']= []
    results['input_pts2d_range_list']= []
    results['valid_mask']= []
    results['vote_features']= []
    results['anchor_features']= []
    results['anchors']= []
    results['aggregated_vote_xyz']= []
    results['original_pc_inds_list']= []
    if mode != 'test':
        results['objectness_label'] = []
        results['shapenet_ids_list'] = []
        results['shapenet_catids_list'] = []

    print(len(test_loader))
    for iter, data in tqdm(enumerate(test_loader)):
        data = tester.to_device(data)
        end_points = tester.net.module.generate_input(data, test_set =mode=='test') #, generate_for_test = True)
        nproposal, _ = end_points['geo_params'].shape
        output_image_paths = end_points['output_image_paths']
        anchor_sampled_pts_list = end_points['anchor_sampled_pts_list']
        original_pc_inds_list = end_points['original_pc_inds_list']
        output_pts2d_range_list = end_points['output_pts2d_range_list']
        valid_mask = end_points['valid_mask'].bool().cpu().numpy()
        num_pts = anchor_sampled_pts_list.shape[1]
        '''#sampled点的个数是一样的，这里是为了给没有图片的proposal占位'''
        # anchor_sampled_pts_list_ = np.zeros((nproposal, num_pts,3))
        output_pts2d_range_list_ = np.zeros((nproposal, 4))

        output_image_paths_ = []
        # original_pc_inds_list_ = []
        ii = 0
        # No image / pts2d_range recorded where valid_mask =Fals
        for i in range(nproposal):
            if valid_mask[i]:
                output_image_paths_.append(output_image_paths[ii])
                # original_pc_inds_list_.append(original_pc_inds_list[ii])
                ii += 1
            else:
                output_image_paths_.append([])
                # original_pc_inds_list_.append([])

        # anchor_sampled_pts_list_[valid_mask] = anchor_sampled_pts_list
        output_pts2d_range_list_[valid_mask] = output_pts2d_range_list
        results['scan_names'].append(data['scan_name'][0])
        results['geo_params'].append(end_points['geo_params'].cpu().numpy())
        results['sem_params'].append(end_points['sem_params'].cpu().numpy())
        results['anchor_sampled_pts_list'].append(anchor_sampled_pts_list)
        results['input_image_paths'].append(output_image_paths_ )
        results['original_pc_inds_list'].append(original_pc_inds_list)
        results['input_pts2d_range_list'].append(output_pts2d_range_list_)
        results['valid_mask'].append(valid_mask)
        results['vote_features'].append(end_points['vote_features'][0].t().detach().cpu().numpy())
        results['anchor_features'].append(end_points['anchor_features'][0].t().detach().cpu().numpy())
        results['aggregated_vote_xyz'].append(end_points['aggregated_vote_xyz'][0].detach().cpu().numpy())
        results['anchors'].append(end_points['anchors'][-1][0].detach().cpu().numpy())
        data['anchor_sampled_xyzs'] =  np.expand_dims(anchor_sampled_pts_list,0)
        if mode != 'test':
            objectness_label, object_assignment, soft_cls_labels = \
                compute_objectness_loss_new(end_points, data, dataset_config, return_loss=False)  # objectness_mask
            shapenet_catids = data['shapenet_catids'][0]
            shapenet_ids = data['shapenet_ids'][0]  # assignment: b,n
            gt_ids = list(object_assignment[0].cpu().numpy())
            objectness_label = objectness_label[0].cpu().numpy()
            shapenet_catids_ = []
            shapenet_ids_ = []
            for i, gt_id in enumerate(gt_ids):
                if objectness_label[i]:
                    shapenet_catids_.append(shapenet_catids[gt_id])
                    shapenet_ids_.append(shapenet_ids[gt_id])
                else:
                    shapenet_catids_.append(0)
                    shapenet_ids_.append(0)
            results['objectness_label'].append(objectness_label)
            results['shapenet_ids_list'].append(shapenet_catids_)
            results['shapenet_catids_list'].append(shapenet_ids_)

    split_name = cfg.config["data"].get("split_name", 'test')
    with open(os.path.join(cfg.config['log']['vis_path'], '%s.pkl'%split_name), 'wb') as f:
        pickle.dump(results, f)

def check_data_for_refine(data, output_pth):
    import cv2
    from net_utils.ap_helper import parse_predictions
    from utils.pc_util import write_ply, write_oriented_bbox
    from configs.path_config import scannet_processed_path, posed_images_path
    from configs.scannet_config import ScannetConfig
    #输出：filtered bbox, cropped images, corresponding gts
    scan_names = data['scan_names']
    geo_params_list = data['geo_params']
    sem_params_list = data['sem_params']  # / weight
    anchor_sampled_pts_list = data['anchor_sampled_pts_list']
    input_image_paths = data['input_image_paths']
    pts2d_range_list = data['input_pts2d_range_list']
    valid_masks = data['valid_mask']
    # vote_features = data['vote_features']
    # anchor_features = data['anchor_features']
    aggregated_vote_xyz = data['aggregated_vote_xyz']
    # anchors = data['anchors']
    # original_pc_inds_list = data['original_pc_inds_list']
    config_dict = {}
    config_dict['remove_empty_box'] = False #True
    config_dict['use_3d_nms'] = True
    config_dict['cls_nms'] = True
    config_dict['use_old_type_nms'] = False
    config_dict['nms_iou'] = 0.25
    config_dict['dataset_config'] = ScannetConfig()
    objectness_label = None
    if 'objectness_label' in data:
        objectness_label = data['objectness_label']
        # shapenet_ids_list = data['shapenet_ids_list']
        # shapenet_catids_list = data['shapenet_catids_list']
    # geo_params/sem_params -> bbox
    data_num = len(scan_names)
    for iter in range(data_num):
        if iter >5:
            break
        scan_name = scan_names[iter]
        scan_data = np.load(os.path.join(scannet_processed_path, scan_name, "full_scan.npz"))
        point_cloud = scan_data['mesh_vertices'][:,:3]
        pc = point_cloud[np.random.choice(len(point_cloud),5000)]
        write_ply(pc, os.path.join(output_pth,"%s.ply"%scan_name))
        geo_params = geo_params_list[iter]
        sem_params = sem_params_list[iter]
        nproposal = len(geo_params)
        end_points = decode_scores_geo(torch.tensor(geo_params).unsqueeze(0),
                                                 torch.tensor(aggregated_vote_xyz[iter]),
                                       {}, bsize=1, nproposal = nproposal,
                                                 num_heading_bin = 12,
                                                 num_size_cluster =8)
        end_points['objectness_scores'] = torch.tensor(sem_params[:, :2]).unsqueeze(0)
        end_points['sem_cls_scores'] = torch.tensor(sem_params[:, 2:]).unsqueeze(0)
        parsed_predictions, eval_dict = parse_predictions({}, {}, end_points, {},
                                                          config_dict, return_batch_pred_map_cls=False)
        pred_centers = parsed_predictions['pred_centers']
        pred_sizes = parsed_predictions['pred_sizes']
        pred_headings = parsed_predictions['pred_headings']
        obj_probs = parsed_predictions['obj_prob']
        pred_mask = parsed_predictions['pred_mask']
        vis_bboxes = np.concatenate([pred_centers,pred_sizes,pred_headings[:,:, np.newaxis]],-1)

        for nid in range(nproposal):
            if not valid_masks[iter][nid]: # or not objectness_label[iter][nid] or not  pred_mask[nid]:
                continue
            if objectness_label is not None and not objectness_label[iter][nid]:
                continue
            img = cv2.imread(input_image_paths[iter][nid])
            minx, miny, maxx, maxy =  pts2d_range_list[iter][nid].astype(int)
            cropped_image = img[miny: maxy, minx: maxx]
            cv2.imwrite(os.path.join(output_pth, "%s_%s.jpg"%(scan_name,nid)), cropped_image)
            write_oriented_bbox(np.expand_dims(vis_bboxes[0][nid],0),os.path.join(output_pth, "%s_%s_bbox.ply"%(scan_name,nid)))
            write_ply(anchor_sampled_pts_list[iter][nid],os.path.join(output_pth, "%s_%s_pts.ply"%(scan_name,nid)))



'''
prepare_data_for_recon: prepare data for the training state of reconstruction (feature alignment)
for the training set: only need the rgb features output from detection-refine network (img-encoder)
for the test set: also need predicted bbox arguments for quick evaluation
'''
def prepare_data_for_recon(cfg, tester, conf_thresh = 0.05):
    '''输出：新的包围盒； 新的rgb features; 原先的input points'''
    # train/val: add correspondence (compute_objectness_loss
    # test: new parsed predictions
    # both: add rgb feats

    results = {}
    results['rgb_features'] = []
    mode = cfg.config['data'].get('split_name','val') #train/val/test
    if mode == 'test':
        results['parsed_predictions'] = []
        results['vote_features'] = []
        results['anchor_features'] = []
        results['anchors'] = []
        results['anchor_sampled_xyzs'] = []
        results['valid_mask_list'] = []
        results['scan_names'] = []
        results['sampled_ids'] = []
        ap_calculator = APCalculator(0.5, cfg.dataset_config.class2type, False)


    test_dataset = AnchorRec_ScanNet_For_Refine(cfg, mode='test', augment=False)
    test_loader = DataLoader(dataset=test_dataset,
                                  num_workers= 1,
                                  batch_size= 1,
                                  collate_fn=collate_fn,
                                  shuffle = False,
                                  worker_init_fn=my_worker_init_fn)

    #rgb feats: Nscene, Nproposal, C
    tester.net.train(False)
    print(len(test_loader))
    for iter, data in tqdm(enumerate(test_loader)):
        data = tester.to_device(data)
        scan_name = data['scan_name'][0]
        if mode =='test':
            end_points = tester.net.module.generate(data)
            parsed_predictions = end_points['parsed_predictions']
            eval_dict = end_points['eval_dict']
            ap_calculator.step(eval_dict['batch_pred_map_cls'], eval_dict['batch_gt_map_cls'])

            obj_probs = parsed_predictions['obj_prob']
            pred_mask = end_points['eval_dict']['pred_mask']
            bid = 0
            if 'above_floor_mask' in parsed_predictions:
                pred_mask = pred_mask * parsed_predictions['above_floor_mask']
            #filter by batch_sample_ids and dump_threshold
            sampled_ids = (obj_probs > conf_thresh) * pred_mask #B,N
            sampled_ids = sampled_ids[bid].astype(np.bool)
            parsed_predictions['pred_sizes'] = parsed_predictions['pred_sizes'][bid][sampled_ids]
            parsed_predictions['pred_headings'] = parsed_predictions['pred_headings'][bid][sampled_ids]
            parsed_predictions['pred_centers'] =parsed_predictions['pred_centers'] [bid][sampled_ids]
            parsed_predictions['pred_corners_3d_upright_camera'] = parsed_predictions['pred_corners_3d_upright_camera'][bid][sampled_ids]
            parsed_predictions['sem_cls_probs'] = parsed_predictions['sem_cls_probs'][bid][sampled_ids]
            parsed_predictions['obj_prob'] = parsed_predictions['obj_prob'][bid][sampled_ids]
            parsed_predictions['pred_sem_cls'] = parsed_predictions['pred_sem_cls'][bid][sampled_ids]
            parsed_predictions['pred_mask'] = parsed_predictions['pred_mask'] [bid]
            parsed_predictions['nms_mask'] = parsed_predictions['nms_mask'] [bid]
            vote_features = data['vote_features'][bid].cpu().numpy()[sampled_ids]
            anchor_features = data['anchor_features'][bid].cpu().numpy()[sampled_ids]
            rgb_features = end_points['rgb_features'][bid].cpu().numpy()[sampled_ids]
            anchors = data['anchors'][bid].cpu().numpy()[sampled_ids]
            results['parsed_predictions'].append(parsed_predictions)
            results['vote_features'].append(vote_features)
            results['anchor_features'].append(anchor_features)
            results['rgb_features'].append(rgb_features)
            results['anchors'].append(anchors)
            results['anchor_sampled_xyzs'].append(data['anchor_sampled_xyzs'][bid].cpu().numpy()[sampled_ids])
            results['valid_mask_list'].append(data['valid_mask'][bid].cpu().numpy()[sampled_ids])
            results['scan_names'].append(scan_name)
            results['sampled_ids'].append(sampled_ids)
        else:
            with torch.no_grad():
                end_points = tester.net.module(data)  # , generate_for_test = True)
                rgb_features = end_points['rgb_features'].cpu().numpy()
            results['rgb_features'].append(rgb_features)

    with open(os.path.join(cfg.config['log']['vis_path'], '%s.pkl' % mode), 'wb') as f:
        pickle.dump(results, f)
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        cfg.log_string('eval %s: %f' % (key, metrics_dict[key]))

def prepare_data_for_recon_no_rgb(cfg, tester, test_loader, mode='train'):
    from models.loss import compute_objectness_loss_new
    tester.net.train(False)
    results = {}
    results['input_points'] = []  # 500x3 #anchor_sampled_xyzs
    results['vote_features'] = []
    results['anchor_features'] = []
    results['anchors'] = []
    results['scan_names'] = []
    results['anchor_sampled_pts_list'] = []
    results['original_pc_inds_list'] = []
    if mode=='test':
        #新的objectness score, sem_score; 原来的geo_params; input points/features         # Nproposal不做固定
        results['parsed_predictions'] = []
    else:
        results['objectness_label'] = []
        results['shapenet_ids_list'] = []
        results['shapenet_catids_list'] = []

    results['aggregated_vote_xyz'] = []

    print(len(test_loader))
    for iter, data in tqdm(enumerate(test_loader)):
        #assert batch size =1
        data = tester.to_device(data)
        assert len(data['scan_name'])==1
        end_points = tester.net.module.generate_input(data, return_bbox_params = (mode=='test'))  # , generate_for_test = True)
        nproposal, _ = end_points['geo_params'].shape
        anchor_sampled_pts_list = end_points['anchor_sampled_pts_list']
        original_pc_inds_list = end_points['original_pc_inds_list']
        '''允许每个Proposal长度不一致'''
        results['scan_names'].append(data['scan_name'][0])
        results['anchor_sampled_pts_list'].append(anchor_sampled_pts_list[0])
        results['original_pc_inds_list'].append(original_pc_inds_list[0])
        results['vote_features'].append(end_points['vote_features'][0].t().detach().cpu().numpy())
        results['anchor_features'].append(end_points['anchor_features'][0].t().detach().cpu().numpy())
        results['aggregated_vote_xyz'].append(end_points['aggregated_vote_xyz'][0].detach().cpu().numpy())
        results['anchors'].append(end_points['anchors'][-1][0].detach().cpu().numpy())

        if mode == 'test':
            parsed_predictions = end_points['parsed_predictions']
            results['parsed_predictions'].append(parsed_predictions)
        else:
            objectness_label, object_assignment, soft_cls_labels = \
                compute_objectness_loss_new(end_points, data, cfg.dataset_config, return_loss=False)  # objectness_mask
            shapenet_catids = data['shapenet_catids']
            shapenet_ids = data['shapenet_ids']  # assignment: b,n
            gt_ids = list(object_assignment[0].cpu().numpy())
            objectness_label = objectness_label[0].cpu().numpy()
            shapenet_catids_ = []
            shapenet_ids_ = []
            for i, gt_id in enumerate(gt_ids):
                if objectness_label[i]:
                    shapenet_catids_.append(shapenet_catids[gt_id])
                    shapenet_ids_.append(shapenet_ids[gt_id])
                else:
                    shapenet_catids_.append(0)
                    shapenet_ids_.append(0)
            results['objectness_label'].append(objectness_label)
            results['shapenet_ids_list'].append(shapenet_catids_)
            results['shapenet_catids_list'].append(shapenet_ids_)

    with open(os.path.join(cfg.config['log']['vis_path'], 'data.pkl'), 'wb') as f:
        pickle.dump(results, f)


