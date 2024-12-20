import os
import pickle
import numpy as np
from configs.scannet_config import ScannetConfig
from models.loss import compute_objectness_loss
from tqdm import tqdm
dataset_config = ScannetConfig()

def prepare_data_for_recon(cfg, tester, test_loader):
    split_name = cfg.config['data'].get('split_name','train')
    tester.net.train(False)
    results = {}
    results['input_points'] = []  # 500x3 #anchor_sampled_xyzs
    results['vote_features'] = []
    results['anchor_features'] = []
    results['anchors'] = []
    results['scan_names'] = []
    results['anchor_sampled_pts_list'] = []
    # results['original_pc_inds_list'] = []
    if split_name=='test':
        results['parsed_predictions'] = []
    else:
        results['objectness_label'] = []
        results['shapenet_ids_list'] = []
        results['shapenet_catids_list'] = []

    results['aggregated_vote_xyz'] = []

    print(len(test_loader))
    for iter, data in tqdm(enumerate(test_loader)):
        #assert batch size == 1
        data = tester.to_device(data)
        assert len(data['scan_name'])==1
        end_points = tester.net.module.generate_input(data)  # , generate_for_test = True)
        nproposal, _ = end_points['vote_features'][0].shape
        anchor_sampled_pts_list = end_points['anchor_sampled_pts_list']

        if split_name == 'test':
            vote_features = end_points['vote_features'][0].t().detach().cpu().numpy()
            anchor_features = end_points['anchor_features'][0].t().detach().cpu().numpy()
            aggregated_vote_xyz = end_points['aggregated_vote_xyz'][0].detach().cpu().numpy()
            anchors = end_points['anchors'][-1][0].detach().cpu().numpy()
            parsed_predictions = end_points['parsed_predictions']
            results['parsed_predictions'].append(parsed_predictions)
        else:
            objectness_label, object_assignment = \
                compute_objectness_loss(end_points, data, dataset_config, return_loss=False)  # objectness_mask
            shapenet_catids = data['shapenet_catids'][0]
            shapenet_ids = data['shapenet_ids'] [0] # assignment: b,n
            gt_ids = list(object_assignment[0].cpu().numpy())
            objectness_label = objectness_label[0].cpu().numpy()
            shapenet_catids_ = []
            shapenet_ids_ = []
            for i, gt_id in enumerate(gt_ids):
                if objectness_label[i]:
                    shapenet_catids_.append(shapenet_catids[gt_id])
                    shapenet_ids_.append(shapenet_ids[gt_id])

            # results['objectness_label'].append(objectness_label)
            results['shapenet_ids_list'].append(shapenet_catids_)
            results['shapenet_catids_list'].append(shapenet_ids_)
            anchor_sampled_pts_list = anchor_sampled_pts_list[objectness_label.astype(np.bool_)]
            vote_features = end_points['vote_features'][0].t().detach().cpu().numpy()[objectness_label.astype(np.bool_)]
            anchor_features = end_points['anchor_features'][0].t().detach().cpu().numpy()[objectness_label.astype(np.bool_)]
            aggregated_vote_xyz = end_points['aggregated_vote_xyz'][0].detach().cpu().numpy()[objectness_label.astype(np.bool_)]
            anchors = end_points['anchors'][-1][0].detach().cpu().numpy()[objectness_label.astype(np.bool_)]

        '''Allow each proposal to have different numbers of points'''
        results['scan_names'].append(data['scan_name'][0])
        results['anchor_sampled_pts_list'].append(anchor_sampled_pts_list)
        results['vote_features'].append(vote_features)
        results['anchor_features'].append(anchor_features)
        results['aggregated_vote_xyz'].append(aggregated_vote_xyz)
        results['anchors'].append(anchors)

    with open(os.path.join(cfg.config['log']['vis_path'], '%s.pkl' % split_name), 'wb') as f:
        pickle.dump(results, f)
