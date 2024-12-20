import argparse
from configs.config_utils import CONFIG
import shutil
from net_utils.common_utils import initiate_environment
import torch
from net_utils.utils import load_device, load_model, load_tester
from net_utils.utils import CheckpointIO
from configs.config_utils import mount_external_config
from configs.path_config import scannet_processed_path, split_path
import os
import numpy as np
import pickle
from utils import pc_util
from save_scene_mesh import save_scene_mesh
from utils.read_and_write import read_json
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Instance Scene Completion.')
    parser.add_argument('--config', type=str, default='configs/final_recon.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='demo', help='train, test or demo.')
    return parser.parse_args()


def load_configurations(cfg):
    '''Mount external config data'''
    cfg = mount_external_config(cfg, eval_map=False, demo=True)
    cfg.LOCAL_RANK = 0
    cfg.dist = False
    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)
    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)
    checkpoint = CheckpointIO(cfg)
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load existing checkpoint'''
    if 'bsp_weight' in cfg.config:
        checkpoint.load_bsp_decoder()
    checkpoint.parse_checkpoint()

    '''Load tester'''
    cfg.log_string('Loading tester.')
    tester = load_tester(cfg=cfg, net=net, device=device)

    '''Start to test'''
    cfg.log_string('Start to test.')
    cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))

    cfg.log_string('Loading dataset.')
    # _, test_loader, _ = load_dataloader(cfg, mode='test')
    return tester #, test_loader

if __name__ == '__main__':
    # dump_dir = "/home/lht/dmy/submit_version/out/recon_all/2024-05-28T17:54:39.374617/visualization"
    # dump_scene_mesh_dir = "/home/lht/dmy/submit_version/out/recon_all/2024-05-28T17:54:39.374617/scene_meshes"
    # scan_name = "scene0378_02"
    # save_scene_mesh(dump_dir, scan_name, dump_scene_mesh_dir, align_to_gt=False)

    args = parse_args()
    cfg = CONFIG(args.config)
    cfg.update_config(args.__dict__)
    initiate_environment(cfg.config)

    shutil.copy(args.config, 'config.yaml')
    cfg.copy_all_code()
    '''Configuration'''
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)
    cfg.write_config()
    tester  = load_configurations(cfg)
    dump_dir = cfg.config['log']['vis_path']
    dump_scene_mesh_dir = '/'.join(dump_dir.split('/')[:-1]) + '/scene_meshes'
    os.makedirs(dump_scene_mesh_dir)
    tester.net.train(False)
    if 'vis_scan_names' not in cfg.config['log'] or type(cfg.config['log']['vis_scan_names']) != list:
        split_file = os.path.join(split_path,'fullscan','original', 'scannetv2_test.json')
        split_data = read_json(split_file)
        scan_names = [item["scan"].split('/')[-2] for item in split_data]
    else:
        scan_names = cfg.config['log']['vis_scan_names']

    pc_num_points = cfg.config['data']['num_point']
    for iter, scan_name in enumerate(scan_names):
        input_pc_pth = os.path.join(scannet_processed_path, scan_name, "full_scan.npz")
        if not os.path.exists(input_pc_pth):
            print("%s doesn't exist!"%input_pc_pth)
            continue
        scan_data = np.load(input_pc_pth)["mesh_vertices"]
        if cfg.config['data']['no_height']:
            point_cloud = scan_data[:,:3]
        else:
            floor_height = np.percentile(scan_data[:, 2], 0.99)
            height =  scan_data[:, 2] - floor_height
            point_cloud = np.concatenate([scan_data[:,:3], np.expand_dims(height, 1)], 1)
        point_cloud, choices = pc_util.random_sampling(point_cloud, pc_num_points, return_choices=True)
        cfg.log_string(scan_name)
        data = dict()
        data['point_clouds'] = torch.tensor(point_cloud.astype(np.float32)).unsqueeze(0).cuda()
        data['scan_name'] = [scan_name]
        data['choices'] = torch.tensor(choices).unsqueeze(0).cuda()
        est_data = tester.net.module.generate(data)
        tester.save_results(iter, data, est_data, dump_dir,save_gt=False)
        save_scene_mesh(dump_dir, scan_name, dump_scene_mesh_dir, align_to_gt=False)


    cfg.write_config()
    cfg.log_string('Generation finished')



