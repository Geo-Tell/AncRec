import re
from evaluation.cd.metrics import *

def extract_score(f):
    return float(f[:-4].split('_')[-1])
    

def eval(data_dir, scene_names, threshs,  per_class_proposal = False, remove_below_floor = False):
    
    log_file = open(f'eval_log_cd.txt', 'a')

    # prepare calcs
    from configs.scannet_config import ScannetConfig
    dataset_config = ScannetConfig()
    ap_calculator_list = [APCalculator(cd_thresh, dataset_config.class2type) for cd_thresh in threshs]

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):
        file_path = os.path.join(data_dir, scene_name)
        if not os.path.isdir(file_path):
            continue
        gt_mesh_paths  = [f for f in os.listdir(file_path) if re.search(r'target_(\d+)_class_(\d+)_mesh.obj', f)]
        pred_mesh_paths  = [f for f in os.listdir(file_path) if re.search(r'target_(\d+)_proposal_(\d+)_class_(\d+)_mesh.ply', f)]
        proposal_ids = [ int( f[:-4].split('_')[-4]) for f in pred_mesh_paths]
        if len(pred_mesh_paths)>1:
            sorted_ids = np.argsort(np.array(proposal_ids))
            pred_mesh_paths= [pred_mesh_paths[sorted_id] for sorted_id in sorted_ids]

        if remove_below_floor:
            above_floor_mask = np.load(os.path.join(file_path, 'above_floor_mask.npy'))
            pred_mesh_paths = pred_mesh_paths[above_floor_mask]


        info_mesh_gts = []
        for f in gt_mesh_paths:
            mesh = trimesh.load(os.path.join(file_path,f), process=False)
            label = int(f[:-4].split('_')[-2])
            info_mesh_gts.append((label, mesh))


        # pred mesh
        obj_probs = np.load(os.path.join(file_path, 'obj_probs.npy'))
        info_mesh_preds = []
        idx = 0
        for f in  pred_mesh_paths:
            mesh = trimesh.load(os.path.join(file_path,f), process=False)
            label = int(f[:-4].split('_')[-2])
            score = obj_probs[idx]
            info_mesh_preds.append((label, mesh, score))
            idx += 1

        # record
        for calc in ap_calculator_list:
            calc.step(info_mesh_preds, info_mesh_gts)

        print(f'[step {sid}/{len(scene_names)}] {scene_name}')

    # output
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- thresh = {threshs[i]} -----')
        print(f'----- thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            if 'Q_mesh' in k: continue
            if 'mesh' not in k: continue
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file)
    
    log_file.close()



if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('gt_dir', type=str)
    # parser.add_argument('pred_dir', type=str)
    #
    # args = parser.parse_args()
    # eval(args.gt_dir, args.pred_dir, threshs=[0.047, 0.1])

    data_dir = '/home/dmy/indoor3D/myResearch/BSPInScene/out/generate/2/test_big/2/visualization'
    scene_names = os.listdir(data_dir)

    eval(data_dir, scene_names, threshs=[0.047, 0.1])



