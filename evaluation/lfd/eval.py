# offline mesh IoU eval

import glob


from metrics import *

from lfd import MeshEncoder

def extract_label(f):
    clsname = f[:-4].split('_')[3]
    if clsname == 'trash': clsname = 'trash_bin'
    return CAD_labels.index(clsname)

def extract_score(f):
    cls_score = float(f[:-4].split('_')[-1])
    obj_score = float(f[:-4].split('_')[-2])
    return  obj_score # * cls_score



def eval(gt_dir, pred_dir, threshs):
    # log_file = open(f'inbox_eval_log_lfd.txt', 'a')
    log_file = open(pred_dir + '/' + 'log_lfd2.txt', 'w')  # a
    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    # collect meshes (ply)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.ply')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))

    data = {}

    for f in gt_files:
        scene_name = os.path.basename(f)[:12]
        if scene_name not in data: data[scene_name] = {}
        if 'gt' not in data[scene_name]: data[scene_name]['gt'] = []
        if 'pred' not in data[scene_name]: data[scene_name]['pred'] = []
        data[scene_name]['gt'].append(f)

    for f in pred_files:
        scene_name = os.path.basename(f)[:12]
        data[scene_name]['pred'].append(f)

    scene_names = data.keys()
    #scene_names = ['scene0011_00']

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):

        # gt mesh
        gt_files_scene = data[scene_name]['gt']
        info_mesh_gts = []
        for f in gt_files_scene:
            
            mesh = MeshEncoder(trimesh.load(f, process=False), './cache_gt', os.path.basename(f)[:-4])
            #mesh = MeshEncoder(trimesh.load(f, process=False))
            mesh.align_mesh(verbose=True)

            label = extract_label(os.path.basename(f))
            info_mesh_gts.append((label, mesh))


        # pred mesh
        pred_files_scene = data[scene_name]['pred']
        info_mesh_preds = []
        for f in pred_files_scene:
            
            mesh = MeshEncoder(trimesh.load(f, process=False), f'./cache_pred_{pred_dir.split("/")[-2]}', os.path.basename(f)[:-4])
            #mesh = MeshEncoder(trimesh.load(f, process=False))
            mesh.align_mesh(verbose=True)

            label = extract_label(os.path.basename(f))
            score = extract_score(os.path.basename(f))
            # label = extract_label_gt(os.path.basename(f))
            # score = 1
            if score > 0.5:
                info_mesh_preds.append((label, mesh, score))

        # record
        for calc in ap_calculator_list:
            calc.step(info_mesh_preds, info_mesh_gts)

        print(f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(gt_files_scene)},  #pred = {len(pred_files_scene)}')

    # output
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str)
    parser.add_argument('pred_dir', type=str)
    
    args = parser.parse_args()
    from pyvirtualdisplay import Display
    with Display(size=(60, 60)) as disp:  # backend="xvfb"
        eval(args.gt_dir, args.pred_dir, threshs=[2500, 5000])


