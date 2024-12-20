import os
import numpy as np
from configs.path_config import layout_path, scannet_processed_path, scans_dir
from utils.scannet.scannet_planes import get_quads, projection2d
from utils.scannet import scannet_utils
import torch
SVOTE_NUM = 1024

def transform(scene_name, mesh_vertices):
    # meta_file = os.path.join(path_config.metadata_root, 'scans', scene_name, scene_name + '.txt')  # includes axis
    meta_file = os.path.join(scans_dir, scene_name, scene_name + '.txt')  # includes axis
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]
    return mesh_vertices


def getSVotes():
    for scene_name in os.listdir(scannet_processed_path):
        plane_dir =layout_path + '/' +  scene_name + '.json'
        if not os.path.exists(plane_dir):
            continue
        if not os.path.exists(os.path.join(scannet_processed_path, scene_name)):
            continue
        vertical_quads, total_quad_num, horizontal_quads = get_quads(scene_name)
        if len(vertical_quads)==0:
            continue

        '''Generate plane votes'''
        '''Start by finding points on the wall planes'''
        DIST_THRESHOLD = 0.1
        scan_data_path = os.path.join(scans_dir, scene_name, scene_name+'_vh_clean_2.ply')
        pc_ = scannet_utils.read_mesh_vertices_rgb(scan_data_path)
        pc = transform(scene_name,pc_)
        pc = torch.tensor(pc[:, 0:3])

        quads = torch.tensor(vertical_quads)
        # quad_center, vertical_normal_vector, w, h)
        quads_center = quads[:,0:3] # M *3
        quads_vector = quads[:,3:6]
        quads_size = quads[:, 6:8]
        floor = np.mean(vertical_quads[:, 2] - vertical_quads[:, 7]/2, axis=0)

        distz = torch.abs(pc[:,2].unsqueeze(1).repeat(1,2) - torch.tensor(floor).unsqueeze(0).repeat(pc.shape[0],1))
        dists = projection2d(pc, quads_center, quads_vector,quads_size)

        min_dist, ind= torch.min(torch.cat((dists,distz),dim=1),dim=1) #N

        N = pc.shape[0]
        point_votes = torch.zeros((N, 3)).to(dists.device)
        for num in range(dists.shape[1]):
            ind1 = torch.where(ind == num)[0]
            ind2 = torch.where(min_dist < torch.tensor(DIST_THRESHOLD))[0]
            indi = set(ind1.cpu().numpy()) & set(ind2.cpu().numpy())
            if len(indi)==0:
                print('no pc in plane')
                continue
            indi = np.array(list(indi))
            # point_votes[indi] =  quads_center[num].float() - pc[indi]
            point_votes[indi] =  quads_center[num].float()
        np.savez(os.path.join(output_path, scene_name, 'quad_votes.npz'), quad_votes=point_votes.numpy())

        '''Generate plane shape anchors'''
        quads_center = vertical_quads[:,0:3] # M *3
        quads_vector = vertical_quads[:,3:6]
        quads_size = vertical_quads[:, 6:8]
        '''Sample shape anchors from the wall plane (uniform sampling) '''
        svote_dist = np.random.uniform(-quads_size/2, quads_size/2, size=(SVOTE_NUM,quads_size.shape[0],2))
        svote_z = quads_center[:,2] + svote_dist[:,:, 1]
        svote_y = quads_center[:, 1] + svote_dist[:,:, 0]*  quads_vector[:, 0] / np.linalg.norm(quads_vector[:, 0:2],axis=1)
        svote_x = quads_center[:, 0] - svote_dist[:, :,0] * quads_vector[:, 1] / np.linalg.norm(quads_vector[:, 0:2],axis=1)
        svote = np.concatenate([np.expand_dims(svote_x,axis=2), np.expand_dims(svote_y,axis=2), np.expand_dims(svote_z,axis=2)],axis = 2)
        np.savez(os.path.join(output_path, scene_name, 'quad_svotes.npz'), quad_svotes= svote)


if __name__ == '__main__':
    getSVotes()
