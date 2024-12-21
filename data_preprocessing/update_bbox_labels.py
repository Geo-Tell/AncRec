from utils.pc_util import write_ply_color, write_ply, write_oriented_bbox, extract_pc_in_box3d
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from utils.scannet.tools import get_box_corners
from configs.scannet_config import ScannetConfig
from configs.path_config import scannet_processed_path, scannet_processed_path2
import numpy as np
import os
import pickle


dataset_config = ScannetConfig()

'''To fit the problem of the discrepancy between scannet instance labels and ShapeNet model labels'''
def make_instance_label_by_shapenet(output_pth):
    DIST_THRESH = 0.1 #add a little noise
    scan_names = os.listdir(scannet_processed_path)
    for scan_name in tqdm(scan_names):
        data_path = os.path.join(scannet_processed_path,scan_name,'full_scan.npz')
        box_path = os.path.join(scannet_processed_path,scan_name,'bbox.pkl')
        if not os.path.exists(data_path) or not os.path.exists(box_path) :
            print("%s path doesn\'t exist"%scan_name)
            continue
        scan_data = np.load(data_path)
        point_cloud = scan_data['mesh_vertices'][:,:3]
        # point_instance_labels = scan_data['instance_labels']
        # point_votes = scan_data['point_votes']
        with open(box_path, 'rb') as file:
            box_info = pickle.load(file)
        surface_points = np.load(
            os.path.join(scannet_processed_path,scan_name,'object_shape_points.npz'))['points']
        boxid = 0
        point_votes = np.zeros((len(point_cloud), 4))
        for item in box_info:
            # object_instance_id = item['instance_id']
            box3D = item['box3D']
            ''' Find inbox points and then search the ones that are close to object surface'''
            enlarged_size_for_search = box3D[3:6] + 0.2
            surface_point = surface_points[boxid]
            center = box3D[:3]
            orientation = box3D[6]
            axis_rectified = np.array(
                [[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0],
                 [0, 0, 1]])
            vectors = np.diag( enlarged_size_for_search / 2.).dot(axis_rectified)
            box3d_pts_3d = np.array(get_box_corners(center, vectors))
            pc_in_box3d, masks = extract_pc_in_box3d(point_cloud, box3d_pts_3d)
            inds = np.where(masks)[0]
            target_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(surface_point)
            #.fit(target); source-to-target = nn.kneighboirs(sourice)
            min_source_to_target = target_nn.kneighbors(pc_in_box3d)[0]
            valid_ids = inds[min_source_to_target.squeeze()<DIST_THRESH]
            instance_pts = point_cloud[valid_ids]
            item['shapenet_instance_inds'] = valid_ids
            '''votes'''
            point_votes[valid_ids,1:4] = np.expand_dims(center, 0) - instance_pts
            point_votes[valid_ids,0] = 1

            '''vis'''
            if len(instance_pts)==0:
                print('error: %s_box%s'%(scan_name, boxid))
            # # write_oriented_bbox(np.expand_dims(box3D,0),'temp/%s_box%s_box.ply'%(scan_name, boxid))
            # write_ply_color(instance_pts,np.array([255,0,0]), 'temp/%s_box%s_instance_pts.ply'%(scan_name, boxid))
            # write_ply_color(pc_in_box3d,np.array([0,0,255]), 'temp/%s_box%s_inbox_pts.ply'%(scan_name, boxid))
            # write_ply_color(surface_point,np.array([0,255,0]), 'temp/%s_box%s_surface_pts.ply'%(scan_name, boxid))
            # if not os.path.exists('temp/%s.ply'%scan_name):
            #     write_ply(point_cloud[np.random.choice(len(point_cloud), 50000)],'temp/%s.ply'%scan_name)

            boxid += 1

        outpth = os.path.join(output_pth, scan_name)
        if not os.path.exists(outpth):
            os.makedirs(outpth)
        np.save(os.path.join(outpth, 'new_votes.npy'), point_votes)
        with open(os.path.join(outpth, 'bbox.pkl'), 'wb') as file:
            pickle.dump(box_info, file)

if __name__ == '__main__':
    make_instance_label_by_shapenet(scannet_processed_path2)