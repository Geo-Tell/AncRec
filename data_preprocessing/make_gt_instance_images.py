import os
import pickle
from tqdm import tqdm
import numpy as np
from configs.path_config import scannet_processed_path, scannet_processed_path2, posed_images_instance_path, \
    posed_images_path, shapenetid2class, instance_gt_rgb_path, split_path
import cv2
from utils.scannet.extract_posed_images import get_random_range
from utils.read_and_write import read_json
MIN_NUM_PTS_THRESH = 500
MIN_RANGE_THRESH = 30
def make_gt_images(output_path, image_size=400):
    scan_names = os.listdir(scannet_processed_path)
    for scan_name in tqdm(scan_names):
        if os.path.exists(os.path.join(output_path,scan_name)):
            continue
        if not os.path.exists(os.path.join(scannet_processed_path, scan_name,'full_scan.npz')):
            continue
        scan_data = np.load(os.path.join(scannet_processed_path, scan_name,'full_scan.npz'))
        point_instance_labels = scan_data['instance_labels']
        with open(os.path.join(scannet_processed_path2, scan_name, 'bbox.pkl'), 'rb') as file:
            box_info = pickle.load(file)
        with open(os.path.join(posed_images_path, scan_name, 'ptids_images.pkl'), 'rb') as f:
            ptids_images = pickle.load(f)
        with open(os.path.join(posed_images_instance_path, scan_name,'output_instance2imageid.pkl'), 'rb') as f:
            instance2imageid = pickle.load(f)
        for boxid, item in enumerate(box_info):
            shapenet_instance_inds  = item['shapenet_instance_inds']
            cls_id = item['cls_id']
            class_ind = shapenetid2class[cls_id]
            '''There exists a small discrepency between the shapeNet instance label (bbox label) and ScanNet instance label'''
            '''Find the corresponding ScanNet instance label for each bbox as follows'''
            box_pts_inst_labels = point_instance_labels[shapenet_instance_inds]
            unique_box_pts_inst_labels = np.unique(box_pts_inst_labels)
            instance_pts_numbers = [len(np.where(box_pts_inst_labels == instance_ind)[0]) for instance_ind in
                                    unique_box_pts_inst_labels]
            best_intance_id = unique_box_pts_inst_labels[np.argmax(np.array(instance_pts_numbers))]
            # max_instance_pts_num = instance_pts_numbers[best_intance_id]
            if best_intance_id not in instance2imageid:
                continue
            '''Find the images for each instance if the number of visible projection points are enough 
               (according to ptids_images.pkl and output_instance2imageid.pkl)
               Sort these images by the number of visible projection points  '''

            for iii in np.arange(10):
                img_id = instance2imageid[best_intance_id][iii].astype(int)
                ptids_image = ptids_images[img_id]  # [max_count_id]
                ptids_image_flattened = ptids_image.flatten()
                ids_pc_mask = np.in1d(shapenet_instance_inds, ptids_image_flattened)  # pc_inds允许重复
                num_pts_in_image = ids_pc_mask.sum()
                if num_pts_in_image > MIN_NUM_PTS_THRESH:
                    output_file_pth = os.path.join(output_path, scan_name, "boxid%s_%s_%s.jpg"%(boxid, iii, class_ind))
                    if os.path.exists(output_file_pth):
                        continue
                    on_image_ids = np.where(np.in1d(ptids_image_flattened,shapenet_instance_inds))[0]
                    ids_image_x = on_image_ids % ptids_image.shape[1]
                    ids_image_y = np.floor(on_image_ids / ptids_image.shape[1])
                    output_pts2d = np.concatenate([ids_image_x, ids_image_y]).astype(int).reshape(2,-1).transpose()  # N,2
                    img_path = os.path.join(posed_images_path, scan_name,
                                            str(img_id).zfill(5) + '.jpg')
                    image = cv2.imread(img_path)
                    minx,miny,maxx, maxy = get_random_range(image, output_pts2d , margin=10, shift_range=0., scale_ratio=1.)
                    if maxy-miny< MIN_RANGE_THRESH  or maxx - minx <MIN_RANGE_THRESH :
                        continue
                    cropped_image = image[miny: maxy, minx: maxx]  # + 1
                    cropped_image = cv2.resize(cropped_image, (image_size, image_size))
                    if not os.path.exists(os.path.join(output_path, scan_name)):
                        os.makedirs(os.path.join(output_path, scan_name))
                    cv2.imwrite(os.path.join(output_path, scan_name, "boxid%s_%s_%s.jpg"%(boxid, iii, class_ind)), cropped_image)


def split_data(input_path, output_path):
    train_split_lines = read_json(os.path.join(split_path, 'fullscan', 'quad_joint', 'scannetv2_train.json'))
    test_split_lines = read_json(os.path.join(split_path, 'fullscan', 'quad_joint', 'scannetv2_test.json'))
    train_scene_names = []
    test_scene_names = []
    train_filename_list = []
    test_filename_list = []
    for split in train_split_lines:
        train_scene_names.append(split.split('/')[-2])
    for split in test_split_lines:
        test_scene_names.append(split.split('/')[-2])

    scan_names = os.listdir(input_path)
    for scan_name in scan_names:
        scene_dir = os.path.join(input_path, scan_name)
        filenames = os.listdir(scene_dir)
        filenames.sort()
        for filename in filenames:
            if scan_name in train_scene_names:
                train_filename_list.append(scan_name + '/' + filename)
            else:
                test_filename_list.append(scan_name + '/' + filename)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, "train.txt"), "w") as f:
        f.writelines(train_filename_list)
    with open(os.path.join(output_path, "test.txt"), "w") as f:
        f.writelines(test_filename_list)

if __name__ == '__main__':
    make_gt_images(instance_gt_rgb_path, image_size=400)
    split_data(instance_gt_rgb_path, instance_gt_rgb_path)