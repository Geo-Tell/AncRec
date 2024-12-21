156:
/home/dmy/indoor3D/data/datasets/posed_images_final
/home/dmy/indoor3D/data/datasets/posed_images_final_instance: output_instance2imageid.pkl,output_instance_ratio.pkl
#point cloud, bboxes, vote and shape points 
/home/dmy/indoor3D/data/datasets/processed_data
/home/dmy/indoor3D/data/datasets/scannet/processed_data2

155
/data/scannet
/data/scannet/scan2cad

sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/scannet_planes /data/dong/anchorrec/scannet_planes

[//]: # (sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets /data/dong/anchorrec/scannet)
sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/processed_data /data/dong/anchorrec/processed_data
[//]: # (sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/scannet/processed_data2 /data/dong/anchorrec/processed_data2)
processed_data2: copyed from 192.168.210.159: /mnt/backup/home/lht/dmy/data/processed_data2
sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/posed_images_final /data/dong/anchorrec/posed_images_final
sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/posed_images_final_instance /data/dong/anchorrec/posed_images_final_instance
sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/ShapeNetv2_data /data/dong/anchorrec/ShapeNetv2_data
sudo mount 192.168.210.155:/data/scannet/scan2cad /data/dong/anchorrec/scan2cad
sudo mount 192.168.210.155:/data/scannet/scannet_planes /data/dong/anchorrec/scannet_planes
sudo mount 192.168.210.155:/data/scannet/scans /data/dong/anchorrec/scans


# sudo mount 192.168.210.155:/data/scannet/dmy/datasets/posed_images_final /home/dmy/data/datasets/scannet/posed_images_final
# sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/posed_images_final /home/dmy/data/datasets/scannet/posed_images_final
# sudo mount 192.168.210.156:/home/dmy/indoor3D/data/datasets/posed_images_final_instance /home/dmy/data/datasets/scannet/posed_images_final_instance
# sudo mount 192.168.210.155:/data/scannet/dmy/datasets /home/dmy/data/datasets/scannet
# sudo mount 192.168.210.155:/data/scannet/scannet_planes /home/dmy/data/datasets/scannet_planes
# sudo mount 192.168.210.155:/data/scannet/scans /home/dmy/data/scans


scannet_path = '/home/dmy/data/datasets/scannet'
scannet_processed_path = '/home/dmy/data/datasets/scannet/processed_data'
scannet_processed_path2 = '/home/lht/dmy/data/processed_data2' # '/home/dmy/data/datasets/processed_data2'
data_root_path = '/home/dmy/data'
layout_path = '/home/dmy/data/datasets/scannet_planes'
scan2cad_path = '/home/dmy/data/scan2cad'
raw_scans_path = '/home/dmy/data/scans'
posed_images_instance_path =  '/home/dmy/data/datasets/scannet/posed_images_final_instance'
posed_images_path =  '/home/dmy/data/datasets/scannet/posed_images_final'
