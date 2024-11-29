import torch
import torch.nn
from KITTI_bin import KITTIDataset
from models import cali_glue
from torch.utils.data import  DataLoader
import time
from torchvision import models, transforms
import argparse
import models.vis
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation
import pandas as pd
import json

seed_value = 42 
torch.manual_seed(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./dataset')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--vis3d", type=int, default=3000)
parser.add_argument("--inT1", type=float, default=10)
parser.add_argument("--inT2", type=float, default=10)
parser.add_argument("--inT3", type=float, default=0)
parser.add_argument("--inT4", type=float, default=2.0)
parser.add_argument("--inT5", type=float, default=0)
parser.add_argument("--inT6", type=float, default=0)
parser.add_argument("--rdiff", type=float, default=0.1)
parser.add_argument("--Idiff", type=float, default=0.2)
args = parser.parse_args()
dataset = args.dataset_path
batch_size = args.batch_size
vis3d = args.vis3d
Rdiff_range = args.rdiff
Idiff_range = args.Idiff

inT = [args.inT1,      
       args.inT2,
       args.inT3,
       args.inT4 * math.pi,
       args.inT5 * math.pi,
       args.inT6 * math.pi, ]

if torch.cuda.is_available():
    device = torch.device(args.device)
    print(f"device: {device}")
else:
    device = torch.device("cpu")
    print("CUDA unavailable")


def denormalize_pointcloud(transformed_points, batch_size, centroid, max_distance):
    transformed_points = transformed_points.to(device)
    max_distance = max_distance.to(device)
    centroid = centroid.to(device)
    centroid = centroid.view(batch_size, 1, -1)  
    max_distance = max_distance.view(batch_size, 1, 1)  
    denormalized_pointcloud = transformed_points * max_distance + centroid
    return denormalized_pointcloud

def main():
    data_root = dataset
    sequence_range_test = [9, 10]
    test_data = KITTIDataset(data_root, sequence_range_test, T_range=inT, Rdiff_range=Rdiff_range, Idiff_range=Idiff_range)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    num_train = len(test_data)
    print('Test dataset size: ', num_train)
    point_cloud_encoder = cali_glue.AttentionModule().to(device)
    loaded_checkpoint = torch.load('ck/KITTI.t7', map_location=device)
    point_cloud_encoder.load_state_dict(loaded_checkpoint)

    with torch.no_grad():
        data = dict()
        data['RRE'] = []
        data['RTE'] = []
        # data['FOV'] = []
        for batch_idx, batch in enumerate(test_loader):
            batch_start_time = time.time()
            pointclouds, kp3d, image, kp2d, inter_matrix, transform_matrix, T, nc, md = batch
            pointclouds = pointclouds.to(device)
            pointclouds = pointclouds.float()
            kp3d = kp3d.to(device)
            image = image.to(device)
            kp2d = kp2d.to(device)
            inter_matrix = inter_matrix.float()
            transform_matrix = transform_matrix.float()
            T_real = torch.bmm(transform_matrix, T)
            nc = nc.float()
            md = md.float()
            matches, sim_list, fov_score, score0, score1 = point_cloud_encoder(pointclouds, image, kp3d, kp2d, device)

            H, W = image.size(2), image.size(3)
            _, top_indices = torch.topk(fov_score, vis3d, dim=1)
            selected_points = torch.gather(kp3d[:, :, :3], dim=1, index=top_indices.expand(-1, -1, 3).to(device))
            selected_points = torch.cat((selected_points, torch.ones_like(selected_points[:, :, :1])), dim=-1)
            restored_pointclouds = denormalize_pointcloud(kp3d[:, :, :3], batch_size, nc, md)

            for b in range(batch_size):
                mat = matches[b].to('cpu')
                matdim1 = mat[:, 0]
                is_in_top_indices = torch.isin(matdim1, top_indices[:, :, 0].to('cpu'))
                filtered_mat = mat[is_in_top_indices]

                camera_matrix = inter_matrix[b, :, :3].numpy()
                image_points = kp2d[b][filtered_mat[:, 1], :].to('cpu').numpy()
                image_points = np.array(image_points, dtype=np.float32)
                world_points = restored_pointclouds[b, filtered_mat[:, 0], :].to('cpu').numpy()
                pscore0 = score0[b].squeeze(0)
                pscore0 = pscore0[filtered_mat[:, 0], :].to('cpu').numpy()
                pscore1 = score1[b].squeeze(0)
                pscore1 = pscore1[filtered_mat[:, 1], :].to('cpu').numpy()

                success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(world_points, image_points,
                                                                                           camera_matrix, None, iterationsCount=5000, reprojectionError=6.5, flags=cv2.SOLVEPNP_EPNP)
                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    rotation_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)
                    
                    extra_row = np.array([0, 0, 0, 1])
                    GT_Tr = T_real[b].numpy()
                    GT_Tr = np.vstack([GT_Tr, extra_row])
                    
                    R,_=cv2.Rodrigues(rotation_vector)
                    T_pred=np.eye(4)
                    T_pred[0:3,0:3] = R
                    T_pred[0:3,3:] = translation_vector
                    P_diff=np.dot(np.linalg.inv(T_pred),GT_Tr)
                    t_diff=np.linalg.norm(P_diff[0:3,3])
                    r_diff=P_diff[0:3,0:3]
                    R_diff=Rotation.from_matrix(r_diff)
                    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
                    rte = t_diff
                    rre = angles_diff
                    print(rre, rte)
                    print('success')
                else:
                    rre = np.inf
                    rte = np.inf
                    print('G')
                data['RRE'].append(rre)
                data['RTE'].append(rte)

    df = pd.DataFrame(data)

    excel_file_path = './KITTI.xlsx'
    
    df.to_excel(excel_file_path, index=True)
if __name__ == "__main__":
    main()
