import torch
import torch.nn
from Nuscene import NusceneDataset
from models import cali_glue
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import time
from torchvision import models
import argparse
import models.vis
from PIL import Image, ImageDraw
import numpy as np
import torch.nn.functional as F
import math
import cv2
from scipy.spatial.transform import Rotation
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='../')
parser.add_argument("--log_dir", default='log_Nuscene_currect/')
parser.add_argument("--checkpoint_dir", type=str, default="ck_Nuscene_currect/")
parser.add_argument("--device", type=str, default='cuda:7')
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--vis3d", type=int, default=3000)
parser.add_argument("--lr", type=float, default=0.00001)
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
log_dir = args.log_dir
model_path = args.checkpoint_dir
batch_size = args.batch_size
num_epochs = args.epoch
lr = args.lr
vis3d = args.vis3d
inT = [args.inT1,
       args.inT2,
       args.inT3,
       args.inT4 * math.pi,
       args.inT5 * math.pi,
       args.inT6 * math.pi, ]
Rdiff_range = args.rdiff
Idiff_range = args.Idiff
if torch.cuda.is_available():
    device = torch.device(args.device)
    print(f"device: {device}")
else:
    device = torch.device("cpu")
    print("CUDA unavailable")


def denormalize_pointcloud(transformed_points, batch_size, centroid, max_distance):
    transformed_points = transformed_points.to('cpu')
    centroid = centroid.view(batch_size, 1, -1) 
    max_distance = max_distance.view(batch_size, 1, 1) 
    denormalized_pointcloud = transformed_points * max_distance + centroid
    return denormalized_pointcloud


def fov_loss(xy_points, fov_score):
    xy_points = xy_points.to('cpu')
    fov_score = fov_score.to('cpu')
    BCELoss = torch.nn.BCELoss()
    score_tensor = torch.ones(fov_score.shape).squeeze(0)
    zero_indices = (xy_points[:, :, 0] == 0) & (xy_points[:, :, 1] == 0)
    score_tensor[zero_indices] = 0
    return BCELoss(fov_score, score_tensor)


def des_loss_org(sxy_points, kp2d, sim_list, scores0, scores1, th1=5, th2=5):
    loss1 = 0
    loss2 = 0
    BCELoss = torch.nn.BCELoss()

    for i in range(kp2d.shape[0]):
        p1 = sxy_points[i].clone().to('cpu')
        p2 = kp2d[i].clone().to('cpu').to(torch.float32)
        sc0 = torch.exp(scores0[i]).squeeze(0).to('cpu')
        sc1 = torch.exp(scores1[i]).squeeze(0).to('cpu')
        score_tensor0 = torch.zeros(sc0.shape)
        score_tensor1 = torch.zeros(sc1.shape)
        zero_indices = (p1[:, 0] == 0) & (p1[:, 1] == 0)
        zero_indices = zero_indices.nonzero().squeeze(1)

        distances = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)

        min_distances1, min_index1 = torch.min(distances, dim=1)
        min_distances2, min_index2 = torch.min(distances, dim=0)
        min_distances1[zero_indices] = th2
        pairs_indices1 = torch.nonzero(min_distances1 < th1, as_tuple=False)
        no_pairs_indices1 = torch.nonzero(min_distances1 >= th2, as_tuple=False)
        pairs_indices2 = torch.nonzero(min_distances2 < th1, as_tuple=False)
        no_pairs_indices2 = torch.nonzero(min_distances2 >= th2, as_tuple=False)

        des_pairs1 = sim_list[i][pairs_indices1, min_index1[pairs_indices1]]
        score_tensor0[pairs_indices1[:, 0]] = 1
        score_tensor1[pairs_indices2[:, 0]] = 1
        sc0loss = BCELoss(sc0, score_tensor0)
        sc1loss = BCELoss(sc1, score_tensor1)
        loss1 += - des_pairs1.mean()
        loss2 += sc0loss + sc1loss
    return loss1 / len(kp2d), loss2 / len(kp2d)


def main():
    writer = SummaryWriter(log_dir)  
    print(f"Params: epochs: {num_epochs}, batch: {batch_size}, lr: {lr}\n")
    data_root = dataset
    sequence_range_train = 'v1.0-trainval'

    train_data = NusceneDataset(data_root, sequence_range_train, T_range=inT, Rdiff_range=Rdiff_range, Idiff_range=Idiff_range)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)


    num_train = len(train_data)
    print('Train dataset size: ', num_train)
    point_cloud_encoder = cali_glue.AttentionModule().to(device)

    optimizer = optim.Adam(point_cloud_encoder.parameters(), lr=lr)
    loaded_checkpoint = torch.load('ck_Nuscene_currect/Encoder_epoch_3.pth', map_location='cpu')
    point_cloud_encoder.load_state_dict(loaded_checkpoint['model_state_dict'])
    optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
    loss_epoch_avg = []
    point_cloud_encoder.train()

    for epoch in range(loaded_checkpoint['epoch'] + 1, num_epochs):
        start_time = time.time() 
        print(f"epoch #{epoch}")
        loss_epoch = []
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
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

            optimizer.zero_grad()

            matches, sim_list, fov_score, score0, score1 = point_cloud_encoder(pointclouds, image, kp3d, kp2d, device)
            H = 900
            W = 1600
            _, top_indices = torch.topk(fov_score, vis3d, dim=1)
            selected_points = torch.gather(kp3d[:, :, :3], dim=1, index=top_indices.expand(-1, -1, 3).to(device))
            selected_points = torch.cat((selected_points, torch.ones_like(selected_points[:, :, :1])), dim=-1)
            restored_pointclouds = denormalize_pointcloud(kp3d[:, :, :3], batch_size, nc, md)
            restored_pointclouds_selected = denormalize_pointcloud(selected_points[:, :, :3], batch_size, nc, md)
            _, xy_points = models.vis.tr3d2d(restored_pointclouds, inter_matrix, transform_matrix, T, H, W)
            pre_image, sxy_points = models.vis.tr3d2d(restored_pointclouds_selected, inter_matrix, transform_matrix, T,
                                                      H, W)

            loss1 = fov_loss(xy_points, fov_score)
            loss2, loss3 = des_loss_org(xy_points, kp2d, sim_list, score0, score1)
            loss = loss1 + loss2 + loss3

            print(loss)

            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_epoch += [loss.item()]
            print("--- %s seconds ---" % (time.time() - start_time))
            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('Loss/Total_Loss', loss.item(), global_step=epoch)
                writer.add_scalar('Loss1/Total_Loss1', loss1.item(), global_step=batch_idx)
                writer.add_scalar('Loss2/Total_Loss2', loss2.item(), global_step=batch_idx)
                writer.add_scalar('Loss3/Total_Loss3', loss3.item(), global_step=batch_idx)
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(
                    epoch, num_epochs, batch_idx, loss.item()))
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Batch Time: {batch_duration:.4f} seconds')
                running_loss = 0.0

                GT_feature_points = kp2d[0].to('cpu').numpy()  
                pre_feature_points = xy_points[0].to('cpu').numpy()
                mat = matches[0].to('cpu')

                x_coords1 = np.array(GT_feature_points[mat[:, 1], 0])
                y_coords1 = np.array(GT_feature_points[mat[:, 1], 1])
                x_coords2 = np.array(pre_feature_points[mat[:, 0], 0])
                y_coords2 = np.array(pre_feature_points[mat[:, 0], 1])

                image_np = image.to('cpu').detach().numpy()
                image_np = np.uint8(image_np[0, :, :, :] * 255)  
                image_np = np.squeeze(image_np)
                pil_image = Image.fromarray(image_np, 'L')
                pil_image = pil_image.convert('RGB')


                draw = ImageDraw.Draw(pil_image)

                camera_matrix = inter_matrix[0, :, :3].numpy()
                image_points = kp2d[0][mat[:, 1], :].to('cpu').numpy()
                image_points = np.array(image_points, dtype=np.float32)
                world_points = restored_pointclouds[0, mat[:, 0], :].to('cpu').numpy()   
                sxy_points = sxy_points.numpy()
                zero_indices = np.where((sxy_points == [0, 0]).all(axis=2))
                zero_point_ratio = len(zero_indices[0]) / (sxy_points.shape[0]*sxy_points.shape[1])
                writer.add_scalar('Z_Rate', zero_point_ratio, global_step=batch_idx)
                try:
                    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(world_points, image_points,
                                                                                           camera_matrix, None, iterationsCount=5000, reprojectionError=8.0, flags=cv2.SOLVEPNP_EPNP)

                except: success = False
                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    rotation_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)   
                    GT_Tr = T_real[0].numpy()
                    GT_R = GT_Tr[:3, :3]
                    GT_R = Rotation.from_matrix(GT_R)
                    GT_RE = GT_R.as_euler('zyx', degrees=True)
                    GT_T = GT_Tr[:3, 3]  
                    Pr_R = Rotation.from_rotvec(rotation_vector.reshape([rotation_vector.shape[0]]))
                    Pr_RE = Pr_R.as_euler('zyx', degrees=True)
                    Pr_T = translation_vector

                    rre = np.sum(np.abs(GT_RE - Pr_RE))   
                    rte = np.linalg.norm(GT_T.reshape(-1, 1) - Pr_T)
                    writer.add_scalar('RRE', rre, global_step=batch_idx)
                    writer.add_scalar('RTE', rte, global_step=batch_idx)
                else:
                    writer.add_scalar('RRE', 360, global_step=batch_idx)
                    writer.add_scalar('RTE', 100, global_step=batch_idx)

                out_point = 0
                if (x_coords1 != np.array([])):
                    for n in range(x_coords1.shape[0]):
                        if (x_coords2[n] == 0) and (x_coords2[n] == 0):
                            out_point += 1
                            continue
                        x1 = x_coords1[n]
                        y1 = y_coords1[n]
                        x2 = x_coords2[n]
                        y2 = y_coords2[n]
                        draw.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], outline='red', width=3)  
                        draw.ellipse([x2 - 3, y2 - 3, x2 + 3, y2 + 3], outline='blue', width=3)  
                        draw.line([(x1, y1), (x2, y2)], fill='green', width=3)
                    pil_image_np = np.array(pil_image)
                    writer.add_image('image', pil_image_np, epoch * len(train_loader) + batch_idx, dataformats='HWC')
                    writer.add_scalar('outline', out_point / x_coords1.shape[0], global_step=batch_idx)

                image_2d = models.vis.proj(kp2d, kp2d.size(0), H, W)
                image_2d = image_2d.permute(0, 3, 2, 1)
                writer.add_image('kp2d', image_2d[0], epoch * len(train_loader) + batch_idx)
                writer.add_image('Predicted Mask point', pre_image[0], epoch * len(train_loader) + batch_idx)

        end_time = time.time()  
        epoch_duration = end_time - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}] Total Time: {epoch_duration:.4f} seconds')
        checkpoint = {'model_state_dict': point_cloud_encoder.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, model_path + f"/Encoder_epoch_{epoch + 1}.pth")

        loss_epoch_avg += [sum(loss_epoch) / len(loss_epoch)]


if __name__ == "__main__":
    main()

