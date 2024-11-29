import os
import math
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from scipy.spatial.transform import Rotation
import cv2

def cartesian_to_spherical(cartesian_points):
    x, y, z = cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return r


def angles2rotation_matrix(self, angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def filter_vertical_lines(lines, angle_threshold=30):
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle_threshold <= angle <= (180 - angle_threshold):
            vertical_lines.append(line)

    return np.array(vertical_lines)


def normalize_pointcloud(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud -= centroid
    max_distance = np.max(np.sqrt(np.sum(pointcloud ** 2, axis=1)))
    pointcloud /= max_distance
    return pointcloud, centroid, max_distance


def normalize_pointcloud_selected(pointcloud, nc, md):
    centroid = nc
    pointcloud -= centroid
    max_distance = md
    pointcloud /= max_distance
    return pointcloud


class KITTIDataset(Dataset):
    def __init__(self, data_root, sequence_range, T_range, Rdiff_range, Idiff_range, shuffle_data=False):
        self.data_root = data_root
        self.sequence_range = sequence_range
        self.data = self.load_data()
        self.T_range = T_range
        self.Rdiff_range = Rdiff_range
        self.Idiff_range = Idiff_range
        if shuffle_data:
            random.shuffle(self.data) 

        self.transform = transforms.Compose([
            transforms.Resize((376, 1250)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485], std=[0.229])  
        ])

    def load_data(self):
        data = []
        for sequence_num in self.sequence_range:
            sequence_dir = os.path.join(self.data_root, f"sequences/{sequence_num:02d}")
            file2_dir = os.path.join(sequence_dir, "test_data_P2_Inst")
            file3_dir = os.path.join(sequence_dir, "test_data_P3_Inst")
            image2_dir = os.path.join(sequence_dir, "image_2")
            image3_dir = os.path.join(sequence_dir, "image_3")

            image2_files = sorted(os.listdir(image2_dir))
            image3_files = sorted(os.listdir(image3_dir))
            file2_name = sorted(os.listdir(file2_dir))
            file3_name = sorted(os.listdir(file3_dir))

            for file2, file3, image2_file, image3_file in zip(file2_name, file3_name, image2_files, image3_files):
                file2_path = os.path.join(file2_dir, file2)
                file3_path = os.path.join(file3_dir, file3)
                image2_path = os.path.join(image2_dir, image2_file)
                image3_path = os.path.join(image3_dir, image3_file)
                data.append((file2_path, image2_path))
                data.append((file3_path, image3_path))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, image_path = self.data[idx]
        image = Image.open(image_path).convert("L")

        new_image = Image.new('L', (1250, 376), 0)

        new_image.paste(image, (0, 0))

        image = self.transform(new_image)
        with open(filename, 'rb') as fin:
            normalized_pointcloud = np.frombuffer(fin.read(20480 * 4 * 4), dtype=np.float32).reshape(-1, 4)
            merged_point_cloud = np.frombuffer(fin.read(3000 * 4 * 4), dtype=np.float32).reshape(-1, 4)
            all_pixels_coordinates = np.frombuffer(fin.read(3000 * 2 * 2), dtype=np.int16).reshape(-1, 2)
            inter_matrix = np.frombuffer(fin.read(3 * 4 * 4), dtype=np.float32).reshape(-1, 4)
            transform_matrix = np.frombuffer(fin.read(3 * 4 * 4), dtype=np.float32).reshape(-1, 4)
            T_inv = np.frombuffer(fin.read(4 * 4 * 4), dtype=np.float32).reshape(-1, 4)
            nc = np.frombuffer(fin.read(3 * 4), dtype=np.float32)
            md = np.frombuffer(fin.read(1 * 4), dtype=np.float32)
            
        return torch.from_numpy(normalized_pointcloud), torch.from_numpy(merged_point_cloud), image, torch.from_numpy(
            all_pixels_coordinates), inter_matrix, transform_matrix, T_inv, nc, md


if __name__ == "__main__":
    data_root = "../dataset"
    sequence_range_train = range(10) 
    data = KITTIDataset(data_root, sequence_range_train)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for pointclouds, image, inter_matrix, transform_matrix, nc, md in DataLoader:
        print('PC:', pointclouds.shape)
        print('img:', image.shape)
        print('inter', inter_matrix.shape)
        print('tran', transform_matrix.shape)
        print('nc', nc.shape)
        print('md', md.shape)