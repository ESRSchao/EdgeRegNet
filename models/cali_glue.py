import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from torchvision import models
from models.superpoint import SuperPoint, sample_descriptors
import math
import numpy as np
import copy
from typing import Optional, List, Callable, Tuple
import warnings
torch.autograd.set_detect_anomaly(True)

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_coords(feature_points, image_width, image_height):
    # 归一化特征点坐标到[0, 1]范围
    normalized_points = copy.deepcopy(feature_points.float())
    # normalized_points[:, :, 0] /= image_width
    # normalized_points[:, :, 1] /= image_height

    one = normalized_points.new_tensor(1)
    size = torch.stack([one*image_width, one*image_height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (normalized_points - center[:, None, :]) / scaling[:, None, :]

    # return normalized_points

class KeypointEncoder2d(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder2d = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder2d[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        e = self.encoder2d(inputs)
        return e

class KeypointEncoder3d(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder3d = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder3d[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts
        return self.encoder3d(inputs)

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([copy.deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1



def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    oa = F.logsigmoid(z0)
    ob = F.logsigmoid(z1)
    certainties = oa + ob.transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores, oa, ob


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores, oa, ob = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim, oa, ob

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1

class AttentionModule(nn.Module):
    def __init__(self, d_model=256, num_layers=8, nhead=4):
        super(AttentionModule, self).__init__()
        self.superpoint = SuperPoint(config={'descriptor_dim': 256,
                                             'nms_radius': 4,
                                             'keypoint_threshold': 0.005,
                                             'max_keypoints': -1,
                                             'remove_borders': 4,
                                             })
        # for param in self.superpoint.parameters():
        #     param.requires_grad = False
        self.num_layers = num_layers
        self.sa1 = PointNetSetAbstraction(5120, 0.1, 256, 4 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(1280, 0.2, 128, 64 + 3, [64, 64, 128], False)
        # self.sa3 = PointNetSetAbstraction(320, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.fp = PointNetFeaturePropagation(132, [128, 256, 256])
        # self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        # self.fp2 = PointNetFeaturePropagation(320, [256, 256])
        # self.fp1 = PointNetFeaturePropagation(256, [256, 256, 256])
        # self.output_layer = nn.Linear(d_model, 256)
        # self.output_linear = nn.Linear(d_model, 1)
        self.feature_layer=nn.Sequential(nn.Conv1d(256,256,1,bias=False),nn.BatchNorm1d(256),nn.ReLU())
        self.pc_score_layer=nn.Sequential(nn.Conv1d(256,256,1,bias=False),nn.BatchNorm1d(256),nn.ReLU(),nn.Conv1d(256,64,1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,1,1,bias=False),nn.Sigmoid())
        self.kenc2d = KeypointEncoder2d(d_model, [32, 64, 128, 256])
        self.kenc3d = KeypointEncoder3d(d_model, [32, 64, 128, 256])
        self.gnn = AttentionalGNN(d_model, ['self', 'cross'] * 6)
        self.log_assignment = MatchAssignment(d_model)
        self.input_proj = nn.Identity()
    def forward(self, xyz, image, kp3d, kp2d, device):
        H, W = image.size(2), image.size(3)
        _, _, descriptors = self.superpoint(image, kp2d)
        kp3d = kp3d.to(torch.float32).permute(0, 2, 1)
        kp2d_nor = normalize_coords(kp2d, W, H)
        des2d = torch.stack(descriptors)
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        kp_3d_xyz = kp3d[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        des3d = self.fp(kp_3d_xyz, l2_xyz, kp3d, l2_points)
        desc0 = self.input_proj(des3d)
        desc1 = self.input_proj(des2d)
        encoding0 = self.kenc3d(kp_3d_xyz)
        encoding1 = self.kenc2d(kp2d_nor)
        desc0 = desc0 + encoding0
        desc1 = desc1 + encoding1
        desc0, desc1 = self.gnn(desc0, desc1)
        cloud_features = desc0
        image_features = desc1

        # cloud_features = cloud_features.permute(0, 2, 1)
        # image_features = image_features.permute(0, 2, 1)
        # l3_points[i] = cloud_features
        # des2d[i] = image_features

        # l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        des3d = self.feature_layer(cloud_features)
        des2d = self.feature_layer(image_features)
        # x = torch.empty(batch_size, l0_points.size(2), 1)
        out = self.pc_score_layer(cloud_features).permute(0, 2, 1)
        # out = self.sigmoid(out)
        # out = self.tanh(out)
        match_list = []
        sim_list = []
        scores0 = []
        scores1 = []

        for i in range(des3d.shape[0]):
            scores, sim, oa, ob = self.log_assignment(des3d[i].transpose(1, 0).unsqueeze(0), des2d[i].transpose(1, 0).unsqueeze(0))  
            # scores0.append(self.log_assignment.get_matchability(l0_points[i].unsqueeze(0).permute(0, 2, 1)))
            # scores1.append(self.log_assignment.get_matchability(des2d[i].permute(0, 2, 1)))
            scores0.append(oa)
            scores1.append(ob)
            
            m0, m1, mscores0, mscores1 = filter_matches(scores, 0.0)
            valid = m0[0] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[0][valid]
            match_list.append(torch.stack([m_indices_0, m_indices_1], -1))
            sim_list.append(scores.squeeze(0))  # Use this while training
            # sim_list.append(torch.exp(scores.squeeze(0)[m_indices_0, m_indices_1]))
        return  match_list, sim_list, out, scores0, scores1
        # return  kp2d, kp3d, matches, sim_list, fov_score, score0, score1

