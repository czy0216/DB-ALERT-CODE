import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet(nn.Module):
    def __init__(self, num_channels, num_samples, num_classes):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, (1, 5), stride=(1, 3), bias=False)
        self.conv2 = nn.Conv2d(25, 25, (num_channels, 1), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(25)
        self.pooling1 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(25, 50, (1, 5), stride=(1, 3), bias=False)
        self.batch_norm2 = nn.BatchNorm2d(50)
        self.pooling2 = nn.MaxPool2d((1, 2))
        self.conv4 = nn.Conv2d(50, 100, (1, 5), stride=(1, 3), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(100)
        self.pooling3 = nn.MaxPool2d((1, 2))
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(100 * 27, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.batch_norm1(self.conv2(F.relu(self.conv1(x)))))
        x = self.pooling1(x)
        x = F.relu(self.batch_norm2(self.conv3(x)))
        x = self.pooling2(x)
        x = F.relu(self.batch_norm3(self.conv4(x)))
        x = self.pooling3(x)
        # print(x.shape)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class ShallowConvNet(nn.Module):
    def __init__(self, num_channels, num_samples, num_classes):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (num_channels, 13), padding=(0, 6))
        self.conv2 = nn.Conv2d(40, 40, (1, 1))
        self.batch_norm1 = nn.BatchNorm2d(40)
        self.pooling = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(40 * 853, 40),
            nn.Linear(40, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = x * torch.sigmoid(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class EEGNet(nn.Module):
    def __init__(self, num_channels, samples_rate, num_classes):
        super(EEGNet, self).__init__()
        self.first_conv = nn.Conv2d(1, 16, (1, samples_rate // 2), padding=(0, samples_rate // 4))
        self.depthwise_conv = nn.Conv2d(16, 32, (num_channels, 1), groups=16)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.separable_conv = nn.Conv2d(32, 32, (1, 16))
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(32 * 185, 32),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.first_conv(x))
        x = F.relu(self.depthwise_conv(x))
        x = self.batch_norm1(x)
        x = self.pooling1(x)
        x = F.relu(self.separable_conv(x))
        x = self.batch_norm2(x)
        x = self.pooling2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


import torch
import torch.nn as nn
from torchvision.models import resnet18, vit_b_16


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 加载预训练的ResNet18模型
        self.resnet = resnet18()
        # 修改第一层卷积以适应单通道输入
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, bias=False)
        # 根据需要更改输出层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # 假设为二分类问题

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.resnet(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=emb_size, kernel_size=3, stride=3)

    def forward(self, x):
        batch, num_node, feature_dim = x.shape
        x = x.reshape(batch * num_node, feature_dim)
        x = x.unsqueeze(1)  # (batch * num_node, 1, feature_dim)
        x = F.relu(self.conv(x))
        x = x.reshape(batch, num_node, x.shape[1], x.shape[2])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch, -1, x.shape[-1])
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=16, num_heads=2, ff_hidden=16, dropout=0.2):
        super().__init__()
        self.query = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)

        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, emb_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        q = self.query(x)
        k = self.keys(x)
        identity = x
        x, _ = self.attention(q, k, x)
        x = x + identity
        x = self.norm1(x)
        identity = x
        x = self.ff(x)
        x = x + identity
        x = self.norm2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, emb_size=64, depth=1, num_heads=1, ff_hidden=64 * 4):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels=1, emb_size=emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = 304
        self.positional_embedding = self.create_positional_encoding(num_patches, emb_size)
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoderBlock(emb_size, num_heads, ff_hidden) for _ in range(depth)])
        self.mlp_head = nn.Sequential(nn.Linear(emb_size, 16),
                                      nn.ReLU(),
                                      nn.Linear(16, num_classes))

    @staticmethod
    def create_positional_encoding(num_positions, emb_size):
        # Use sine and cosine functions to generate positional encodings
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pos_embedding = torch.zeros((num_positions, emb_size))
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pos_embedding.unsqueeze(0))

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.positional_embedding
        x = self.transformer_blocks(x)
        cls_token_final = torch.mean(x, dim=1)
        x = self.mlp_head(cls_token_final)
        return x
