import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn


def compute_pagerank(weights):
    # 由于networkx不支持批处理，我们将处理单个权重矩阵
    G = nx.DiGraph(weights)
    important_values = nx.pagerank(G, weight='weight', alpha=0.85)
    score = np.array([important_values[i] for i in range(len(important_values))])
    return torch.tensor(score, dtype=torch.float32)


class RC_1D(nn.Module):
    def __init__(self, hid_channel):
        super(RC_1D, self).__init__()
        """
        input (batch, num_node, num_length)
        output (batch, num_node, num_length)
        """
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hid_channel, kernel_size=1),
        )
        self.conv2 = nn.Conv1d(in_channels=hid_channel, out_channels=hid_channel, kernel_size=3,
                               padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv1d(in_channels=hid_channel, out_channels=hid_channel, kernel_size=3,
                               padding=1, padding_mode='replicate')

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=hid_channel, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        batch, num_node, feature_dim = x.shape
        x = x.reshape(batch * num_node, feature_dim)
        x = x.unsqueeze(1)  # (batch * num_node, 1, feature_dim)
        x = self.conv1(x)

        out = self.conv2(x)
        x = torch.nn.functional.relu(out + x)

        out = self.conv3(x)
        x = torch.nn.functional.relu(out + x)

        x = self.conv4(x)
        x = x.reshape(batch, num_node, feature_dim)
        return x


class SAPL(nn.Module):
    def __init__(self, feature_dim, n_head, dropout):
        super(SAPL, self).__init__()
        self.n_head = n_head
        self.hid_dim = (feature_dim // 2) * 2 # 转换偶数
        self.d_k = self.hid_dim // n_head
        self.in_fc = nn.Linear(feature_dim, self.hid_dim)

        self.keys = nn.Linear(self.hid_dim, self.hid_dim)
        self.queries = nn.Linear(self.hid_dim, self.hid_dim)
        self.values = nn.Linear(self.hid_dim, self.hid_dim)

        self.ff = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim)
        )
        self.norm1 = nn.LayerNorm(self.hid_dim)
        self.norm2 = nn.LayerNorm(self.hid_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def positional_encoding(self, x):
        batch_size, num_node, feature_dim = x.shape
        # 生成位置索引
        position = torch.arange(0, num_node, dtype=torch.float, device=x.device).unsqueeze(1)  # (num_node, 1)
        # 计算位置编码的除数项
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, device=x.device).float() * (-math.log(10000.0) / feature_dim))
        # 初始化位置编码张量
        pe = torch.zeros(num_node, feature_dim, device=x.device)  # (num_node, feature_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_node, feature_dim)
        x = x + pe
        return x

    def forward(self, x):
        batch_size, num_node, feature_dim = x.shape

        x = self.in_fc(x)
        q = self.queries(x).view(batch_size, num_node, self.n_head, self.d_k)
        k = self.keys(x).view(batch_size, num_node, self.n_head, self.d_k)
        v = self.values(x).view(batch_size, num_node, self.n_head, self.d_k)

        # 转置和调整形状以便点积
        q = q.transpose(1, 2)  # (batch, n_head, num_node, d_k)
        k = k.transpose(1, 2)  # (batch, n_head, num_node, d_k)
        v = v.transpose(1, 2)  # (batch, n_head, num_node, d_k)

        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5  # (batch, n_head, num_node, num_node)
        attn = torch.nn.functional.softmax(attn_scores, dim=-1)  # (batch, n_head, num_node, num_node)

        attn_output = torch.matmul(attn, v)  # (batch, n_head, num_node_less, d_k)
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                    self.hid_dim)  # (batch, num_node_less, hid_dim)
        # 应用残差连接、层归一化和前馈网络
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        x = torch.mean(x, dim=1)
        return x


class RC1D_SAPL(torch.nn.Module):
    def __init__(self, feature_dim, n_head):
        super(RC1D_SAPL, self).__init__()
        self.rc1d = RC_1D(hid_channel=16)  # 超参数
        self.sapl = SAPL(feature_dim=feature_dim,
                         n_head=n_head,  # 超参数
                         dropout=0.2)
        self.out_feature = self.sapl.hid_dim

    def forward(self, x):
        # batch, num_node, feature_dim
        x = self.rc1d(x)
        x_atte = self.sapl(x)

        return x, x_atte


class Each_Branch(torch.nn.Module):
    def __init__(self, feature_dim):
        super(Each_Branch, self).__init__()
        self.block1 = RC1D_SAPL(
            feature_dim=feature_dim,
            n_head=1,
        )
        self.out1 = nn.Sequential(
            nn.Linear(in_features=self.block1.out_feature, out_features=self.block1.out_feature),
        )
        self.block2 = RC1D_SAPL(
            feature_dim=feature_dim,
            n_head=1
        )
        self.out2 = nn.Sequential(
            nn.Linear(in_features=self.block2.out_feature, out_features=self.block2.out_feature),
        )
        self.out_size = self.block2.out_feature

    def forward(self, x):
        batch, num_node, feature_dim = x.shape

        x, x_att1 = self.block1(x)

        x, x_att2 = self.block2(x)

        """融合层级"""
        # delete_ratio越大说明保留的突出属性越多，则要求模型决策分配的权重越大
        x_att1 = self.out1(x_att1)
        x_att2 = self.out2(x_att2)

        branch_out = (x_att1 + x_att2)
        return branch_out


class No_Prune(torch.nn.Module):
    def __init__(self, num_channels, num_frequencies):
        super(No_Prune, self).__init__()
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies

        self.chan_freq_branch = Each_Branch(feature_dim=self.num_frequencies)
        self.freq_chan_branch = Each_Branch(feature_dim=self.num_channels)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.chan_freq_branch.out_size + self.freq_chan_branch.out_size, out_features=16),
            nn.ReLU(),
            torch.nn.Linear(in_features=16,out_features=2),
        )

    def forward(self, x):
        # 双分支
        chan_freq_out = self.chan_freq_branch(x)
        freq_chan_out = self.freq_chan_branch(x.permute(0, 2, 1))
        out = torch.cat((chan_freq_out, freq_chan_out), dim=1)
        out = self.out(out)
        return out

