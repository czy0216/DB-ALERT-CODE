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

class RC1D_SAPL(torch.nn.Module):
    def __init__(self, feature_dim, n_head):
        super(RC1D_SAPL, self).__init__()
        self.rc1d = RC_1D(hid_channel=16)  # 超参数

    def forward(self, x):
        # batch, num_node, feature_dim
        x = self.rc1d(x)
        return x


class Each_Branch(torch.nn.Module):
    def __init__(self, feature_dim):
        super(Each_Branch, self).__init__()
        self.block1 = RC1D_SAPL(
            feature_dim=feature_dim,
            n_head=1,
        )

        self.block2 = RC1D_SAPL(
            feature_dim=feature_dim,
            n_head=1
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        branch_out = torch.mean(x, dim=1)
        return branch_out


class Only_RC(torch.nn.Module):
    def __init__(self, num_channels, num_frequencies):
        super(Only_RC, self).__init__()
        self.num_channels = num_channels
        self.num_frequencies = num_frequencies

        self.chan_freq_branch = Each_Branch(feature_dim=self.num_frequencies)
        self.freq_chan_branch = Each_Branch(feature_dim=self.num_channels)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_channels + self.num_frequencies, out_features=16),
            nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=2),
        )

    def forward(self, x):
        # 双分支
        chan_freq_out = self.chan_freq_branch(x)
        freq_chan_out = self.freq_chan_branch(x.permute(0, 2, 1))

        out = torch.cat((chan_freq_out, freq_chan_out), dim=1)
        out = self.out(out)
        return out

