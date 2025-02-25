import math
import scipy.stats as stats
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway


def calculate_p_values(X, Y):
    num_electrodes, num_points = X.shape[1], X.shape[2]
    p_values = np.zeros((num_electrodes, num_points))
    for i in range(num_electrodes):
        for j in range(num_points):
            group1 = X[Y == 0, i, j]
            group2 = X[Y == 1, i, j]
            f_stat, p_value = stats.f_oneway(group1, group2)  # ANOVA

            p_values[i, j] = p_value
    threshold = 0.001
    row_proportions = np.mean(p_values < threshold, axis=1)  # 对应电极，比例越高，效果越显著
    col_proportions = np.mean(p_values < threshold, axis=0)  # 对应频率
    sorted_row_indices = np.argsort(-row_proportions)  # 从大到小排序，负号实现降序
    sorted_col_indices = np.argsort(-col_proportions)  # 同上

    sorted_channel_names = channel_names[sorted_row_indices]
    sorted_freqs = freqs[sorted_col_indices]
    print(sorted_channel_names)
    print(sorted_freqs)
    return p_values

n_fft = 512
X = np.load(f'../EEG/welch_{n_fft}_rel_all_psd.npy')  # 全脑相对
Y = np.load(f'../Label/label_2.npy').ravel()


N_sample = X.shape[0]
num_channels = X.shape[1]
num_frequencies = X.shape[2]
freqs = np.arange(1, 50, 1)
"""
前额叶 pre-frontal (Fp) 
额叶 frontal lobe (F) 
颞叶 temporal lobe (T) 
顶叶 parietal lobe (P) 
枕叶 occipital lobe (O)
中央 central  (C)
"""
channel_names = np.array(['Fp1', 'Fp2', # 前额叶
                          'F3', 'F4', # 额叶
                          'C3', 'C4', # 中央区
                          'P3', 'P4', # 顶叶
                          'O1', 'O2', # 枕叶
                          'F7', # 左侧额叶
                          'F8', # 右侧额叶
                          'T3', # 左颞叶
                          'T4', # 右颞叶
                          'T5', # 左后颞叶
                          'T6', # 右后颞叶
                          'Fz', # 额叶
                          'Cz', # 中央区
                          'Pz']) # # 顶叶
p_values = calculate_p_values(X, Y)



