import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import mne
import scipy

np.random.seed(42)
# 数据划分

def draw_perclos():
    label_path = 'SEEDVIG(Raw)/perclos_Labels'
    label_files = os.listdir(label_path)
    all_label = []
    for label_file in label_files:
        mat = scipy.io.loadmat(os.path.join(label_path, label_file))
        all_label.append(mat['perclos'])
    all_label = np.concatenate(all_label, axis=0)
    np.save('EEG/perclos.npy', all_label)  # (20355,1)
    print(np.median(all_label))
    plt.figure(figsize=(10, 6))
    plt.hist(all_label, bins=100, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Randomly Generated Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def label_segmentation():
    # 二分类分类，全体阈值0.42
    perclos = np.load('EEG/perclos.npy')
    perclos = perclos.ravel()
    print(perclos.shape)
    for t in np.arange(0.35, 0.55, 0.01):
        label = np.zeros_like(perclos)
        label[perclos <= round(t, 4)] = 0 # 高度警觉性
        label[perclos > round(t, 4)] = 1  # 低度警觉性
        np.save(f'EEG/label_{round(t, 2)}.npy', label)
        count_ones = np.count_nonzero(label)
        print(f'{round(t, 2)} label 1 rate', count_ones / label.shape[0])

if __name__ == '__main__':
    label_segmentation()
