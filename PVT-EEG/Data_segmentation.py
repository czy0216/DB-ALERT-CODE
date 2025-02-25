import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import mne

"""对RT EEG数据进行分段"""
np.random.seed(42)

def eeg_segmentation(time_size, load):
    os.makedirs(f'{time_size}_Seg_EEG', exist_ok=True)
    if load:  # 重新计算
        eeg_path = 'Clean_EEG'
        sfreq = 200
        sampling_points = int(time_size * sfreq)
        eeg_files = os.listdir(eeg_path)
        eeg_data = []
        count = 1
        for eeg_file in eeg_files:
            epochs = mne.read_epochs(os.path.join(eeg_path, eeg_file), verbose=False)
            info = epochs.info
            data = epochs.get_data(copy=True)  # data.shape 是 (n_epochs, n_channels, n_times)
            # 将所有 epoch 拼接成连续信号，shape 为 (n_channels, n_samples)
            data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)  # (n_channels, n_samples)
            # 分段
            for i in range(0, data.shape[1], sampling_points):
                if i + sampling_points <= data.shape[1]:  # 忽略结尾小于分段长度的部分
                    segment = data[:, i:i + sampling_points]
                    new_raw = mne.io.RawArray(segment, info, verbose=False)
                    new_raw.save('{}_Seg_EEG/{:04}_eeg.fif'.format(time_size, count), overwrite=True)
                    count += 1
                    eeg_data.append(segment)
        eeg_data = np.array(eeg_data)
        np.save(f'EEG/{time_size}s_eeg.npy', eeg_data)
        return eeg_data
    else:
        eeg_data = np.load(f'EEG/{time_size}s_eeg.npy')
        return eeg_data


def rt_segmentation(time_size, load):
    """分类问题"""
    n_class = 2
    if load:  # 重新计算
        with open('rt.json', 'r', encoding='utf-8') as json_file:
            rt_json = json.load(json_file)
        rt_seg = []
        for key, value in rt_json.items():
            value = value['rt']
            n_segments = (600 // time_size)
            rt_avg = [[] for _ in range(n_segments)]
            for (start_time, end_time, rt_ms) in value:
                segment_idx = int(end_time // time_size)
                if segment_idx < n_segments:
                    rt_avg[segment_idx].append(rt_ms)
            rt_avg = [np.mean(seg) if seg else 0 for seg in rt_avg]
            rt_seg.append(rt_avg)
        rt_seg = np.array(rt_seg)  # 每一段的平均值
        rt_seg = rt_seg.ravel()
        # 使用 for 循环遍历 rt_seg,去掉平均值为0的段
        for i in range(1, len(rt_seg)):
            if rt_seg[i] == 0:
                rt_seg[i] = rt_seg[i-1]
        print(np.median(rt_seg))

        label = np.zeros_like(rt_seg)
        if n_class == 2:
            """二分类"""
            label[rt_seg <= 357.18] = 0 # 低RT————高警觉性
            label[rt_seg > 357.18] = 1 # 高RT————低警觉性

        np.save(f'Label/{time_size}s_label_{n_class}.npy', label)
        return label
    else:
        label = np.load(f'Label/{time_size}s_label_{n_class}.npy')
        return label

def draw_rc():
    with open('rt.json', 'r', encoding='utf-8') as json_file:
        rt_json = json.load(json_file)
    all_rt = []
    for key, value in rt_json.items():
        value = value['rt']
        for (start_time, end_time, rt_ms) in value:
            all_rt.append(rt_ms)
    print(all_rt.__len__())

if __name__ == '__main__':
    time_size = 10 #　10秒一段
    load = True
    label = rt_segmentation(time_size, load)
    # print(label.shape)
    # eeg_data = eeg_segmentation(time_size, load)
