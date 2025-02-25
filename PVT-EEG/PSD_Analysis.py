import mne
import numpy as np
import os
from matplotlib import pyplot as plt, gridspec
from mne.time_frequency import psd_array_welch, psd_array_multitaper
import json

from sklearn.preprocessing import StandardScaler


def get_delete_sample():
    with open('delete_index.json', 'r') as f:
        data = json.load(f)
    delete_index = data['delete_index']
    return delete_index


def median_psd(psd_data):
    return np.median(psd_data, axis=0)


def mean_psd(psd_data):
    return np.mean(psd_data, axis=0)


def mean_psd_without_max_and_min(psd_data, trim_ratio=0.01, return_mean=True):
    # 对每个通道、每个频率的值排序
    sorted_psd_data = np.sort(psd_data, axis=0)
    lower_idx = int(trim_ratio * len(psd_data))
    upper_idx = int((1 - trim_ratio) * len(psd_data))
    trimmed_psd_data = sorted_psd_data[lower_idx:upper_idx]
    if return_mean:

        trimmed_mean_psd = np.mean(trimmed_psd_data, axis=0)
        return trimmed_mean_psd
    else:
        return trimmed_psd_data



def draw_psd_time_freqs(psd_data, psd_freqs, save_path, name, num_subject):
    global im
    if not psd_data or not isinstance(psd_data, list) or any(not isinstance(item, np.ndarray) for item in psd_data):
        raise ValueError("psd_data must be a list of numpy arrays.")
    if len(psd_data) != len(name):
        raise ValueError("Each PSD matrix must have a corresponding name.")

    # 计算全局 vmin 和 vmax (使用 10% 和 90% 分位数)
    all_values = np.concatenate([matrix.flatten() for matrix in psd_data])
    vmin = np.percentile(all_values, 10)
    vmax = np.percentile(all_values, 90)

    # 确定子图布局
    num_plots = len(psd_data)
    cols = 2  # 两列
    rows = (num_plots + 1) // cols  # 计算所需行数

    # 创建一个 GridSpec 来控制子图和颜色条的布局
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1] * cols + [0.05])  # 最后一列为颜色条留出空间

    # 绘制每个PSD数据矩阵的热力图
    for i, data in enumerate(psd_data):
        ax = fig.add_subplot(gs[i // cols, i % cols])  # 动态布局
        im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis',
                       extent=[psd_freqs[0], psd_freqs[-1], 0, data.shape[0]], vmin=vmin, vmax=vmax)
        ax.set_title(name[i])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Channel')

    # 添加颜色条
    cbar_ax = fig.add_subplot(gs[:, -1])  # 使用最后一列的所有行
    fig.colorbar(im, cax=cbar_ax)

    # 调整布局和保存图像
    plt.tight_layout()
    # plt.savefig(f'{save_path}/{num_subject}.png')
    plt.show()
    plt.close()


def compute_psd(time_size, fmin, fmax):
    eeg_raw = np.load(f'EEG/{time_size}s_eeg.npy')  # (1400, 19, 6000)
    print('原始信号: ', eeg_raw.shape)
    for n_fft in [256, 512, 1024]:
        psds, freqs = psd_array_welch(x=eeg_raw, sfreq=200.0, n_jobs=-1, average='mean',
                                      n_fft=n_fft,
                                      n_overlap=int(n_fft / 2),  # 50%
                                      fmin=fmin, fmax=fmax, verbose=False)  # 频域分辨率=采样率/n_fft n_fft越大分辨率越大
        print(psds.shape, freqs.shape)
        np.save(f'EEG/welch_{n_fft}_psd.npy', psds)
        np.save(f'EEG/welch_{n_fft}_freqs.npy', freqs)
        print(freqs)
        """rel_psd2 全脑相对psd 各通道 频率段的相对psd"""
        psds = np.load(f'EEG/welch_{n_fft}_psd.npy')
        freqs = np.load(f'EEG/welch_{n_fft}_freqs.npy')
        print(f'psd {psds.shape} freqs {freqs.shape}')
        # 设置窗口大小和步长
        window_size = 1
        step_size = 1
        min_freq = 1  # 最小频率
        max_freq = 50  # 最大频率
        # 计算 M
        M = (max_freq - min_freq) // step_size - (window_size - step_size)
        # 初始化结果矩阵，用于存储每个窗口的绝对PSD，以落在该窗口内的PSD均值为特征。
        abs_psd = np.zeros((psds.shape[0], psds.shape[1], M))
        for i in range(M):
            start_freq = min_freq + i * step_size
            end_freq = start_freq + window_size
            # 找到当前窗口对应的频率索引
            mask = (freqs >= start_freq) & (freqs <= end_freq)
            # 计算绝对PSD(一个区间的平均值)
            abs_psd[:, :, i] = np.mean(psds[:, :, mask], axis=2)
        print(abs_psd.shape)
        all_psd = np.sum(abs_psd, axis=(1, 2), keepdims=True)
        rel_psd = abs_psd / all_psd
        np.save(f'EEG/welch_{n_fft}_rel_all_psd.npy', rel_psd)
        np.save(f'EEG/welch_{n_fft}_abs_psd.npy', abs_psd)
        """rel_psd1 各通道 频率段的相对psd"""
        psds = np.load(f'EEG/welch_{n_fft}_psd.npy')
        freqs = np.load(f'EEG/welch_{n_fft}_freqs.npy')
        print(f'psd {psds.shape} freqs {freqs.shape}')
        # 设置窗口大小和步长
        window_size = 1
        step_size = 1
        min_freq = 1  # 最小频率
        max_freq = 50  # 最大频率
        # 计算 M
        M = (max_freq - min_freq) // step_size - (window_size - step_size)
        # 初始化结果矩阵，用于存储每个窗口的绝对PSD，以落在该窗口内的PSD均值为特征。
        abs_psd = np.zeros((psds.shape[0], psds.shape[1], M))
        for i in range(M):
            start_freq = min_freq + i * step_size
            end_freq = start_freq + window_size
            # 找到当前窗口对应的频率索引
            mask = (freqs >= start_freq) & (freqs <= end_freq)
            # 计算绝对PSD(一个区间的平均值)
            abs_psd[:, :, i] = np.mean(psds[:, :, mask], axis=2)
        print(abs_psd.shape)
        all_psd = np.sum(abs_psd, axis=2, keepdims=True)
        rel_psd = abs_psd / all_psd
        np.save(f'EEG/welch_{n_fft}_rel_psd.npy', rel_psd)
        np.save(f'EEG/welch_{n_fft}_abs_psd.npy', abs_psd)

def draw_psd():
    n_fft = 512
    psd = np.load(f'EEG/welch_{n_fft}_psd.npy') # PSD值（为窗口）
    rel_psd = np.load(f'EEG/welch_{n_fft}_rel_psd.npy')  # 通道内相对
    rel_all_psd = np.load(f'EEG/welch_{n_fft}_rel_all_psd.npy')  # 全脑相对
    abs_psd = np.load(f'EEG/welch_{n_fft}_abs_psd.npy')  # 绝对

    psd_freqs = np.load(f'EEG/welch_{n_fft}_freqs.npy')
    for psd_type in [
        'psd'
        # 'rel_psd',
        # 'rel_all_psd'
        # 'abs_psd'
    ]:
        target_psd = None
        if psd_type == 'psd':
            target_psd = psd
        if psd_type == 'rel_psd':
            target_psd = rel_psd
        if psd_type == 'rel_all_psd':
            target_psd = rel_all_psd
        if psd_type == 'abs_psd':
            target_psd = abs_psd
        N, num_channels, num_frequencies = target_psd.shape
        print(N, num_channels, num_frequencies)

        # target_psd = target_psd.reshape(target_psd.shape[0], -1)
        # scaler = StandardScaler()
        # target_psd = scaler.fit_transform(target_psd)
        # target_psd = target_psd.reshape(target_psd.shape[0], num_channels, num_frequencies)

        label = np.load('Label/30s_label_2.npy').ravel()

        for i in range(14):
            start_index = i * 100
            end_index = (i + 1) * 100
            sub_psd = target_psd[start_index:end_index]
            sub_label = label[start_index:end_index]

            sub_psd = sub_psd.reshape(sub_psd.shape[0], -1)
            scaler = StandardScaler()
            sub_psd = scaler.fit_transform(sub_psd)
            sub_psd = sub_psd.reshape(sub_psd.shape[0], num_channels, num_frequencies)

            print(sub_psd.shape, sub_label.shape)
            psd_h_alert = sub_psd[sub_label == 0]  # 清醒(高度警觉性)
            psd_l_alert = sub_psd[sub_label == 1]  # 困倦(低度警觉性)

            num_samples = psd_h_alert.shape[0]
            indices = np.random.permutation(num_samples)
            shuffled_data = psd_h_alert[indices]
            split_index = num_samples // 2
            h1 = shuffled_data[:split_index]
            h2 = shuffled_data[split_index:]

            num_samples = psd_l_alert.shape[0]
            indices = np.random.permutation(num_samples)
            shuffled_data = psd_l_alert[indices]
            split_index = num_samples // 2
            l1 = shuffled_data[:split_index]
            l2 = shuffled_data[split_index:]
            print(h1.shape, h2.shape, l1.shape, l2.shape)
            """绘图：时频图"""
            # save_path = f'PSD对比/tf_{psd_method}/平均值'
            # os.makedirs(save_path, exist_ok=True)
            # draw_psd_time_freqs([mean_psd(h1),
            #                      mean_psd(h2),
            #                      mean_psd(l1),
            #                      mean_psd(l2)],
            #                     psd_freqs,
            #                     save_path,
            #                     ['high alert 1', 'high alert 2', 'low alert 1','low alert 2'])
            # save_path = f'PSD对比/tf_{psd_method}/中位数'
            # os.makedirs(save_path, exist_ok=True)
            # draw_psd_time_freqs([median_psd(h1),
            #                      median_psd(h2),
            #                      median_psd(l1),
            #                      median_psd(l2)],
            #                     psd_freqs,
            #                     save_path,
            #                     ['high alert 1', 'high alert 2', 'low alert 1', 'low alert 2'])
            save_path = f'PSD/{psd_type}/平均值去掉两端最值'
            os.makedirs(save_path, exist_ok=True)
            draw_psd_time_freqs([mean_psd_without_max_and_min(h1),
                                 mean_psd_without_max_and_min(h2),
                                 mean_psd_without_max_and_min(l1),
                                 mean_psd_without_max_and_min(l2)],
                                psd_freqs,
                                save_path,
                                ['high alert 1', 'high alert 2', 'low alert 1', 'low alert 2'], i)


if __name__ == '__main__':
    """
    stat_ana_method = ['mannwhitneyu','kruskal']
    """
    time_size = 30  # 秒 分段时间
    n_class = 2
    is_draw = True
    # """计算psd值"""
    compute_psd(time_size=time_size, fmin=1, fmax=50)
    draw_psd()
