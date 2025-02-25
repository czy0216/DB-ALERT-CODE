import numpy as np
import os
from matplotlib import pyplot as plt, gridspec
from mne.time_frequency import psd_array_welch, psd_array_multitaper
import json
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from sklearn.preprocessing import StandardScaler

from scipy.integrate import quad
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


def draw_psd_each_channel(psd_data: [], psd_freqs, save_path, name):
    """绘制PSD，逐个通道的频谱图 psd_data [(17,n),(17,n),(17,n),..]"""
    ch_name = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
               'CP1', 'CP2', 'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']
    for ch in range(len(ch_name)):
        plt_data = []
        for item in psd_data:
            plt_data.append(item[ch])
        plt.figure(figsize=(12, 8))
        for i, item in enumerate(plt_data):
            plt.subplot(3, 2, i + 1)
            plt.plot(psd_freqs, item)
            plt.title(name[i])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (μV^2/Hz)')
            plt.grid()
        plt.tight_layout()
        plt.suptitle(f'Each channel spectrum {ch_name[ch]}', fontsize=16)
        plt.subplots_adjust(top=0.9)  # 调整标题和子图之间的距离
        plt.savefig(f'{save_path}/{ch_name[ch]}.png')
        # plt.show()
        plt.close()


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


def compute_psd():
    """
    abs_psd 各通道各频率段的平均psd
    rel_psd1 各通道 频率段的相对psd
    rel_psd2 全脑相对psd
    """
    # eeg_raw = np.load('EEG/eeg_raw.npy')  # (20355,17,1600)
    # print('原始信号: ', eeg_raw.shape)
    for n_fft in [256, 512, 1024]:
        # # method 1
        # psds = np.load(f'EEG/welch_{n_fft}_psd.npy')
        # freqs = np.load(f'EEG/welch_{n_fft}_freqs.npy')
        # print(psds.shape)
        # # 参数设置
        # window_size = 1  # 每个频率窗口宽度为1Hz
        # min_freq = 1  # 最小频率
        # max_freq = 50  # 最大频率
        # # 初始化保存积分结果的数组
        # integrated_psds = np.zeros((psds.shape[0], psds.shape[1], max_freq - min_freq))
        # # 遍历所有样本和电极
        # for i in range(psds.shape[0]):
        #     print(i)
        #     for j in range(psds.shape[1]):
        #         # 获取当前电极的PSD值
        #         psd_values = psds[i, j, :]
        #         # 生成样条插值器，避免每次积分时都生成
        #         interpolator = interp1d(freqs, psd_values, kind='quadratic', bounds_error=False,
        #                                 fill_value="extrapolate") # 二阶函数插值
        #         # 遍历每个频率窗口
        #         for k in range(min_freq, max_freq):
        #             # 使用样条插值并计算每个频率窗口的曲线下面积
        #             integrated_psd, _ = quad(func=interpolator, a=k, b=(k + window_size))
        #             integrated_psds[i, j, k - min_freq] = integrated_psd
        # # 输出查看积分结果的形状
        # print(integrated_psds.shape)  # 应该是 (20355, 19, 49)
        # # 保存积分结果
        # np.save(f'EEG/{n_fft}_abs_psd_integral.npy', integrated_psds)

        # method 2
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
    psd = np.load(f'EEG/{n_fft}_abs_psd_integral.npy') # PSD值（为窗口）
    # rel_psd = np.load(f'EEG/welch_{n_fft}_rel_psd.npy')  # 通道内相对
    # rel_all_psd = np.load(f'EEG/welch_{n_fft}_rel_all_psd.npy')  # 全脑相对
    # abs_psd = np.load(f'EEG/welch_{n_fft}_abs_psd.npy')  # 绝对

    psd_freqs = np.arange(1, 50, 1)
    label = np.load('EEG/label_0.42.npy').ravel()

    for i in range(23):
        start_index = i * 885
        end_index = (i + 1) * 885
        sub_psd = psd[start_index:end_index]
        sub_label = label[start_index:end_index]

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
        save_path = f'PSD/平均值去掉两端最值'
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
    psd_method = ['welch', 'multitaper']
    stat_ana_method = ['mannwhitneyu','kruskal']
    """
    is_draw = True
    """计算psd值"""
    for psd_method in ['welch']:
        """计算psd值"""
        compute_psd()
        """绘制PSD图"""
        draw_psd()
