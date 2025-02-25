import os
import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components
from EEG_Preprocess_Pipeline import create_raw_data, detect_bad_epochs
from EEG_Preprocess_Pipeline import apply_filter
from EEG_Preprocess_Pipeline import create_epochs
from EEG_Preprocess_Pipeline import detect_bad_channels
import scipy

import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
"""EEG预处理流水线"""

def eeg_preprocess():
    raw_eeg_root_path = 'SEEDVIG(Raw)/EEG_Raw_mat'
    clean_eeg_root_path = 'EEG_Clean'

    raw_eeg_root_path_listdir = os.listdir(raw_eeg_root_path)
    raw_all_data = []
    for i_subject, eeg in enumerate(raw_eeg_root_path_listdir):
        mat = scipy.io.loadmat(os.path.join(raw_eeg_root_path, eeg))
        eeg = mat['dataMatrix'].T
        print(f'第{i_subject + 1}名被试,数据规模:', eeg.shape)
        """1.创建raw数据"""
        raw = create_raw_data(eeg)
        """2.滤波(1~50) """
        raw_filtered = apply_filter(raw, l_freq=1, h_freq=50)
        """3.创建epochs,以8s为一段"""
        epochs = create_epochs(raw_filtered, duration=8)
        data = epochs.get_data(copy=True)  # (885,17,1600)
        raw_all_data.append(data)
    raw_all_data = np.concatenate(raw_all_data, axis=0)
    print('finish 1.2.3:', raw_all_data.shape)
    N = raw_all_data.shape[0]
    clean_all_data = []
    for i_sample in range(N):
        eeg = raw_all_data[i_sample]
        eeg = np.expand_dims(eeg, axis=0)
        print(f'第 {i_sample + 1} 样本,数据规模:', eeg.shape)
        sfreq = 200
        channel_names = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1',
                         'Pz', 'P2', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
        channel_types = ['eeg'] * len(channel_names)
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
        epoch = mne.EpochsArray(data=eeg, info=info, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        epoch.info.set_montage(montage)
        """4.检测且修复坏电极"""
        epoch_god, bad_chs = detect_bad_channels(epoch=epoch)
        print('坏导:', bad_chs)
        """5.ICA """
        N_ICA = 16  # 尽量比17少一些，因为有坏导
        ica = ICA(n_components=N_ICA,
                  max_iter="auto",
                  method='infomax',
                  random_state=97,
                  fit_params=dict(extended=True))
        ica.fit(epoch_god, verbose=False)
        """
        ICLabel 自动识别ICA成分　注意！！　
        ICLabel 要求应用平均重参考，我们的参考方式不同。然而MNE中并没有否认这种做法,经过人眼对比，发现其去除的成分正确
        ICLabel 要求的ICA算法是 infomax
        """
        # 自动识别ICA成分
        ic_labels = label_components(epoch_god, ica, method="iclabel")
        labels = ic_labels["labels"]
        print(ic_labels["labels"])
        exclude_idx = [
            idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
        ]
        exclude_str = '_'.join(map(str, exclude_idx))
        print(f"Excluding these ICA components: {exclude_idx}")
        # ICA可视化
        fig = ica.plot_components(show=False)
        fig.savefig(f'ICA成分/{i_sample + 1}@{exclude_str}.png')
        # 去除ICA成分 : reconst_epochs
        reconst_epochs = epoch_god.copy()
        ica.apply(reconst_epochs, exclude=exclude_idx, verbose=False)
        """保存"""
        index = i_sample + 1
        save_path = os.path.join(clean_eeg_root_path, '{:05}_epo.fif'.format(index))
        reconst_epochs.save(fname=save_path, overwrite=True)
        data = reconst_epochs.get_data(copy=True)
        clean_all_data.append(data)
        print(f'save {save_path} ', data.shape)
    clean_all_data = np.concatenate(clean_all_data, axis=0)
    print('finish 1.2.3.4.5:', clean_all_data.shape)
    np.save('EEG/eeg_raw.npy', clean_all_data)


if __name__ == '__main__':
    eeg_preprocess()

