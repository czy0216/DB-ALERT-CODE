import os
from mne.preprocessing import ICA
from mne_icalabel import label_components
from EEG_Preprocess_Pipeline import create_raw_data, detect_bad_epochs, baseline_cor
from EEG_Preprocess_Pipeline import apply_filter_ref
from EEG_Preprocess_Pipeline import create_epochs
from EEG_Preprocess_Pipeline import detect_bad_channels
from matplotlib import pyplot as plt

random_state = 10
"""EEG预处理流水线  
19导联 
"""
def eeg_preprocess():
    raw_eeg_root_path = '../Raw_EEG'
    clean_eeg_root_path = '../Clean_EEG'
    raw_eeg_root_path_listdir = os.listdir(raw_eeg_root_path)
    for i_subject, raw_eeg_root_path_dir in enumerate(raw_eeg_root_path_listdir):
        print(f'============第{i_subject + 1}名被试============')
        raw_eegs = os.listdir(os.path.join(raw_eeg_root_path, raw_eeg_root_path_dir))
        if len(raw_eegs) < 5:
            continue
        raw_eegs.sort()
        for i_trial, eeg_filename in enumerate(raw_eegs):
            print(f'第{i_trial + 1}试次：{eeg_filename}')
            edf_path = os.path.join(raw_eeg_root_path, raw_eeg_root_path_dir, eeg_filename)
            """1.创建600s数据 1.0s-601.0s的数据"""
            raw = create_raw_data(edf_path)
            """2.滤波(1~50)  参考电极A1 A2"""
            raw_filtered = apply_filter_ref(raw.copy(), l_freq=1, h_freq=50)
            """3.创建epochs,以10s为一段"""
            epochs = create_epochs(raw_filtered.copy(), duration=10)
            """4.修复坏导联"""
            epochs, bad_chs = detect_bad_channels(epochs=epochs.copy(), repair=True)
            print('ICA前坏导:', bad_chs)
            """5.坏epochs【检测】"""
            reject_log = detect_bad_epochs(epochs=epochs.copy(), repair=False)
            """6.ICA without bad channel/bat epoch"""
            N_ICA = 14  # 尽量比19少一些，因为有坏导
            ica = ICA(n_components=N_ICA, max_iter="auto", method='infomax', fit_params=dict(extended=True),
                      random_state=random_state)
            # 无坏导联 无坏epoch情况下ICA
            ica_fig_path = '../ICA_无坏epoch'
            ica.fit(epochs[~reject_log.bad_epochs], verbose=False)
            """
            ICLabel 自动识别ICA成分　注意！！　
            ICLabel 要求应用平均重参考，但是这里应用双侧耳垂平均重参考。然而MNE中并没有否认这种做法,经过人眼对比，发现其去除的成分正确
            ICLabel 要求的ICA算法是 infomax
            """
            # 自动识别ICA成分
            ic_labels = label_components(epochs, ica, method="iclabel")
            labels = ic_labels["labels"]
            print(ic_labels["labels"])
            exclude_idx = [
                idx for idx, label in enumerate(labels) if label in ["eye blink", "muscle artifact"]
            ]
            exclude_str = '_'.join(map(str, exclude_idx))
            print(f"移除ICA成分: {exclude_idx}")
            # ICA可视化
            fig = ica.plot_components(show=False)
            fig.savefig(f'{ica_fig_path}/{i_subject + 1}_{i_trial + 1}@{exclude_str}.png')
            # 去除ICA成分 : reconst_epochs
            reconst_epochs = ica.apply(epochs.copy(), exclude=exclude_idx)

            """保存"""
            print(reconst_epochs.__len__)
            index = i_trial + 1 + 5 * i_subject
            save_path = os.path.join(clean_eeg_root_path, '{:02}_epo.fif'.format(index))
            print('保存至：', save_path)
            reconst_epochs.save(fname=save_path, overwrite=True)
        print('】')

if __name__ == '__main__':
    eeg_preprocess()
