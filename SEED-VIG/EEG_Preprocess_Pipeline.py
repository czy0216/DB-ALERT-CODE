import mne
import autoreject

def create_raw_data(eeg_data):
    sfreq = 200
    channel_names = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1',
                     'Pz', 'P2', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
    channel_types = ['eeg'] * len(channel_names)
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.info.set_montage(montage)
    return raw


def apply_filter(raw_filtered, l_freq, h_freq):
    # 陷波滤波（去除50Hz工频噪声）
    raw_filtered = raw_filtered.notch_filter(freqs=50, fir_design='firwin', verbose=False)
    # 带通滤波
    raw_filtered = raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg', filter_length='auto',
                                       fir_design='firwin', verbose=False)
    return raw_filtered


def create_epochs(raw, duration):
    # 固定时间间隔划分
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, verbose=False, preload=True)
    return epochs


def detect_bad_channels(epoch):
    ransac = autoreject.Ransac(n_jobs=2, verbose=True, picks='eeg')
    epoch_clean = ransac.fit_transform(epoch)
    bad_chs = ransac.bad_chs_  # 获取坏导列表
    return epoch_clean, bad_chs


def detect_bad_epochs(epoch):
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4],
                               verbose=False)# 自动清理坏epoch
    epoch_clean, reject_log = ar.fit_transform(epoch, return_log=True)
    return epoch_clean, reject_log



