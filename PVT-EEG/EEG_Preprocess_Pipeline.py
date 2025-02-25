import mne
import autoreject

random_state = 10

def create_raw_data(file):
    raw = mne.io.read_raw_edf(input_fname=file, eog=None, misc=None,
                              infer_types=True,
                              preload=True,
                              exclude=['POL E', 'POL PG1', 'POL PG2',
                                       'POL T1', 'POL T2', 'POL X1',
                                       'POL X2', 'POL X3', 'POL X4',
                                       'POL X5', 'POL X6', 'POL X7',
                                       'POL SpO2', 'POL EtCO2', 'POL DC03',
                                       'POL DC04', 'POL DC05', 'POL DC06',
                                       'POL Pulse', 'POL CO2Wave', 'POL $A1',
                                       'POL $A2',
                                       ],  # 排除22个无关的
                              encoding='latin1',
                              verbose=False)
    new_ch_names = {ch_name: ch_name.replace('-Ref', '') for ch_name in raw.ch_names}
    raw.rename_channels(new_ch_names)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.info.set_montage(montage)
    # 裁剪数据到1.0秒到601.0秒
    raw.crop(tmin=1.0, tmax=601.0)
    print(raw.info['meas_date'])
    print(raw.get_data().shape)
    # print('-----END : 1.加载数据-----')
    return raw


def apply_filter_ref(raw, l_freq, h_freq):
    raw_filtered = raw.notch_filter(freqs=50, fir_design='firwin', verbose=False)
    raw_filtered = raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg', filter_length='auto',
                                       fir_design='firwin', verbose=False)
    raw_filtered.set_eeg_reference(ref_channels=['A1', 'A2'], projection=False, verbose=False)
    raw_filtered.drop_channels(['A1', 'A2'])

    return raw_filtered


def create_epochs(raw, duration):
    # 固定时间间隔划分
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, verbose=False, preload=True)
    # print('-----END : 3.创建Epochs-----')
    return epochs


def detect_bad_channels(epochs):
    ransac = autoreject.Ransac(n_jobs=1, verbose=True, picks='eeg')
    epochs_clean = ransac.fit_transform(epochs)
    bad_chs = ransac.bad_chs_  # 获取坏导列表
    return epochs_clean, bad_chs


def detect_bad_epochs(epochs, repair=False):
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=random_state,
                               verbose=False)
    epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)
    if not repair:
        # fig = reject_log.plot('horizontal', show=False)
        # if save:
        #     fig_path = os.path.join(Preprocess_info_path, '_autoreject_before_ica.png')
        #     fig.savefig(fig_path)
        # print('-----END : 5.坏epochs【检测】-----')
        return reject_log
    else:
        # fig = reject_log.plot('horizontal', show=False)
        # if save:
        #     fig_path = os.path.join(Preprocess_info_path, '_autoreject_after_ica.png')
        #     fig.savefig(fig_path)
        # print('-----END : 8.坏epochs检测并去除【检测】-----')
        return epochs_ar, reject_log



