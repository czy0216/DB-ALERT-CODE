import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import StandardScaler

n_fft = 512
X = np.load(f'../EEG/welch_{n_fft}_rel_all_psd.npy')  # 全脑相对
# X = np.load(f'../EEG/welch_{n_fft}_rel_psd.npy')  # 通道内相对
# X = np.load(f'../EEG/welch_{n_fft}_abs_psd.npy')  # 绝对
# Y = np.load(f'../EEG/label2_0.45.npy').ravel() # 中位数

N_sample = X.shape[0]
num_channels = X.shape[1]
num_frequencies = X.shape[2]

standard_type = 'All_standard'
if standard_type == 'Subject_standard':
    X_within_subject = []
    for i in range(23):
        start_index = i * 885
        end_index = (i + 1) * 885
        sub_psd = X[start_index:end_index]
        sub_psd = sub_psd.reshape(sub_psd.shape[0], -1)
        scaler = StandardScaler()
        sub_psd = scaler.fit_transform(sub_psd)
        sub_psd = sub_psd.reshape(sub_psd.shape[0], num_channels, num_frequencies)
        X_within_subject.append(sub_psd)
    X = np.concatenate(X_within_subject, axis=0)  # 个体内 标准化
if standard_type == 'All_standard':  # 全体标准化
    X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(N_sample, num_channels, num_frequencies)
"""
前额叶 pre-frontal (Fp) 
额叶 frontal lobe (F) 
颞叶 temporal lobe (T) 
顶叶 parietal lobe (P) 
枕叶 occipital lobe (O)
中央 central  (C)
"""
freqs = np.arange(1, 50, 1)
channel_names = np.array(['FT7', # 左侧 额叶颞叶之间
                          'FT8', # 右侧 额叶颞叶之间
                          'T7',  # 左颞叶
                          'T8',  # 右颞叶
                          'TP7', # 左侧 顶叶颞叶之间
                          'TP8', # 右侧 顶叶颞叶之间
                          'CP1', 'CP2', # 中央区顶叶之间
                          'P1', 'Pz', 'P2', # 顶叶
                          'PO3', 'POz', 'PO4', # 顶枕叶
                          'O1', 'Oz', 'O2'] # 枕叶
                         )
def calculate_p_values(X, Y):
    num_electrodes, num_points = X.shape[1], X.shape[2]
    p_values = np.zeros((num_electrodes, num_points))

    for i in range(num_electrodes):
        for j in range(num_points):
            group1 = X[Y == 0, i, j]
            group2 = X[Y == 1, i, j]

            f_stat, p_value = stats.f_oneway(group1, group2) # ANOVA

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

p_all = np.zeros((num_channels, num_frequencies))
count = 0
for t in np.arange(0.40, 0.50, 0.01):
    count += 1
    print(round(t, 2))
    Y = np.load(f'../EEG/label2_{round(t, 2)}.npy').ravel()  # 中位数
    p_values = calculate_p_values(X, Y)
    p_all += p_values
print('全体平均')
p_all = p_all / count
threshold = 0.001
row_proportions = np.mean(p_all < threshold, axis=1)  # 对应电极，比例越高，效果越显著
col_proportions = np.mean(p_all < threshold, axis=0)  # 对应频率
sorted_row_indices = np.argsort(-row_proportions)  # 从大到小排序，负号实现降序
sorted_col_indices = np.argsort(-col_proportions)  # 同上

sorted_channel_names = channel_names[sorted_row_indices]
sorted_freqs = freqs[sorted_col_indices]
print(sorted_channel_names)
print(sorted_freqs)


