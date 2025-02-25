import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from dataset import EEGData
from model import EEG_Alert

import json

"""深度学习的代码"""
# 设置随机种子
# seed = 10
# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)

# 如果有 GPU 可用，则使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 加载数据
"""psd_method = ['welch','raw']"""
psd_method = 'welch'
n_fft = 512
if psd_method in ['welch']:
    X = np.load(f'../EEG/welch_{n_fft}_feature.npy')  # 1: 相对PSD + 0: 绝对PSD
    X = X[:, :, :, 1] # 相对PSD
    # X = np.load(f'../EEG/welch_{n_fft}_psd.npy')
else:
    X = np.load(f'../EEG/eeg_raw.npy')  # raw data
Y = np.load(f'../EEG/label_2.npy').ravel()
print('数据集规模  ', X.shape, Y.shape)
N_sample = X.shape[0]
num_channels = X.shape[1]
num_frequencies = X.shape[2]

"""全体标准化处理"""
X = X.reshape(X.shape[0], -1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = X.reshape(X.shape[0], num_channels, num_frequencies)
"""逐个个体分别标准化"""

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 100

def train_kf():
    N_subject = 23
    best_metrics_per_fold = []
    for i_subject in range(N_subject):
        print(f"Fold {i_subject}")
        val_idx = np.arange(i_subject * 885, (i_subject + 1) * 885)
        all_idx = np.arange(N_sample)
        train_idx = np.setdiff1d(all_idx, val_idx)

        train_x, train_y = X[train_idx], Y[train_idx]
        train_num = train_x.shape[0]

        val_x, val_y = X[val_idx], Y[val_idx]
        val_num = val_x.shape[0]

        train_subset = EEGData(train_x, train_y)
        val_subset = EEGData(val_x, val_y)

        print('训练集: ', train_num, ' 验证集: ', val_num)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=885, shuffle=False)
        # 初始化模型、优化器等
        model = EEG_Alert(num_channels=num_channels, num_frequencies=num_frequencies).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
        train_losses = []
        val_losses = []
        best_val_f1 = 0
        best_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'attn_weights': []}
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs, _, _ = model(inputs)
                # outputs = model(inputs)
                classification_loss = criterion(outputs, labels)
                classification_loss.backward()
                optimizer.step()
                train_loss += classification_loss.item() * inputs.size(0)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _, _ = model(inputs)
                    # outputs = model(inputs)
                    classification_loss = criterion(outputs, labels)
                    val_loss += classification_loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # 计算宏平均准确率、精确率、召回率和F1分数
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

            # 保存每折中最佳的宏平均f1分数及其对应的指标
            if val_f1 > best_val_f1:
                print('save best!')
                best_val_f1 = val_f1
                best_metrics = {
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1,
                }
                torch.save(model.state_dict(), f'best_model_{i_subject}.pth')
        best_metrics_per_fold.append(best_metrics)
        # break

    avg_accuracy = sum([m['accuracy'] for m in best_metrics_per_fold]) / N_subject
    avg_precision = sum([m['precision'] for m in best_metrics_per_fold]) / N_subject
    avg_recall = sum([m['recall'] for m in best_metrics_per_fold]) / N_subject
    avg_f1 = sum([m['f1'] for m in best_metrics_per_fold]) / N_subject
    print(f"Average Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, "
          f"Recall: {avg_recall:.4f}, F1-Measure: {avg_f1:.4f}")


"""可视化自注意力分数"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def vis_result():
    freqs = np.load(f'../EEG/welch_freqs.npy')
    channel_names = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1',
                     'Pz', 'P2', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
    for i in range(17):
        # 加载预训练模型
        model = EEG_Alert(num_channels=num_channels, num_frequencies=num_frequencies).to(device)
        model.load_state_dict(torch.load(f'temp_model_{i + 1}.pth', weights_only=True))
        model.eval()

        # 加载数据集
        dataset = EEGData(X, Y)
        loader = DataLoader(dataset, batch_size=885, shuffle=True)
        all_labels = []
        all_preds = []
        for j, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, chan_freq_results, freq_chan_results = model(inputs)
            _, predicted = torch.max(outputs, 1)
            labels = labels.cpu().numpy()
            predicted = predicted.cpu().numpy()
            accuracy = accuracy_score(labels, predicted)
            all_labels.extend(labels)
            all_preds.extend(predicted)
            print(f'{j} subject: ', accuracy)
            # 可视化注意力权重、掩蔽向量和注意力分数
            visualize_results(chan_freq_results, np.array(channel_names), 'chan_freq')
            visualize_results(freq_chan_results, np.array(freqs), 'freq_chan')
            # break # 只看一个batch

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        # Display results
        print(f"Fold {i + 1}:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1-Measure: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        break


def visualize_results(result_dict, node_name, branch):
    attn_weights = result_dict['attn_weights']
    scores = result_dict['score']
    if len(attn_weights) == len(scores):
        print(f'{len(attn_weights)} layers PA-Block')
    n = len(attn_weights)
    for i in range(1):  # 只分析第一层
        attn_weight = attn_weights[i].detach().numpy()
        np.save('attn_weight.npy', attn_weight)
        # 创建热力图
        plt.figure(figsize=(8, 8))
        plt.imshow(attn_weight, cmap='viridis', interpolation='nearest')  # 使用 'viridis' 颜色映射

        # 设置横纵坐标标签
        plt.xticks(ticks=np.arange(len(node_name)), labels=node_name, rotation=90)  # 横坐标标签旋转90度
        plt.yticks(ticks=np.arange(len(node_name)), labels=node_name)

        # 添加颜色条
        plt.colorbar(label='Attention Weight')
        # 显示热力图
        plt.title('Attention Weight Heatmap')
        plt.show()
        score = scores[i].detach().numpy()
        sorted_indices = np.argsort(score)[::-1]
        print(score[sorted_indices])
        print(node_name[sorted_indices])


train_kf()

