import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dataset import EEGData
from Ablation.EB_Fusion import EB_Fusion
from Ablation.EB import EB
from Ablation.FB_Fusion import FB_Fusion
from Ablation.FB import FB
from Ablation.Only_RC import Only_RC
from Ablation.Only_SAPL import Only_SAPL
from Ablation.No_Prune import No_Prune

def create_model(model_name):
    if model_name == 'EB_Fusion':
        return EB_Fusion(num_channels=17, num_frequencies=49).to(device)
    if model_name == 'EB':
        return EB(num_channels=17, num_frequencies=49).to(device)
    if model_name == 'FB_Fusion':
        return FB_Fusion(num_channels=17, num_frequencies=49).to(device)
    if model_name == 'FB':
        return FB(num_channels=17, num_frequencies=49).to(device)
    if model_name == 'Only_RC':
        return Only_RC(num_channels=17, num_frequencies=49).to(device)
    if model_name == 'Only_SAPL':
        return Only_SAPL(num_channels=17, num_frequencies=49).to(device)
    if model_name == 'No_Prune':
        return No_Prune(num_channels=17, num_frequencies=49).to(device)
    else:
        return None
# 如果有 GPU 可用，则使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train_kf(X, Y, model_name):
    kf = KFold(n_splits=n_splits, shuffle=True)
    best_metrics_per_fold = []
    fold = 1
    for train_idx, val_idx in kf.split(X, Y):
        print(f"Fold {fold}")

        train_x, train_y = X[train_idx], Y[train_idx]
        train_num = train_x.shape[0]

        val_x, val_y = X[val_idx], Y[val_idx]
        val_num = val_x.shape[0]

        train_subset = EEGData(train_x, train_y)
        val_subset = EEGData(val_x, val_y)

        print('训练集: ', train_num, ' 验证集: ', val_num)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=500, shuffle=False)
        # 初始化模型、优化器等
        model = create_model(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
                outputs = model(inputs)
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
                    outputs = model(inputs)
                    # outputs = model(inputs)
                    classification_loss = criterion(outputs, labels)
                    val_loss += classification_loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # 计算平均准确率、精确率、召回率和F1分数
            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

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
        best_metrics_per_fold.append(best_metrics)
        fold += 1
        # break

    # 计算 accuracy 的均值和标准差
    print(best_metrics_per_fold)
    avg_accuracy = sum([m['accuracy'] for m in best_metrics_per_fold]) / n_splits
    std_accuracy = math.sqrt(sum([(m['accuracy'] - avg_accuracy) ** 2 for m in best_metrics_per_fold]) / n_splits)
    # 计算 precision 的均值和标准差
    avg_precision = sum([m['precision'] for m in best_metrics_per_fold]) / n_splits
    std_precision = math.sqrt(sum([(m['precision'] - avg_precision) ** 2 for m in best_metrics_per_fold]) / n_splits)
    # 计算 recall 的均值和标准差
    avg_recall = sum([m['recall'] for m in best_metrics_per_fold]) / n_splits
    std_recall = math.sqrt(sum([(m['recall'] - avg_recall) ** 2 for m in best_metrics_per_fold]) / n_splits)
    # 计算 f1 的均值和标准差
    avg_f1 = sum([m['f1'] for m in best_metrics_per_fold]) / n_splits
    std_f1 = math.sqrt(sum([(m['f1'] - avg_f1) ** 2 for m in best_metrics_per_fold]) / n_splits)
    # 打印输出均值 ± 标准差
    print(f"ACC: {avg_accuracy:.4f} ± {std_accuracy:.4f}, "
          f"PRE: {avg_precision:.4f} ± {std_precision:.4f}, "
          f"REC: {avg_recall:.4f} ± {std_recall:.4f}, "
          f"F1: {avg_f1:.4f} ± {std_f1:.4f}")


"""深度学习的代码"""

batch_size = 64
learning_rate = 0.001
num_epochs = 100
n_splits = 23
n_fft = 512


X = np.load(f'../EEG/welch_{n_fft}_rel_all_psd.npy')
# X = np.load(f'../EEG/eeg_raw.npy')
Y = np.load(f'../EEG/label_0.42.npy').ravel()  # 中位数阈值0.42
count_ones = np.count_nonzero(Y)
print('label 1 rate', count_ones / Y.shape[0])
print('数据集规模  ', X.shape, Y.shape)
N_sample = X.shape[0]
num_channels = X.shape[1]
num_frequencies = X.shape[2]

standard_type = 'Subject_standard'
if standard_type == 'All_standard':
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
if standard_type == 'Subject_standard':
    X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(N_sample, num_channels, num_frequencies)
"""消融实验"""
for name in ['EB_Fusion', 'EB',
             'FB_Fusion', 'FB',
             'Only_RC', 'Only_SAPL', 'No_Prune']:
    print(f'==========Model Name {name}==========')
    train_kf(X, Y, name)

