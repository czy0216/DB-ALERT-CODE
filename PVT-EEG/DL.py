import json
import math
import os

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dataset import EEGData
from model import EEG_Alert
from com_model import DeepConvNet, ShallowConvNet, EEGNet
from com_model import ResNet, VisionTransformer

# 如果有 GPU 可用，则使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
"""可视化自注意力分数"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_pagerank(weights):
    # 由于networkx不支持批处理，我们将处理单个权重矩阵
    G = nx.DiGraph(weights)
    important_values = nx.pagerank(G, weight='weight', alpha=0.85)
    score = np.array([important_values[i] for i in range(len(important_values))])
    return score


def vis_result(model_name, f):
    EB_attn_map = []
    EB_mask = []
    FB_attn_map = []
    FB_mask = []
    freqs = np.arange(1, 50, 1)
    channel_names = np.array(['Fp1', 'Fp2', 'F3', 'F4', 'C3',
                              'C4', 'P3', 'P4', 'O1', 'O2',
                              'F7', 'F8', 'T3', 'T4', 'T5',
                              'T6', 'Fz', 'Cz', 'Pz'])
    for i in range(n_splits):
        # 加载预训练模型
        model = create_model(model_name, num_channels, num_frequencies)
        model.load_state_dict(torch.load(f'Control/best_model_{i + 1}.pth', weights_only=True))
        model.eval()
        dataset = EEGData(X, Y)
        loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=True)  # 一次加载全部数据集
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(loader):  # 只有一次循环
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, chan_freq_results, freq_chan_results = model(inputs)

                EB_attn_map.append(chan_freq_results['attn_weights'])
                EB_mask.append(chan_freq_results['mask'])
                FB_attn_map.append(freq_chan_results['attn_weights'])
                FB_mask.append(freq_chan_results['mask'])

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)
            # Display results
            print(f"Fold {i + 1}:")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1-Measure: {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)
        # 释放不再使用的变量
        del model
        del dataset
    print(len(EB_attn_map), len(EB_mask), len(FB_attn_map), len(FB_mask))
    """计算每一个模型每一层的重要性分数，然后相加取平均"""
    eb_score = {str(key): [] for key in channel_names}
    fb_score = {str(key): [] for key in freqs}
    for i in range(len(EB_attn_map)):
        eb_dict = branch_exp(EB_attn_map[i], EB_mask[i], channel_names)
        fb_dict = branch_exp(FB_attn_map[i], FB_mask[i], freqs)
        for key, value in eb_dict.items():
            eb_score[key].append(value)
        for key, value in fb_dict.items():
            fb_score[key].append(value)
    with open(f'eb_score_{f}.json', 'w') as json_file:
        json.dump(eb_score, json_file, indent=4)
    with open(f'fb_score_{f}.json', 'w') as json_file:
        json.dump(fb_score, json_file, indent=4)


def branch_exp(attn_maps, masks, index_name):
    index_dict = {str(key): 0 for key in index_name}
    num_layer = len(attn_maps)
    for i in range(num_layer):
        score = compute_pagerank(attn_maps[i])
        median_value = np.median(score)
        score[score <= median_value] = 0
        if len(index_name) != len(score):
            print('wrong')
        for key, value in zip(index_name, score):
            index_dict[str(key)] += value
        mask = np.array(masks[i])
        index_name = index_name[~mask]
    return index_dict  # 全部节点的分数


def create_model(model_name, num_channels, num_frequencies):
    if model_name == 'DeepConvNet':
        return DeepConvNet(num_channels=19, num_samples=2000, num_classes=2).to(device)
    if model_name == 'ShallowConvNet':
        return ShallowConvNet(num_channels=19, num_samples=2000, num_classes=2).to(device)
    if model_name == 'EEGNet':
        return EEGNet(num_channels=19, samples_rate=2000, num_classes=2).to(device)
    if model_name == 'ResNet':
        return ResNet().to(device)
    if model_name == 'VisionTransformer':
        return VisionTransformer().to(device)
    else:
        return EEG_Alert(num_channels=num_channels, num_frequencies=num_frequencies).to(device)


def cal_avg_results(X, Y, model_name, path_name):
    samples_per_subject = 300
    N_sample = X.shape[0]
    num_channels = X.shape[1]
    num_frequencies = X.shape[2]
    all_results = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }
    for i in range(n_splits):
        # 加载预训练模型
        model = create_model(model_name, num_channels, num_frequencies)  # 假设create_model()返回模型结构
        model.load_state_dict(torch.load(f'{path_name}/best_model_{i + 1}.pth', weights_only=True))
        model.eval()
        # 数据加载
        dataset = EEGData(X, Y)  # 假设EEGData是您定义的数据集类
        loader = DataLoader(dataset, batch_size=samples_per_subject, shuffle=False)
        # 存储真实标签和预测标签
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _ = model(inputs)  # 模型输出
                _, predicted = torch.max(outputs, 1)  # 获取预测类别
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                print(accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy()))
        # 计算性能指标
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        cm = confusion_matrix(all_labels, all_predictions)
        acc = accuracy_score(all_labels, all_predictions)
        pre = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        print(cm)
        # 保存每次模型的指标结果
        all_results['Accuracy'].append(acc)
        all_results['Precision'].append(pre)
        all_results['Recall'].append(rec)
        all_results['F1 Score'].append(f1)

    # 计算平均值和标准差
    for metric, values in all_results.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")


def train_kf(X, Y, model_name, path_name):
    print('数据集规模  ', X.shape, Y.shape)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    N_sample = X.shape[0]
    num_channels = X.shape[1]
    num_frequencies = X.shape[2]
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
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        # 初始化模型、优化器等
        model = create_model(model_name, num_channels, num_frequencies)
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
                outputs, chan_freq_results, freq_chan_results = model(inputs)
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
                    outputs, chan_freq_results, freq_chan_results = model(inputs)
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
                  f'Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f},Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
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
                torch.save(model.state_dict(), f'{path_name}/best_model_{fold}.pth')
        best_metrics_per_fold.append(best_metrics)
        fold += 1

    # 计算 accuracy 的均值和标准差
    print(best_metrics_per_fold)
    avg_accuracy = sum([m['accuracy'] for m in best_metrics_per_fold]) / n_splits
    std_accuracy = math.sqrt(sum([(m['accuracy'] - avg_accuracy) ** 2 for m in best_metrics_per_fold]) / n_splits)
    # 计算 precision 的均值和标准差
    avg_precision = sum([m['precision'] for m in best_metrics_per_fold]) / n_splits
    std_precision = math.sqrt(
        sum([(m['precision'] - avg_precision) ** 2 for m in best_metrics_per_fold]) / n_splits)
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



batch_size = 64
learning_rate = 0.001
num_epochs = 100
n_splits = 17
n_fft = 512
X = np.load(f'../EEG/welch_{n_fft}_rel_all_psd.npy')  # 全脑相对
# X = np.load('../EEG/30s_eeg.npy')
Y = np.load(f'../Label/label_2.npy').ravel()
count_ones = np.count_nonzero(Y)
print('label 1 rate', count_ones / Y.shape[0])
print('数据集规模  ', X.shape, Y.shape)

N_sample = X.shape[0]
num_channels = X.shape[1]
num_frequencies = X.shape[2]
# 'All_standard', 'Subject_standard'
standard_type = 'All_standard'
if standard_type == 'Subject_standard':
    X_within_subject = []
    for i in range(14):
        start_index = i * 100
        end_index = (i + 1) * 100
        sub_psd = X[start_index:end_index]
        sub_psd = sub_psd.reshape(sub_psd.shape[0], -1)
        scaler = StandardScaler()
        sub_psd = scaler.fit_transform(sub_psd)
        sub_psd = sub_psd.reshape(sub_psd.shape[0], num_channels, num_frequencies)
        X_within_subject.append(sub_psd)
    X = np.concatenate(X_within_subject, axis=0)  # 个体内 标准化
if standard_type == 'All_standard':
    X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(N_sample, num_channels, num_frequencies)

for name in [
    'DeepConvNet',
    'ShallowConvNet',
    'EEGNet']:
    print(f'==========Model Name {name}==========')
    train_kf(X, Y, name)

"""第三组对比实验"""
for name in [
    'VisionTransformer',
    'ResNet'
]:
    print(f'==========Model Name {name}==========')
    train_kf(X, Y, name)




