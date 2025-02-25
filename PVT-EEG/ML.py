import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import openpyxl
def train():
    # Data preparation
    n_fft = 512

    # X = np.load(f'../EEG/welch_{n_fft}_rel_all_psd.npy')  # (N, 17, 49)
    # X = np.load(f'../EEG/welch_{n_fft}_psd.npy')  # PSD值（为窗口）
    X = np.load(f'../EEG/welch_{n_fft}_rel_psd.npy')  # 通道内相对
    # rel_all_psd = np.load(f'../EEG/welch_{n_fft}_rel_all_psd.npy')  # 全脑相对
    # abs_psd = np.load(f'../EEG/welch_{n_fft}_abs_psd.npy')  # 绝对

    Y = np.load(f'../Label/label_2.npy').ravel()  # (N,)
    count_ones = np.count_nonzero(Y)
    print('label 1 rate', count_ones / Y.shape[0])
    print('数据集规模  ', X.shape, Y.shape)
    N_sample = X.shape[0]
    # 全体数据标准化
    X = X.reshape(N_sample, -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Model setup
    models = {
        "1-SVM": SVC(max_iter=-1),
        "2-KNN": KNeighborsClassifier(),
        "3-Random Forest": RandomForestClassifier(),
        "4-Decision Tree": DecisionTreeClassifier(),
        "5-Naive Bayes": GaussianNB()
    }
    # 训练和测试
    results = {}
    kf = KFold(n_splits=17, shuffle=False)
    fold = 1
    for train_idx, val_idx in kf.split(X, Y):
        print(f"Fold {fold}")
        fold += 1

        train_x, train_y = X[train_idx], Y[train_idx]
        val_x, val_y = X[val_idx], Y[val_idx]

        for name, model in models.items():
            print(f"Processing {name}...")

            # 训练模型
            model.fit(train_x, train_y)
            # 预测
            Y_pred = model.predict(val_x)

            # 计算评价指标
            cm = confusion_matrix(val_y, Y_pred)
            acc = accuracy_score(val_y, Y_pred)
            pre = precision_score(val_y, Y_pred, average='binary', zero_division=0)
            rec = recall_score(val_y, Y_pred, average='binary', zero_division=0)
            f1 = f1_score(val_y, Y_pred, average='binary', zero_division=0)

            # 将结果保存在results字典中
            if name not in results:
                results[name] = []
            results[name].append({
                'Confusion Matrix': cm,
                'Accuracy': acc,
                'Precision': pre,
                'Recall': rec,
                'F1 Score': f1
            })
            print(cm, acc, pre, rec, f1)

    # 输出所有模型的平均性能指标
    for name, metrics in results.items():
        avg_acc = np.mean([m['Accuracy'] for m in metrics])
        avg_pre = np.mean([m['Precision'] for m in metrics])
        avg_rec = np.mean([m['Recall'] for m in metrics])
        avg_f1 = np.mean([m['F1 Score'] for m in metrics])
        print(
            f"{name} - Avg Accuracy: {avg_acc:.4f}, Avg Precision: {avg_pre:.4f}, Avg Recall: {avg_rec:.4f}, Avg F1 Score: {avg_f1:.4f}")

    # 在输出之前，转换results中的Numpy数组为普通列表
    for model_name, model_results in results.items():
        for result in model_results:
            result['Confusion Matrix'] = result['Confusion Matrix'].tolist()  # 将Numpy数组转换为列表

    # 保存为JSON文件
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Results have been saved to 'model_results.json'")

train()

with open('model_results.json', 'r') as f:
    loaded_results = json.load(f)
# 输出所有模型的平均性能指标
for name, metrics in loaded_results.items():
    acc_list = [m['Accuracy'] for m in metrics]
    pre_list = [m['Precision'] for m in metrics]
    rec_list = [m['Recall'] for m in metrics]
    f1_list = [m['F1 Score'] for m in metrics]
    # plt.figure(figsize=(10, 6))
    # plt.plot(acc_list, label='Accuracy', marker='o')
    # plt.plot(pre_list, label='Precision', marker='o')
    # plt.plot(rec_list, label='Recall', marker='o')
    # plt.plot(f1_list, label='F1 Score', marker='o')
    #
    # plt.title('Performance Metrics for Each Model')
    # plt.ylabel('Metric Value')
    # plt.xlabel('Subject Number')
    # plt.xticks(ticks=range(len(acc_list)), labels=[f'{i + 1}' for i in range(len(acc_list))])
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)

    avg_pre = np.mean(pre_list)
    std_pre = np.std(pre_list)

    avg_rec = np.mean(rec_list)
    std_rec = np.std(rec_list)

    avg_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    print(f'{name} - Avg Accuracy: {avg_acc:.4f} ± {std_acc:.4f}')
    print(f'{name} - Avg Precision: {avg_pre:.4f} ± {std_pre:.4f}')
    print(f'{name} - Avg Recall: {avg_rec:.4f} ± {std_rec:.4f}')
    print(f'{name} - Avg F1 Score: {avg_f1:.4f} ± {std_f1:.4f}')
    print('------------------------------------------------------')



