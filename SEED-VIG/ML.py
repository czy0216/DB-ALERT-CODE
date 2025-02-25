import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
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
    X = np.load(f'../EEG/welch_{n_fft}_rel_all_psd.npy')  # (N, 17, 49)
    Y = np.load(f'../EEG/label_0.42.npy').ravel()  # (N,)
    N_sample = X.shape[0]
    X = X.reshape(N_sample, -1)  # Flattening feature dimensions

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
    N_subject = 23
    N_sample = X.shape[0]  # 假定X是全体样本数据

    for i_subject in range(N_subject):
        print(f"Fold {i_subject + 1}")
        val_idx = np.arange(i_subject * 885, (i_subject + 1) * 885)
        all_idx = np.arange(N_sample)
        train_idx = np.setdiff1d(all_idx, val_idx)

        # 提取训练和验证数据
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        for name, model in models.items():
            print(f"Processing {name}...")

            # 训练模型
            model.fit(X_train, Y_train)
            # 预测
            Y_pred = model.predict(X_val)

            # 计算评价指标
            cm = confusion_matrix(Y_val, Y_pred)
            acc = accuracy_score(Y_val, Y_pred)
            pre = precision_score(Y_val, Y_pred, average='weighted', zero_division=0)
            rec = recall_score(Y_val, Y_pred, average='weighted', zero_division=0)
            f1 = f1_score(Y_val, Y_pred, average='weighted', zero_division=0)

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


# 直接读取
# 读取 JSON 文件
with open('model_results.json', 'r') as f:
    loaded_results = json.load(f)
# 输出所有模型的平均性能指标
for name, metrics in loaded_results.items():
    acc_list = [m['Accuracy'] for m in metrics]
    pre_list = [m['Precision'] for m in metrics]
    rec_list = [m['Recall'] for m in metrics]
    f1_list = [m['F1 Score'] for m in metrics]
    plt.figure(figsize=(10, 6))
    plt.plot(acc_list, label='Accuracy', marker='o')
    plt.plot(pre_list, label='Precision', marker='o')
    plt.plot(rec_list, label='Recall', marker='o')
    plt.plot(f1_list, label='F1 Score', marker='o')

    plt.title('Performance Metrics for Each Model')
    plt.ylabel('Metric Value')
    plt.xlabel('Subject Number')
    plt.xticks(ticks=range(len(acc_list)), labels=[f'{i + 1}' for i in range(len(acc_list))])
    plt.legend()
    plt.grid(True)
    plt.show()
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



