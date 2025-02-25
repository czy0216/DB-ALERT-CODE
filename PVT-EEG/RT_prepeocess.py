import numpy as np
import pandas as pd
import os
import json
from scipy.io import savemat
import matplotlib.pyplot as plt
import math
"""读取原始文件，并生成rt.json"""
def create_rt_json():
    rt_json = {}
    rt_root_path = '../Raw_RT'
    rt_subject_dirs = os.listdir(rt_root_path)
    for rt_subject_dir in rt_subject_dirs:
        i_subject = int(rt_subject_dir.split('_')[0])
        rt_subject_path = os.path.join(rt_root_path, rt_subject_dir)
        rt_trial_dirs = [file for file in os.listdir(rt_subject_path) if file.endswith('.csv')]
        if len(rt_trial_dirs) < 5:
            print('被试数据丢失')
        for rt_trial_dir in rt_trial_dirs:
            _, i_trial = rt_trial_dir.split('.')[0].split('_')
            index = int(i_trial) + (i_subject - 1) * 5
            print(f'第{i_subject}被试第{int(i_trial)}试次编号{index}')
            csv_path = os.path.join(rt_subject_path, rt_trial_dir)
            df = pd.read_csv(csv_path)
            df = df[1:-1]
            # df['thisRow.t']记录该次反应测试起始时间，df['Feedback_text.stopped']记录该次反应测试结束时间
            # PVT任务开始时间
            pvt_start_time = df['thisRow.t'].iloc[0]
            # PVT任务结束时间
            pvt_end_time = df['Feedback_text.stopped'].iloc[-1]
            rt_record = []
            for i_row, row in df.iterrows():
                if not math.isnan(row['RTms']):
                    # if 100.0 < row['RTms'] < 2000.0: # 有效的Rt时间
                    rt_record.append((row['thisRow.t'] - pvt_start_time,
                                      row['Feedback_text.stopped'] - pvt_start_time,
                                      row['RTms']))
            rt_json[f'{index}'] = {"pvt_start_time": pvt_start_time,
                                   "pvt_end_time": pvt_end_time,
                                   "rt": rt_record}
    with open('../rt_all.json', 'w', encoding='utf-8') as json_file:
        json.dump(rt_json, json_file, indent=4)

if __name__ == '__main__':
    create_rt_json()
    """RT统计分析"""
    with open('../rt_all.json', 'r', encoding='utf-8') as json_file:
        rt_json = json.load(json_file)

