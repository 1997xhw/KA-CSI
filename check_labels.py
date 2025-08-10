#!/usr/bin/env python3
"""
检查标签数据
"""

import numpy as np
import os

def check_labels():
    """检查our数据集的标签"""
    label_file = "./data/our/our_data_label_1000_270_150.npy"
    
    if os.path.exists(label_file):
        label = np.load(label_file, allow_pickle=True)
        print(f"标签文件: {label_file}")
        print(f"标签形状: {label.shape}")
        print(f"标签值范围: {label.min()} - {label.max()}")
        print(f"唯一标签值: {np.unique(label)}")
        print(f"标签值计数:")
        unique, counts = np.unique(label, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  类别 {val}: {count} 个样本")
    else:
        print(f"标签文件不存在: {label_file}")

if __name__ == "__main__":
    check_labels() 