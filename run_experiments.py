#!/usr/bin/env python3
"""
实验运行脚本
用于快速运行不同的实验配置
"""

import json
import os
from pathlib import Path
from train import ExperimentConfig, ExperimentManager


def create_experiment_configs():
    """创建不同的实验配置"""
    configs = {
        # 基础配置 - KAN_CA_CSI模型
        "baseline_kan": {
            "model": {
                "hlayers": 5, "vlayers": 1, "hheads": 6, "vheads": 10,
                "K": 10, "sample": 2, "maxlen": 2000, "embed_dim": 60, "num_class": 6
            },
            "training": {
                "batch_size": 4, "epochs": 50, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": "npy", "root_dir": "./data/",
                "class_names": ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
            },
            "experiment": {
                "model_type": "KAN_CA_CSI", "save_dir": "runs"
            }
        },
        
        # 对比实验 - CA_CSI模型
        "baseline_ca": {
            "model": {
                "hlayers": 5, "vlayers": 1, "hheads": 6, "vheads": 10,
                "K": 10, "sample": 2, "maxlen": 2000, "embed_dim": 60, "num_class": 6
            },
            "training": {
                "batch_size": 4, "epochs": 50, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": "npy", "root_dir": "./data/",
                "class_names": ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
            },
            "experiment": {
                "model_type": "CA_CSI", "save_dir": "runs"
            }
        },
        
        # 对比实验 - THAT_CSI模型
        "baseline_that": {
            "model": {
                "hlayers": 5, "vlayers": 1, "hheads": 6, "vheads": 10,
                "K": 10, "sample": 2, "maxlen": 2000, "embed_dim": 60, "num_class": 6
            },
            "training": {
                "batch_size": 4, "epochs": 50, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": "npy", "root_dir": "./data/",
                "class_names": ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
            },
            "experiment": {
                "model_type": "THAT_CSI", "save_dir": "runs"
            }
        },
        
        # 超参数调优实验 - 更大的模型
        "large_model": {
            "model": {
                "hlayers": 8, "vlayers": 2, "hheads": 8, "vheads": 12,
                "K": 15, "sample": 2, "maxlen": 2000, "embed_dim": 60, "num_class": 6
            },
            "training": {
                "batch_size": 2, "epochs": 50, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": "npy", "root_dir": "./data/",
                "class_names": ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
            },
            "experiment": {
                "model_type": "KAN_CA_CSI", "save_dir": "runs"
            }
        },
        
        # 超参数调优实验 - 更小的模型
        "small_model": {
            "model": {
                "hlayers": 3, "vlayers": 1, "hheads": 4, "vheads": 8,
                "K": 8, "sample": 2, "maxlen": 2000, "embed_dim": 60, "num_class": 6
            },
            "training": {
                "batch_size": 8, "epochs": 50, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": "npy", "root_dir": "./data/",
                "class_names": ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
            },
            "experiment": {
                "model_type": "KAN_CA_CSI", "save_dir": "runs"
            }
        },
        
        # 学习率实验
        "lr_experiment": {
            "model": {
                "hlayers": 5, "vlayers": 1, "hheads": 6, "vheads": 10,
                "K": 10, "sample": 2, "maxlen": 2000, "embed_dim": 60, "num_class": 6
            },
            "training": {
                "batch_size": 4, "epochs": 50, "learning_rate": 5e-5
            },
            "data": {
                "dataset_type": "npy", "root_dir": "./data/",
                "class_names": ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
            },
            "experiment": {
                "model_type": "KAN_CA_CSI", "save_dir": "runs"
            }
        }
    }
    return configs


def run_single_experiment(exp_name, config_dict):
    """运行单个实验"""
    print(f"\n{'='*50}")
    print(f"开始运行实验: {exp_name}")
    print(f"{'='*50}")
    
    try:
        config = ExperimentConfig(config_dict)
        experiment_manager = ExperimentManager(config)
        history, best_acc = experiment_manager.run_experiment()
        
        print(f"\n实验 {exp_name} 完成!")
        print(f"最佳验证准确率: {best_acc:.4f}")
        
        return best_acc
        
    except Exception as e:
        print(f"实验 {exp_name} 失败: {str(e)}")
        return None


def run_all_experiments():
    """运行所有实验"""
    configs = create_experiment_configs()
    results = {}
    
    print("开始批量实验...")
    print(f"总共 {len(configs)} 个实验配置")
    
    for exp_name, config_dict in configs.items():
        best_acc = run_single_experiment(exp_name, config_dict)
        results[exp_name] = best_acc
    
    # 打印结果汇总
    print(f"\n{'='*60}")
    print("实验结果汇总:")
    print(f"{'='*60}")
    
    for exp_name, best_acc in results.items():
        if best_acc is not None:
            print(f"{exp_name:20s}: {best_acc:.4f}")
        else:
            print(f"{exp_name:20s}: 失败")
    
    # 保存结果
    results_file = "experiment_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")


def run_specific_experiment(exp_name):
    """运行指定的实验"""
    configs = create_experiment_configs()
    
    if exp_name not in configs:
        print(f"错误: 未找到实验配置 '{exp_name}'")
        print(f"可用的实验配置: {list(configs.keys())}")
        return
    
    config_dict = configs[exp_name]
    run_single_experiment(exp_name, config_dict)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行CSI实验")
    parser.add_argument('--exp_name', type=str, help='指定要运行的实验名称')
    parser.add_argument('--all', action='store_true', help='运行所有实验')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_experiments()
    elif args.exp_name:
        run_specific_experiment(args.exp_name)
    else:
        print("请指定 --exp_name 运行特定实验，或使用 --all 运行所有实验")
        print("可用的实验配置:")
        configs = create_experiment_configs()
        for exp_name in configs.keys():
            print(f"  - {exp_name}") 