#!/usr/bin/env python3
"""
我的实验运行脚本
专门为您的数据设计的实验配置
"""

import json
import os
from pathlib import Path
from train import ExperimentConfig, ExperimentManager
from data_loader import list_available_datasets, get_class_names


def create_my_experiment_configs():
    """创建针对您数据的实验配置"""
    
    # 获取可用数据集
    available_datasets = list_available_datasets("./data")
    print(f"发现可用数据集: {available_datasets}")
    
    configs = {}
    
    # 为每个数据集创建实验配置
    for dataset_name in available_datasets:
        # 获取该数据集对应的动作类别
        class_names = get_class_names(dataset_name, "./data")
        num_class = len(class_names)
        
        # 基础实验 - KAN_CA_CSI模型
        configs[f"baseline_kan_{dataset_name}"] = {
            "model": {
                "hlayers": 5, "vlayers": 1, "hheads": 5, "vheads": 10,
                "K": 10, "sample": 2, "maxlen": 1000, "embed_dim": 90, "num_class": num_class
            },
            "training": {
                "batch_size": 12, "epochs": 50, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": dataset_name, "root_dir": "./data",
                "class_names": class_names
            },
            "experiment": {
                "model_type": "KAN_CA_CSI", "save_dir": "runs"
            }
        }
        
        # 对比实验 - CA_CSI模型
        configs[f"baseline_ca_{dataset_name}"] = {
            "model": {
                "hlayers": 5, "vlayers": 1, "hheads": 6, "vheads": 10,
                "K": 10, "sample": 2, "maxlen": 1000, "embed_dim": 90, "num_class": num_class
            },
            "training": {
                "batch_size": 8, "epochs": 2, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": dataset_name, "root_dir": "./data",
                "class_names": class_names
            },
            "experiment": {
                "model_type": "CA_CSI", "save_dir": "runs"
            }
        }
        
        # 对比实验 - THAT_CSI模型
        configs[f"baseline_that_{dataset_name}"] = {
            "model": {
                "hlayers": 5, "vlayers": 1, "hheads": 6, "vheads": 10,
                "K": 10, "sample": 2, "maxlen": 1000, "embed_dim": 90, "num_class": num_class
            },
            "training": {
                "batch_size": 8, "epochs": 2, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": dataset_name, "root_dir": "./data",
                "class_names": class_names
            },
            "experiment": {
                "model_type": "THAT_CSI", "save_dir": "runs"
            }
        }
    
    # 添加一些特殊实验配置
    if "our" in available_datasets:
        # 获取our数据集的类别
        our_class_names = get_class_names("our", "./data")
        our_num_class = len(our_class_names)
        
        # 针对our数据集的特殊配置
        configs["large_model_our"] = {
            "model": {
                "hlayers": 8, "vlayers": 2, "hheads": 8, "vheads": 12,
                "K": 15, "sample": 2, "maxlen": 1000, "embed_dim": 90, "num_class": our_num_class
            },
            "training": {
                "batch_size": 4, "epochs": 2, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": "our", "root_dir": "./data",
                "class_names": our_class_names
            },
            "experiment": {
                "model_type": "KAN_CA_CSI", "save_dir": "runs"
            }
        }
        
        configs["small_model_our"] = {
            "model": {
                "hlayers": 3, "vlayers": 1, "hheads": 4, "vheads": 8,
                "K": 8, "sample": 2, "maxlen": 1000, "embed_dim": 90, "num_class": our_num_class
            },
            "training": {
                "batch_size": 16, "epochs": 2, "learning_rate": 1e-4
            },
            "data": {
                "dataset_type": "our", "root_dir": "./data",
                "class_names": our_class_names
            },
            "experiment": {
                "model_type": "KAN_CA_CSI", "save_dir": "runs"
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
    configs = create_my_experiment_configs()
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
            print(f"{exp_name:30s}: {best_acc:.4f}")
        else:
            print(f"{exp_name:30s}: 失败")
    
    # 保存结果
    results_file = "my_experiment_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_file}")


def run_specific_experiment(exp_name):
    """运行指定的实验"""
    configs = create_my_experiment_configs()
    
    if exp_name not in configs:
        print(f"错误: 未找到实验配置 '{exp_name}'")
        print(f"可用的实验配置: {list(configs.keys())}")
        return
    
    config_dict = configs[exp_name]
    run_single_experiment(exp_name, config_dict)


def run_dataset_experiments(dataset_name):
    """运行特定数据集的所有实验"""
    configs = create_my_experiment_configs()
    results = {}
    
    # 筛选出该数据集的实验
    dataset_configs = {k: v for k, v in configs.items() if dataset_name in k}
    
    if not dataset_configs:
        print(f"错误: 未找到数据集 '{dataset_name}' 的实验配置")
        return
    
    print(f"开始运行数据集 '{dataset_name}' 的实验...")
    print(f"找到 {len(dataset_configs)} 个实验配置")
    
    for exp_name, config_dict in dataset_configs.items():
        best_acc = run_single_experiment(exp_name, config_dict)
        results[exp_name] = best_acc
    
    # 打印结果汇总
    print(f"\n{'='*60}")
    print(f"数据集 '{dataset_name}' 实验结果汇总:")
    print(f"{'='*60}")
    
    for exp_name, best_acc in results.items():
        if best_acc is not None:
            print(f"{exp_name:30s}: {best_acc:.4f}")
        else:
            print(f"{exp_name:30s}: 失败")
    
    return results


def run_dataset_model_experiment(dataset_name, model_type):
    """运行特定数据集的特定模型实验"""
    configs = create_my_experiment_configs()
    
    # 构建实验名称
    exp_name = f"baseline_{model_type.lower()}_{dataset_name}"
    
    if exp_name not in configs:
        print(f"错误: 未找到实验配置 '{exp_name}'")
        print(f"可用的实验配置:")
        dataset_configs = {k: v for k, v in configs.items() if dataset_name in k}
        for name in dataset_configs.keys():
            print(f"  - {name}")
        return
    
    print(f"开始运行数据集 '{dataset_name}' 的 '{model_type}' 模型实验...")
    config_dict = configs[exp_name]
    best_acc = run_single_experiment(exp_name, config_dict)
    
    return best_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行我的CSI实验")
    parser.add_argument('--exp_name', type=str, help='指定要运行的实验名称')
    parser.add_argument('--all', action='store_true', help='运行所有实验')
    parser.add_argument('--dataset', type=str, help='运行特定数据集的所有实验')
    parser.add_argument('--model', type=str, help='指定模型类型 (kan, ca, that)')
    parser.add_argument('--list', action='store_true', help='列出所有可用的实验配置')
    
    args = parser.parse_args()
    
    if args.list:
        configs = create_my_experiment_configs()
        print("可用的实验配置:")
        for exp_name in configs.keys():
            print(f"  - {exp_name}")
    elif args.all:
        run_all_experiments()
    elif args.dataset and args.model:
        run_dataset_model_experiment(args.dataset, args.model)
    elif args.dataset:
        run_dataset_experiments(args.dataset)
    elif args.exp_name:
        run_specific_experiment(args.exp_name)
    else:
        print("请指定参数:")
        print("  --exp_name: 运行特定实验")
        print("  --dataset: 运行特定数据集的所有实验")
        print("  --dataset + --model: 运行特定数据集的特定模型实验")
        print("  --all: 运行所有实验")
        print("  --list: 列出所有实验配置")
        
        # 显示可用数据集
        available_datasets = list_available_datasets()
        if available_datasets:
            print(f"\n可用数据集: {available_datasets}")
            print("示例:")
            print("  python run_my_experiments.py --dataset our")
            print("  python run_my_experiments.py --dataset our --model kan")
            print("  python run_my_experiments.py --dataset our --model ca") 