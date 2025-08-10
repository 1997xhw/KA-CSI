import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from pathlib import Path
from numpy import unwrap


def detect_data_files(data_dir):
    """
    自动检测数据目录下的文件
    
    Args:
        data_dir (str): 数据目录路径
        
    Returns:
        dict: 包含检测到的文件路径
    """
    data_dir = Path(data_dir)
    
    # 支持的文件模式
    file_patterns = {
        'amp': ['data_amp_*.npy', 'our_data_amp_*.npy', '*_amp_*.npy'],
        'phase': ['data_phase_*.npy', 'our_data_phase_*.npy', '*_phase_*.npy'],
        'label': ['label_*.npy', 'our_data_label_*.npy', '*_label_*.npy']
    }
    
    detected_files = {}
    
    for data_type, patterns in file_patterns.items():
        for pattern in patterns:
            files = list(data_dir.glob(pattern))
            if files:
                detected_files[data_type] = str(files[0])
                print(f"检测到{data_type}文件: {files[0].name}")
                break
    
    return detected_files


def load_numpy_data(data_dir):
    """
    加载numpy格式的数据
    
    Args:
        data_dir (str): 数据目录路径
        
    Returns:
        tuple: (x_amp, x_phase, label)
    """
    data_dir = Path(data_dir)
    
    # 检测文件
    detected_files = detect_data_files(data_dir)
    
    if not detected_files:
        raise FileNotFoundError(f"在目录 {data_dir} 中未找到数据文件")
    
    # 加载数据
    if 'amp' in detected_files:
        x_amp = np.load(detected_files['amp'], allow_pickle=True)
        print(f"加载幅度数据: {x_amp.shape}")
    else:
        raise FileNotFoundError(f"未找到幅度数据文件")
    
    if 'phase' in detected_files:
        x_phase = np.load(detected_files['phase'], allow_pickle=True)
        # 相位解缠绕
        x_phase = np.unwrap(x_phase, axis=1)
        print(f"加载相位数据: {x_phase.shape}")
    else:
        raise FileNotFoundError(f"未找到相位数据文件")
    
    if 'label' in detected_files:
        label = np.load(detected_files['label'], allow_pickle=True)
        print(f"加载标签数据: {label.shape}")
        print(f"标签值范围: {label.min()} - {label.max()}")
        print(f"唯一标签值: {np.unique(label)}")
    else:
        raise FileNotFoundError(f"未找到标签数据文件")
    
    # 检查数据一致性
    if x_amp.shape[0] != x_phase.shape[0] or x_amp.shape[0] != label.shape[0]:
        raise ValueError(f"数据样本数量不一致: amp={x_amp.shape[0]}, phase={x_phase.shape[0]}, label={label.shape[0]}")
    
    return x_amp, x_phase, label


def get_class_names(dataset_type, root_dir):
    """
    根据数据集类型返回对应的动作类别名称
    
    Args:
        dataset_type (str): 数据集类型
        root_dir (str): 数据根目录
        
    Returns:
        list: 动作类别名称列表
    """
    # 检查是否为'our'数据集
    if 'our' in str(root_dir).lower() or 'our' in dataset_type.lower():
        return ["Stand up / Squat down", "Raise / Lower right hand", "Open / Close arms", "Kick right leg", "Kick left leg", "Unknown"]
    
    # 检查是否为'stan'数据集
    if 'stan' in dataset_type.lower():
        return ["fall", "run", "lie down", "walk", "sit down", "stand up"]
    
    # 检查是否为'wjq'数据集
    if 'wjq' in dataset_type.lower():
        return ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
    
    # 默认返回wjq的动作列表
    return ["wave", "beckon", "push", "pull", "sitdown", "getdown"]


def get_dataloader(dataset_type, root_dir, batch_size=32, shuffle=True, num_workers=1):
    """
    根据数据集类型和根目录返回 PyTorch DataLoader

    参数：
        dataset_type (str): 数据集名称，对应data目录下的文件夹名
        root_dir (str): 数据所在根路径
        batch_size (int)
        shuffle (bool)
        num_workers (int)

    返回：
        train_loader, val_loader, class_names
    """
    print(f"正在加载数据集: {dataset_type}")
    print(f"数据根目录: {root_dir}")
    
    if dataset_type.lower() == 'csi':
        # dataset = CSI_Dataset(root_dir)
        raise NotImplementedError("CSI数据集暂未实现")
    elif dataset_type.lower() == 'widar':
        # dataset = Widar_Dataset(root_dir)
        raise NotImplementedError("Widar数据集暂未实现")
    elif dataset_type.lower() == 'npy':
        # 使用默认的npy格式
        # data_dir = Path(root_dir)
        x_amp, x_phase, label = load_numpy_data(root_dir)
    else:
        # 尝试将dataset_type作为数据集名称
        data_dir = Path(root_dir) / dataset_type
        if not data_dir.exists():
            raise ValueError(f"数据集目录不存在: {data_dir}")
        
        x_amp, x_phase, label = load_numpy_data(data_dir)
    
    # 转换为torch tensor
    x_amp = torch.tensor(x_amp, dtype=torch.float32)
    x_phase = torch.tensor(x_phase, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)
    
    # 特殊处理'our'数据集，只取前90个天线
    if 'our' in str(root_dir).lower() or 'our' in dataset_type.lower():
        print("检测到'our'数据集，只取前90个天线")
        if len(x_amp.shape) == 3 and x_amp.shape[2] > 90:
            # 数据形状为 [samples, features, antennas]，取前90个天线
            x_amp = x_amp[:, :, :90]
            x_phase = x_phase[:, :, :90]
            print(f"已截取前90个天线数据")
        elif len(x_amp.shape) == 2 and x_amp.shape[1] > 90:
            # 数据形状为 [samples, features]，取前90列
            x_amp = x_amp[:, :90]
            x_phase = x_phase[:, :90]
            print(f"已截取前90列数据")
    
    print(f"数据形状:")
    print(f"  幅度: {x_amp.shape}")
    print(f"  相位: {x_phase.shape}")
    print(f"  标签: {label.shape}")
    
    # 创建数据集
    dataset = TensorDataset(x_amp, x_phase, label)
    
    # 划分训练集和验证集
    total_len = len(dataset)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len
    
    print(f"数据集划分: 训练集 {train_len}, 验证集 {val_len}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # 获取对应的动作类别名称
    class_names = get_class_names(dataset_type, root_dir)
    
    return train_loader, val_loader, class_names


def list_available_datasets(root_dir="./data"):
    """
    列出可用的数据集
    
    Args:
        root_dir (str): 数据根目录
        
    Returns:
        list: 可用数据集名称列表
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"数据根目录不存在: {root_dir}")
        return []
    
    datasets = []
    for item in root_path.iterdir():
        if item.is_dir():
            # 检查是否包含数据文件
            data_files = detect_data_files(item)
            if data_files:
                datasets.append(item.name)
                print(f"✓ {item.name}: 包含 {len(data_files)} 个数据文件")
            else:
                print(f"✗ {item.name}: 未找到数据文件")
    
    return datasets


if __name__ == "__main__":
    # 测试可用数据集
    print("可用的数据集:")
    available_datasets = list_available_datasets()
    print(f"\n总共找到 {len(available_datasets)} 个数据集")
    
    # 测试加载特定数据集
    if available_datasets:
        test_dataset = available_datasets[0]
        print(f"\n测试加载数据集: {test_dataset}")
        try:
            train_loader, val_loader, class_names = get_dataloader(test_dataset, "./data")
            print(f"✓ 成功加载数据集: {test_dataset}")
            print(f"检测到的动作类别: {class_names}")
        except Exception as e:
            print(f"✗ 加载数据集失败: {e}")
