import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# from dataset import CSI_Dataset, Widar_Dataset
from numpy import unwrap
def get_dataloader(dataset_type, root_dir, batch_size=32, shuffle=True, num_workers=1):
    """
    根据数据集类型和根目录返回 PyTorch DataLoader

    参数：
        dataset_type (str): 支持 ['CSI', 'Widar', 'npy']
        root_dir (str): 数据所在根路径
        batch_size (int)
        shuffle (bool)
        num_workers (int)

    返回：
        train_loader, val_loader
    """
    if dataset_type.lower() == 'csi':
        # dataset = CSI_Dataset(root_dir)
        pass
    elif dataset_type.lower() == 'widar':
        # dataset = Widar_Dataset(root_dir)
        pass
    elif dataset_type.lower() == 'npy':
        # mutil env
        # x_amp = np.load(f"{root_dir}/our/our_data_amp_1000_270_150-002.npy", allow_pickle=True)
        # x_phase = np.load(f"{root_dir}/our/our_data_phase_1000_270_150-001.npy", allow_pickle=True)
        # x_phase = np.unwrap(x_phase, axis=1)  # 相位解包，通常对时间轴解包
        # label = np.load(f"{root_dir}/our/our_data_label_1000_270_150.npy", allow_pickle=True)

        # x_amp = np.load(f"{root_dir}/stan/data_amp_2000.npy", allow_pickle=True)
        # x_phase = np.load(f"{root_dir}/stan/data_phase_2000.npy", allow_pickle=True)
        # x_phase = np.unwrap(x_phase, axis=1)  # 相位解包，通常对时间轴解包
        # label = np.load(f"{root_dir}/stan/label_2000.npy", allow_pickle=True)


        x_amp = np.load(f"data/wjq_two/data_amp_2000.npy", allow_pickle=True)
        x_phase = np.load(f"data/wjq_two/data_phase_2000.npy", allow_pickle=True)
        x_phase = np.unwrap(x_phase, axis=1)  # 相位解包，通常对时间轴解包
        label = np.load(f"data/wjq_two/label_2000.npy", allow_pickle=True)

        x_amp = torch.tensor(x_amp, dtype=torch.float32)
        x_phase = torch.tensor(x_phase, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        print(x_amp.shape)
        print(x_phase.shape)
        print(label.shape)
        dataset = TensorDataset(x_amp, x_phase, label)
    else:


        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    total_len = len(dataset)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
