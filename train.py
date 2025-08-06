import torch.nn as nn
import datetime
import os
import json
from pathlib import Path

from models import TwoStreamModel
from grn import GatesResidualNetwork
from visualization_data import draw_confusion_matrix_3, draw_metrics_curves
from data_loader import get_dataloader
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import time
from tqdm import tqdm
import csv
import argparse
from kan_grn import KAN_GatesResidualNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentConfig:
    """实验配置管理类"""
    
    def __init__(self, config_dict=None):
        self.config = {
            'model': {
                'hlayers': 5, 'vlayers': 1, 'hheads': 6, 'vheads': 10,
                'K': 10, 'sample': 2, 'maxlen': 2000, 'embed_dim': 60, 'num_class': 6
            },
            'training': {
                'batch_size': 4, 'epochs': 50, 'learning_rate': 1e-4
            },
            'data': {
                'dataset_type': 'npy', 'root_dir': './data/',
                'class_names': ['wave', 'beckon', 'push', 'pull', 'sitdown', 'getdown']
            },
            'experiment': {
                'model_type': 'KAN_CA_CSI', 'save_dir': 'runs'
            }
        }
        
        if config_dict:
            self._update_config(config_dict)
    
    def _update_config(self, config_dict):
        for key, value in config_dict.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get_model_config(self):
        return self.config['model']
    
    def get_training_config(self):
        return self.config['training']
    
    def get_data_config(self):
        return self.config['data']
    
    def get_experiment_config(self):
        return self.config['experiment']
    
    def save_config(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_type, config, device):
        model_config = config.get_model_config()
        
        if model_type == 'THAT_CSI':
            return THAT_CSI_Model(**model_config).to(device)
        elif model_type == 'CA_CSI':
            return CA_CSI_Model(**model_config).to(device)
        elif model_type == 'KAN_CA_CSI':
            return KAN_CA_CSI_Model(**model_config).to(device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


class Trainer:
    """训练器类"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.train_config = config.get_training_config()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.train_config['learning_rate']
        )
        
        self.history = {
            'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [],
            'precision': [], 'recall': [], 'f1': []
        }
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for x1, x2, y in tqdm(train_loader, desc="Training"):
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(x1, x2)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
        
        return total_loss / len(train_loader), total_correct / total_samples
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader, desc="Validating"):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)
                preds = torch.argmax(logits, dim=1)
                
                total_loss += loss.item()
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_acc = total_correct / total_samples
        
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return val_loss, val_acc, precision, recall, f1, all_labels, all_preds
    
    def train(self, train_loader, val_loader, class_names=None, metrics_file=None):
        epochs = self.train_config['epochs']
        
        if metrics_file is None:
            metrics_file = 'runs/metrics.csv'
        write_header = not os.path.exists(metrics_file)
        
        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Precision', 'Recall', 'F1'])
        
        for epoch in range(epochs):
            print(f"\n{'='*20} Epoch {epoch+1}/{epochs} {'='*20}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, precision, recall, f1, all_labels, all_preds = self.validate_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['precision'].append(precision)
            self.history['recall'].append(recall)
            self.history['f1'].append(f1)
            
            with open(metrics_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, precision, recall, f1])
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"新的最佳验证准确率: {val_acc:.4f}")
            
            if epoch == epochs - 1 and class_names:
                draw_confusion_matrix_3(all_labels, all_preds, class_names)
                draw_metrics_curves(epochs, self.history['train_acc'], self.history['val_acc'],
                                  self.history['precision'], self.history['recall'], self.history['f1'])
        
        return self.history


class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, config):
        self.config = config
        self.experiment_config = config.get_experiment_config()
        self.data_config = config.get_data_config()
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(self.experiment_config['save_dir']) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        config.save_config(self.run_dir / 'config.json')
    
    def setup_data(self):
        train_loader, val_loader = get_dataloader(
            dataset_type=self.data_config['dataset_type'],
            root_dir=self.data_config['root_dir'],
            batch_size=self.config.get_training_config()['batch_size']
        )
        return train_loader, val_loader
    
    def setup_model(self):
        model_type = self.experiment_config['model_type']
        model = ModelFactory.create_model(model_type, self.config, device)
        
        total, trainable = self.count_parameters(model)
        print(f"模型类型: {model_type}")
        print(f"总参数数量: {total:,}")
        print(f"可训练参数数量: {trainable:,}")
        
        model_structure_file = self.run_dir / 'model_structure.txt'
        with open(model_structure_file, "w", encoding='utf-8') as f:
            f.write(str(model))
            f.write(f"\n总参数数量: {total:,}\n")
            f.write(f"可训练参数数量: {trainable:,}\n")
        
        return model
    
    @staticmethod
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def run_experiment(self):
        print(f"开始实验，结果保存在: {self.run_dir}")
        
        train_loader, val_loader = self.setup_data()
        model = self.setup_model()
        
        trainer = Trainer(model, self.config, device)
        
        metrics_file = self.run_dir / 'metrics.csv'
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            class_names=self.data_config['class_names'],
            metrics_file=str(metrics_file)
        )
        
        print(f"\n实验完成！最佳验证准确率: {trainer.best_val_acc:.4f}")
        print(f"结果保存在: {self.run_dir}")
        
        return history, trainer.best_val_acc


# 模型定义
class THAT_CSI_Model(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim):
        super().__init__()
        self.stream1 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim)
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x_amp, x_phase):
        feat1 = self.stream1(x_amp)
        return feat1


class CA_CSI_Model(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim):
        super().__init__()
        self.stream1 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim)
        self.stream2 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim)
        self.grn = GatesResidualNetwork(256)
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x_amp, x_phase):
        feat1 = self.stream1(x_amp)
        feat2 = self.stream2(x_phase)
        fused = self.grn(feat1, feat2)
        logits = self.classifier(fused)
        return logits


class KAN_CA_CSI_Model(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim):
        super().__init__()
        self.stream1 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim)
        self.stream2 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim)
        self.grn = KAN_GatesResidualNetwork(256)
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x_amp, x_phase):
        feat1 = self.stream1.extract_feature(x_amp)
        feat2 = self.stream2.extract_feature(x_phase)
        fused = self.grn(feat1, feat2)
        logits = self.classifier(fused)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(description="CSI WiFi Activity Recognition Training")
    
    # 模型参数
    parser.add_argument('--hlayers', type=int, default=5, help='Horizontal transformer layers')
    parser.add_argument('--vlayers', type=int, default=1, help='Vertical transformer layers')
    parser.add_argument('--hheads', type=int, default=6, help='Horizontal attention heads')
    parser.add_argument('--vheads', type=int, default=10, help='Vertical attention heads')
    parser.add_argument('--K', type=int, default=10, help='Number of Gaussian kernels')
    parser.add_argument('--sample', type=int, default=2, help='Sampling rate')
    parser.add_argument('--maxlen', type=int, default=2000, help='Max sequence length')
    parser.add_argument('--embed_dim', type=int, default=60, help='Input embedding dimension')
    parser.add_argument('--num_class', type=int, default=6, help='Number of classes')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    # 实验参数
    parser.add_argument('--model_type', type=str, default='KAN_CA_CSI', 
                       choices=['THAT_CSI', 'CA_CSI', 'KAN_CA_CSI'], help='Model type')
    parser.add_argument('--config_file', type=str, help='Configuration file path')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.config_file and os.path.exists(args.config_file):
        config = ExperimentConfig.load_config(args.config_file)
    else:
        config_dict = {
            'model': {
                'hlayers': args.hlayers, 'vlayers': args.vlayers, 'hheads': args.hheads, 'vheads': args.vheads,
                'K': args.K, 'sample': args.sample, 'maxlen': args.maxlen, 'embed_dim': args.embed_dim, 'num_class': args.num_class
            },
            'training': {
                'batch_size': args.batch_size, 'epochs': args.epochs, 'learning_rate': args.learning_rate
            },
            'experiment': {
                'model_type': args.model_type
            },
            'data': {
                'dataset_type': 'npy', 'root_dir': './data/',
                'class_names': ['wave', 'beckon', 'push', 'pull', 'sitdown', 'getdown']
            },
        }
        config = ExperimentConfig(config_dict)

    print(config)
    experiment_manager = ExperimentManager(config)
    history, best_acc = experiment_manager.run_experiment()
    
    return history, best_acc


if __name__ == '__main__':
    main()
