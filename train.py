
import torch.nn as nn

from models import TwoStreamModel
from grn import GatesResidualNetwork
from visualization_data import draw_confusion_matrix_3, draw_metrics_curves
from data_loader import get_dataloader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
import time
from tqdm import tqdm
import csv, os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 超参数配置
# hlayers = 5
# vlayers = 1
# hheads = 6
# vheads = 10
# K = 10
# sample = 2
# batch_size = 8
# # 根据数据集不同需要更改！
# maxlen = 2000
# embed_dim = 60
# # 根据数据集不同需要更改！
# num_class = 6
# EPOCHS = 50
# input_shape = (maxlen, embed_dim)


def parse_args():
    parser = argparse.ArgumentParser(description="CSI WiFi Activity Recognition Training")

    parser.add_argument('--hlayers', type=int, default=5, help='Horizontal transformer layers')
    parser.add_argument('--vlayers', type=int, default=1, help='Vertical transformer layers')
    parser.add_argument('--hheads', type=int, default=6, help='Horizontal attention heads')
    parser.add_argument('--vheads', type=int, default=10, help='Vertical attention heads')
    parser.add_argument('--K', type=int, default=10, help='Number of Gaussian kernels')
    parser.add_argument('--sample', type=int, default=2, help='Sampling rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--maxlen', type=int, default=2000, help='Max sequence length')
    parser.add_argument('--embed_dim', type=int, default=60, help='Input embedding dimension')
    parser.add_argument('--num_class', type=int, default=6, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')

    return parser.parse_args()

# wjq
## 原版that的模型
class THAT_CSI_Model(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim):
        super().__init__()
        self.stream1 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim)
        # self.stream2 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen, embed_dim)
        # self.grn = GatesResidualNetwork(256)
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x_amp, x_phase):
        feat1 = self.stream1(x_amp)
        # feat2 = self.stream2(x_phase)
        # fused = self.grn(feat1, feat2)
        # logits = self.classifier(feat1)
        return feat1
# wjq
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

def train_eval(train_loader, val_loader, model, EPOCHS, device, class_names=None):


    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []
    precision_list, recall_list, f1_list = [], [], []

    metrics_file = 'runs/metrics.csv'
    write_header = not os.path.exists(metrics_file)

    with open(metrics_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Precision', 'Recall', 'F1'])

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x1, x2, y in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        train_acc = total_correct / total_samples
        avg_train_loss = total_loss / len(train_loader)
        train_accuracies.append(train_acc)
        train_losses.append(avg_train_loss)
        print(f"[Train] Accuracy: {train_acc:.4f}, Loss: {avg_train_loss:.4f}")

        # 验证
        model.eval()
        correct, total = 0, 0
        total_val_loss = 0.0
        all_preds, all_labels = [], []
        time_start = time.time()

        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader, desc=f"[Val] Epoch {epoch + 1}"):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                loss = criterion(logits, y)
                preds = torch.argmax(logits, dim=1)

                total_val_loss += loss.item()
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_acc = correct / total
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracies.append(val_acc)
        val_losses.append(avg_val_loss)

        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        with open(metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc, precision, recall, f1])

        print(f"[Val] Acc: {val_acc:.4f}, Loss: {avg_val_loss:.4f}, Best: {best_val_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Time: {time.time() - time_start:.2f}s\n{'-' * 50}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch == EPOCHS - 1:
            draw_confusion_matrix_3(all_labels, all_preds,class_names)

            draw_metrics_curves(EPOCHS, train_accuracies, val_accuracies,
                                precision_list, recall_list, f1_list)

    return train_accuracies, val_accuracies, best_val_acc


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    args = parse_args()
    input_shape = (args.maxlen, args.embed_dim)

    train_loader, val_loader = get_dataloader(
        dataset_type='npy',
        root_dir=r'./data/',
        batch_size=args.batch_size
    )

    # wjq 标准模型
    # class_names=['No movement', 'Falling', 'Sitdown/Standup', 'Walking', 'Turning', 'Picking'])
    # model = CA_CSI_Model(
    #     args.hlayers, args.vlayers, args.hheads, args.vheads, args.K,
    #     args.sample, args.num_class, args.maxlen, args.embed_dim
    # ).to(device)

    class_names = ['wave', 'beckon', 'push', 'pull', 'sitdown', 'getdown']
    model = THAT_CSI_Model(
        args.hlayers, args.vlayers, args.hheads, args.vheads, args.K,
        args.sample, args.num_class, args.maxlen, args.embed_dim
    ).to(device)

    print("Model Structure:\n", model)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    with open("runs/model_structure.txt", "w") as f:
        f.write(str(model))
        f.write(f"\nTotal parameters: {total:,}\n")
        f.write(f"Trainable parametes: {trainable:,}\n")



    train_eval(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        EPOCHS=args.epochs,
        device=device, class_names=class_names,)
