import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models import TwoStreamModel
from grn import GatesResidualNetwork
from visualization_data import draw_confusion_matrix_2
from data_loader import get_dataloader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.utils.data as data_utils
import torch
import time
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
hlayers = 5
vlayers = 1
hheads = 9
vheads = 50
K = 10
sample = 2
batch_size = 4
maxlen = 1000
num_class = 6
EPOCHS = 50
embed_dim = 270
input_shape = (maxlen, embed_dim)

# wjq
class CA_CSI_Model(nn.Module):
    def __init__(self, hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen):
        super().__init__()
        self.stream1 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen)
        self.stream2 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen)
        self.grn = GatesResidualNetwork(256) # NOTE 后面改成别的kan
        self.classifier = nn.Linear(256, num_class)

    def forward(self, x_amp, x_phase):
        """
        Args:
            x_amp: Tensor [B, T, 270] - Amplitude stream input
            x_phase: Tensor [B, T, 270] - Phase stream input (after unwrapping)
        Returns:
            logits: Tensor [B, num_class]
        """
        feat1 = self.stream1(x_amp)     # From amplitude stream
        feat2 = self.stream2(x_phase)   # From phase stream
        fused = self.grn(feat1, feat2)  # Gating fusion
        logits = self.classifier(fused) # Final classification
        return logits

# def build_model():
#     model_1 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen)
#     model_2 = TwoStreamModel(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen)
#     grn = GatesResidualNetwork(units=256)
#     classifier = nn.Linear(256, num_class)
#     return model_1.to(device), model_2.to(device), grn.to(device), classifier.to(device)
#

# def train_eval(train_loader, val_loader):
#     model1, model2, grn, final_fc = build_model()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(
#         list(model1.parameters()) +
#         list(model2.parameters()) +
#         list(grn.parameters()) +
#         list(final_fc.parameters()),
#         lr=1e-4
#     )
#
#     best_val_acc = 0.0
#
#     for epoch in range(EPOCHS):
#         model1.train()
#         model2.train()
#         grn.train()
#         final_fc.train()
#
#         total_correct = 0
#         total_samples = 0
#
#         for x1, x2, y in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}"):
#             x1, x2, y = x1.to(device), x2.to(device), y.to(device)
#
#             optimizer.zero_grad()
#             out1 = model1(x1)
#             out2 = model2(x2)
#             fused = grn(out1, out2)
#             logits = final_fc(fused)
#
#             loss = criterion(logits, y)
#             loss.backward()
#             optimizer.step()
#
#             preds = torch.argmax(logits, dim=1)
#             total_correct += (preds == y).sum().item()
#             total_samples += y.size(0)
#
#         train_acc = total_correct / total_samples
#         print(f"[Train] Accuracy: {train_acc:.4f}")
#
#         # 验证
#         model1.eval()
#         model2.eval()
#         grn.eval()
#         final_fc.eval()
#
#         correct = 0
#         total = 0
#         all_preds, all_labels = [], []
#
#         with torch.no_grad():
#             for x1, x2, y in tqdm(val_loader, desc="[Val]"):
#                 x1, x2, y = x1.to(device), x2.to(device), y.to(device)
#
#                 out1 = model1(x1)
#                 out2 = model2(x2)
#                 fused = grn(out1, out2)
#                 logits = final_fc(fused)
#
#                 preds = torch.argmax(logits, dim=1)
#                 correct += (preds == y).sum().item()
#                 total += y.size(0)
#
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(y.cpu().numpy())
#
#         val_acc = correct / total
#         print(f"[Val] Accuracy: {val_acc:.4f}")
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             # torch.save(...)
#
#         if epoch == 20:
#             draw_confusion_matrix_2(all_labels, all_preds)
#
#         print(f"Precision: {precision_score(all_labels, all_preds, average='weighted'):.4f}")
#         print(f"Recall: {recall_score(all_labels, all_preds, average='weighted'):.4f}")
#         print(f"F1-score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
#         print('-' * 50)

# def train_eval(x_train_amp, x_train_phase, y_train, batch_size, model, EPOCHS, device):
def train_eval(train_loader, val_loader, model, EPOCHS, device):
    # # 数据划分
    # x_train_amp, x_val_amp, x_train_phase, x_val_phase, y_train, y_val = train_test_split(
    #     x_train_amp, x_train_phase, y_train, test_size=0.2, stratify=y_train, random_state=42
    # )
    # # 构建 DataLoader
    # train_dataset = data_utils.TensorDataset(
    #     torch.tensor(x_train_amp, dtype=torch.float32),
    #     torch.tensor(x_train_phase, dtype=torch.float32),
    #     torch.tensor(y_train, dtype=torch.long)
    # )
    # val_dataset = data_utils.TensorDataset(
    #     torch.tensor(x_val_amp, dtype=torch.float32),
    #     torch.tensor(x_val_phase, dtype=torch.float32),
    #     torch.tensor(y_val, dtype=torch.long)
    # )
    # train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    train_accuracies = []
    val_accuracies = []


    for epoch in range(EPOCHS):
        model.train()
        total_correct = 0
        total_samples = 0

        for x1, x2, y in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        train_acc = total_correct / total_samples
        train_accuracies.append(train_acc)
        print(f"[Train] Accuracy: {train_acc:.4f}")

        # 验证
        model.eval()
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        time_start = time.time()

        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader, desc=f"[Val] Epoch {epoch + 1}"):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"[Val] Accuracy: {val_acc:.4f} (Best: {best_val_acc:.4f})")
        print(f"Time: {time.time() - time_start:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), "best_model.pt")

        if epoch == 20:
            from visualization_data import draw_confusion_matrix_2
            draw_confusion_matrix_2(all_labels, all_preds)

        print(f"Precision: {precision_score(all_labels, all_preds, average='weighted'):.4f}")
        print(f"Recall: {recall_score(all_labels, all_preds, average='weighted'):.4f}")
        print(f"F1-score: {f1_score(all_labels, all_preds, average='weighted'):.4f}")
        print('-' * 50)
    return train_accuracies, val_accuracies, best_val_acc


if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(
        dataset_type='npy',
        root_dir=r'./data/',
        batch_size=batch_size
    )
    model = CA_CSI_Model(hlayers, vlayers, hheads, vheads, K, sample, num_class, maxlen).to(device)
    train_eval(    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    EPOCHS=50,
    device='cuda')
