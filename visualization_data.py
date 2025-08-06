import matplotlib.pyplot as plt 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

def draw_acc(epochs,train,val):
    xpoints = np.arange(0, epochs,dtype=int)
    train = np.array(train)
    val = np.array(val)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(xpoints, train)
    plt.plot(xpoints, val)
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

# def draw_loss(history):
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'val'], loc='upper right')
#     plt.show()
def draw_loss(train_loss, val_loss):
    import matplotlib.pyplot as plt
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

def draw_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    # Plot the confusion matrix
    target_names=['No movement', 'Falling', 'Sitting down/standing up', 'Walking', 'Turning', 'Picking up']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def draw_confusion_matrix_2(y_true, y_pred):
    # Define the mapping from numeric labels to string labels
    label_mapping = {
        0: 'No movement',
        1: 'Falling',
        2: 'Sitting down / Standing up',
        3: 'Walking',
        4: 'Turning',
        5: 'Picking up a pen'
    }

    # Convert the numeric labels to string labels for display purposes
    labels = [label_mapping[i] for i in sorted(label_mapping.keys())]

    # Generate confusion matrix with numeric values
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_mapping.keys()))

    # Convert confusion matrix to percentage format (row-wise)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Set up the figure and axes
    plt.figure(figsize=(8, 6))

    # Plot the heatmap with string labels
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)

    # Add axis labels and title
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Reference', fontsize=12)
    plt.title('Confusion Matrix of MultiEnv LOS (Office)', fontsize=14)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()


def draw_confusion_matrix_3(y_true, y_pred, class_names=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if class_names is None:
        class_names = ['No movement', 'Falling', 'Sitting down / Standing up',
                       'Walking', 'Turning', 'Picking up a pen']

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction')
    plt.ylabel('Reference')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def draw_metrics_curves(epochs, train_acc, val_acc, precision, recall, f1):
    import matplotlib.pyplot as plt
    x = list(range(1, epochs + 1))

    plt.figure(figsize=(12, 8))

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(x, train_acc, label='Train Acc')
    plt.plot(x, val_acc, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision
    plt.subplot(2, 2, 2)
    plt.plot(x, precision, label='Precision', color='orange')
    plt.title('Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    # Recall
    plt.subplot(2, 2, 3)
    plt.plot(x, recall, label='Recall', color='green')
    plt.title('Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    # F1-score
    plt.subplot(2, 2, 4)
    plt.plot(x, f1, label='F1-score', color='red')
    plt.title('F1-score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_from_csv(csv_file_path, output_dir=None, class_names=None, save_plots=True, show_plots=True):
    """
    从CSV文件读取训练结果并生成可视化图像
    
    Args:
        csv_file_path (str): CSV文件路径
        output_dir (str): 输出目录，如果为None则使用CSV文件所在目录
        class_names (list): 类别名称列表
        save_plots (bool): 是否保存图像
        show_plots (bool): 是否显示图像
    """
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        return
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取CSV文件: {csv_file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取CSV文件失败: {str(e)}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(csv_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置默认类别名称
    if class_names is None:
        class_names = ['wave', 'beckon', 'push', 'pull', 'sitdown', 'getdown']
    
    # 检查必要的列是否存在
    required_columns = ['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Precision', 'Recall', 'F1']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: 缺少以下列: {missing_columns}")
        print(f"可用的列: {list(df.columns)}")
        return
    
    # 1. 绘制训练曲线
    plot_training_curves(df, output_dir, save_plots, show_plots)
    
    # 2. 绘制指标曲线
    plot_metrics_curves(df, output_dir, save_plots, show_plots)
    
    # 3. 绘制损失曲线
    plot_loss_curves(df, output_dir, save_plots, show_plots)
    
    # 4. 生成统计摘要
    generate_summary_stats(df, output_dir)
    
    print(f"可视化完成! 结果保存在: {output_dir}")


def plot_training_curves(df, output_dir, save_plots, show_plots):
    """绘制训练准确率曲线"""
    plt.figure(figsize=(10, 6))
    
    epochs = df['Epoch'].values
    train_acc = df['Train Acc'].values
    val_acc = df['Val Acc'].values
    
    plt.plot(epochs, train_acc, label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加最佳性能标注
    best_val_acc = val_acc.max()
    best_epoch = epochs[val_acc.argmax()]
    plt.annotate(f'Best Val Acc: {best_val_acc:.4f}\nEpoch: {best_epoch}', 
                xy=(best_epoch, best_val_acc), xytext=(best_epoch+2, best_val_acc-0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'training_accuracy.png'), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def plot_metrics_curves(df, output_dir, save_plots, show_plots):
    """绘制指标曲线"""
    plt.figure(figsize=(15, 10))
    
    epochs = df['Epoch'].values
    precision = df['Precision'].values
    recall = df['Recall'].values
    f1 = df['F1'].values
    
    # 创建2x2子图
    metrics = [('Precision', precision, 'orange'), 
               ('Recall', recall, 'green'), 
               ('F1-Score', f1, 'red')]
    
    for i, (metric_name, metric_values, color) in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, metric_values, color=color, linewidth=2, marker='o', markersize=4)
        plt.title(f'{metric_name} over Epochs', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(metric_name, fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 添加最佳值标注
        best_value = metric_values.max()
        best_epoch = epochs[metric_values.argmax()]
        plt.annotate(f'Best: {best_value:.4f}\nEpoch: {best_epoch}', 
                    xy=(best_epoch, best_value), xytext=(best_epoch+2, best_value-0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    # 添加综合指标图
    plt.subplot(2, 2, 4)
    plt.plot(epochs, precision, label='Precision', color='orange', linewidth=2)
    plt.plot(epochs, recall, label='Recall', color='green', linewidth=2)
    plt.plot(epochs, f1, label='F1-Score', color='red', linewidth=2)
    plt.title('All Metrics Comparison', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'metrics_curves.png'), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def plot_loss_curves(df, output_dir, save_plots, show_plots):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    
    epochs = df['Epoch'].values
    train_loss = df['Train Loss'].values
    val_loss = df['Val Loss'].values
    
    plt.plot(epochs, train_loss, label='Train Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加最佳损失标注
    best_val_loss = val_loss.min()
    best_epoch = epochs[val_loss.argmin()]
    plt.annotate(f'Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}', 
                xy=(best_epoch, best_val_loss), xytext=(best_epoch+2, best_val_loss+0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()


def generate_summary_stats(df, output_dir):
    """生成统计摘要"""
    summary = {
        '总训练轮数': len(df),
        '最佳验证准确率': df['Val Acc'].max(),
        '最佳验证准确率轮数': df.loc[df['Val Acc'].idxmax(), 'Epoch'],
        '最终验证准确率': df['Val Acc'].iloc[-1],
        '最佳精确率': df['Precision'].max(),
        '最佳召回率': df['Recall'].max(),
        '最佳F1分数': df['F1'].max(),
        '最佳验证损失': df['Val Loss'].min(),
        '最终训练准确率': df['Train Acc'].iloc[-1],
        '最终训练损失': df['Train Loss'].iloc[-1]
    }
    
    # 保存摘要到文件
    summary_file = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("训练结果摘要\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    # 打印摘要
    print("\n训练结果摘要:")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return summary


def create_comparison_plot(csv_files, labels=None, output_dir=None, save_plot=True, show_plot=True):
    """
    创建多个实验结果的对比图
    
    Args:
        csv_files (list): CSV文件路径列表
        labels (list): 实验标签列表
        output_dir (str): 输出目录
        save_plot (bool): 是否保存图像
        show_plot (bool): 是否显示图像
    """
    if labels is None:
        labels = [f"Experiment {i+1}" for i in range(len(csv_files))]
    
    if len(csv_files) != len(labels):
        print("错误: CSV文件数量与标签数量不匹配")
        return
    
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        if not os.path.exists(csv_file):
            print(f"警告: 文件 {csv_file} 不存在，跳过")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            epochs = df['Epoch'].values
            val_acc = df['Val Acc'].values
            
            color = colors[i % len(colors)]
            plt.plot(epochs, val_acc, label=label, color=color, linewidth=2, marker='o', markersize=4)
            
        except Exception as e:
            print(f"读取文件 {csv_file} 失败: {str(e)}")
            continue
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Comparison of Different Experiments', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if save_plot:
            plt.savefig(os.path.join(output_dir, 'experiment_comparison.png'), dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    plt.close()


def analyze_training_trends(csv_file_path, output_dir=None):
    """
    分析训练趋势
    
    Args:
        csv_file_path (str): CSV文件路径
        output_dir (str): 输出目录
    """
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        return
    
    df = pd.read_csv(csv_file_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算趋势
    trends = {
        '训练准确率趋势': 'increasing' if df['Train Acc'].iloc[-1] > df['Train Acc'].iloc[0] else 'decreasing',
        '验证准确率趋势': 'increasing' if df['Val Acc'].iloc[-1] > df['Val Acc'].iloc[0] else 'decreasing',
        '训练损失趋势': 'decreasing' if df['Train Loss'].iloc[-1] < df['Train Loss'].iloc[0] else 'increasing',
        '验证损失趋势': 'decreasing' if df['Val Loss'].iloc[-1] < df['Val Loss'].iloc[0] else 'increasing',
        '过拟合程度': 'high' if (df['Train Acc'].iloc[-1] - df['Val Acc'].iloc[-1]) > 0.1 else 'low'
    }
    
    # 保存趋势分析
    trend_file = os.path.join(output_dir, 'training_trends.txt')
    with open(trend_file, 'w', encoding='utf-8') as f:
        f.write("训练趋势分析\n")
        f.write("=" * 50 + "\n\n")
        for key, value in trends.items():
            f.write(f"{key}: {value}\n")
    
    print("\n训练趋势分析:")
    print("=" * 50)
    for key, value in trends.items():
        print(f"{key}: {value}")
    
    return trends


# 使用示例函数
def example_usage():
    """使用示例"""
    print("可视化工具使用示例:")
    print("1. 从CSV文件生成可视化:")
    print("   visualize_from_csv('runs/20231201_143022/metrics.csv')")
    print()
    print("2. 创建实验对比图:")
    print("   create_comparison_plot(['exp1.csv', 'exp2.csv'], ['KAN_CA_CSI', 'CA_CSI'])")
    print()
    print("3. 分析训练趋势:")
    print("   analyze_training_trends('runs/20231201_143022/metrics.csv')")


if __name__ == "__main__":
    # 示例使用
    example_usage()
