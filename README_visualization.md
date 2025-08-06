# CSI实验结果可视化工具

这是一个专门为CSI WiFi活动识别实验结果设计的可视化工具，支持从CSV文件生成各种可视化图像。

## 功能特性

### 1. 单个实验可视化
- 训练准确率曲线
- 损失曲线
- 精确率、召回率、F1分数曲线
- 统计摘要报告

### 2. 多实验对比
- 验证准确率对比图
- 支持自定义实验标签
- 自动颜色区分

### 3. 训练趋势分析
- 自动分析训练趋势
- 过拟合检测
- 训练稳定性评估

## 使用方法

### 命令行模式

#### 1. 可视化单个实验结果
```bash
# 基本使用
python visualize_results.py single runs/20231201_143022/metrics.csv

# 指定输出目录
python visualize_results.py single runs/20231201_143022/metrics.csv --output_dir ./plots

# 自定义类别名称
python visualize_results.py single metrics.csv --class_names wave beckon push pull sitdown getdown

# 只保存不显示
python visualize_results.py single metrics.csv --no_show
```

#### 2. 对比多个实验结果
```bash
# 基本对比
python visualize_results.py compare exp1.csv exp2.csv exp3.csv

# 自定义标签
python visualize_results.py compare exp1.csv exp2.csv --labels KAN_CA_CSI CA_CSI

# 指定输出目录
python visualize_results.py compare exp1.csv exp2.csv --output_dir ./comparison
```

#### 3. 分析训练趋势
```bash
python visualize_results.py trend runs/20231201_143022/metrics.csv
```

#### 4. 显示使用示例
```bash
python visualize_results.py example
```

### 交互模式

直接运行脚本进入交互模式：
```bash
python visualize_results.py
```

然后按照提示选择操作：
1. 可视化单个实验结果
2. 对比多个实验结果
3. 分析训练趋势
4. 显示使用示例
5. 退出

### 编程接口

#### 1. 单个实验可视化
```python
from visualization_data import visualize_from_csv

# 基本使用
visualize_from_csv('runs/20231201_143022/metrics.csv')

# 自定义参数
visualize_from_csv(
    csv_file_path='metrics.csv',
    output_dir='./plots',
    class_names=['wave', 'beckon', 'push', 'pull', 'sitdown', 'getdown'],
    save_plots=True,
    show_plots=False
)
```

#### 2. 实验对比
```python
from visualization_data import create_comparison_plot

create_comparison_plot(
    csv_files=['exp1.csv', 'exp2.csv', 'exp3.csv'],
    labels=['KAN_CA_CSI', 'CA_CSI', 'THAT_CSI'],
    output_dir='./comparison'
)
```

#### 3. 趋势分析
```python
from visualization_data import analyze_training_trends

trends = analyze_training_trends('metrics.csv')
print(trends)
```

## 输出文件说明

### 单个实验可视化输出
- `training_accuracy.png`: 训练准确率曲线
- `metrics_curves.png`: 指标曲线（精确率、召回率、F1分数）
- `loss_curves.png`: 损失曲线
- `training_summary.txt`: 训练结果摘要

### 实验对比输出
- `experiment_comparison.png`: 验证准确率对比图

### 趋势分析输出
- `training_trends.txt`: 训练趋势分析报告

## CSV文件格式要求

CSV文件应包含以下列：
- `Epoch`: 训练轮数
- `Train Loss`: 训练损失
- `Train Acc`: 训练准确率
- `Val Loss`: 验证损失
- `Val Acc`: 验证准确率
- `Precision`: 精确率
- `Recall`: 召回率
- `F1`: F1分数

示例CSV格式：
```csv
Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Precision,Recall,F1
1,2.3456,0.1234,2.4567,0.1111,0.1234,0.1111,0.1170
2,2.1234,0.2345,2.2345,0.2222,0.2345,0.2222,0.2280
...
```

## 图像特性

### 1. 高质量输出
- 默认DPI: 300
- 矢量格式支持
- 自动布局优化

### 2. 智能标注
- 最佳性能点标注
- 趋势箭头指示
- 数值精确显示

### 3. 专业样式
- 网格背景
- 多色区分
- 清晰图例

## 故障排除

### 常见问题

1. **文件不存在**
   ```
   错误: 文件 metrics.csv 不存在
   ```
   解决：检查文件路径是否正确

2. **缺少必要列**
   ```
   警告: 缺少以下列: ['Epoch', 'Train Acc']
   ```
   解决：确保CSV文件包含所有必要列

3. **图像显示问题**
   - 在服务器环境中使用 `--no_show` 参数
   - 确保安装了matplotlib后端

4. **内存不足**
   - 减少同时处理的文件数量
   - 使用较小的图像尺寸

### 调试技巧

1. **检查CSV格式**
   ```python
   import pandas as pd
   df = pd.read_csv('metrics.csv')
   print(df.columns)
   print(df.head())
   ```

2. **验证数据完整性**
   ```python
   print(df.isnull().sum())
   print(df.describe())
   ```

3. **测试单个功能**
   ```python
   from visualization_data import plot_training_curves
   plot_training_curves(df, './test', True, True)
   ```

## 扩展功能

### 自定义样式
```python
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图像样式
plt.style.use('seaborn-v0_8')
```

### 添加新的可视化类型
```python
def custom_plot(df, output_dir):
    """自定义可视化函数"""
    # 您的自定义代码
    pass
```

## 版本历史

- v1.0: 基础可视化功能
- v1.1: 添加实验对比功能
- v1.2: 添加趋势分析功能
- v1.3: 添加交互模式和命令行接口 