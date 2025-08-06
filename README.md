# KA-CSI: CSI WiFi Activity Recognition

基于KAN (Kolmogorov-Arnold Networks) 和Transformer的CSI WiFi活动识别系统。

## 🚀 项目特性

- **多模型支持**: THAT_CSI、CA_CSI、KAN_CA_CSI三种架构
- **实验管理**: 完整的实验配置和批量运行系统
- **可视化工具**: 专业的训练结果分析和可视化
- **模块化设计**: 清晰的代码结构和易于扩展

## 📋 目录结构

```
KA-CSI/
├── train.py                 # 主训练脚本
├── run_experiments.py       # 批量实验运行
├── visualize_results.py     # 可视化工具
├── visualization_data.py    # 可视化核心功能
├── models.py               # 模型定义
├── data_loader.py          # 数据加载器
├── utils.py                # 工具函数
├── requirements.txt        # 项目依赖
├── setup.py               # 安装配置
├── config_example.json    # 配置文件示例
├── README_training.md     # 训练系统说明
├── README_visualization.md # 可视化工具说明
├── data/                  # 数据目录
├── runs/                  # 实验结果
└── test/                  # 测试文件
```

## 🛠️ 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/yourusername/KA-CSI.git
cd KA-CSI

# 安装依赖
pip install -r requirements.txt

# 或者使用开发模式安装
pip install -e .
```

### 2. 数据准备

将您的CSI数据放在 `data/` 目录下：
- `our_data_amp_1000_270_200.npy`: 幅度数据
- `our_data_phase_1000_270_200.npy`: 相位数据  
- `our_data_label_1000_270_200.npy`: 标签数据

### 3. 开始训练

```bash
# 使用默认配置训练
python train.py

# 使用自定义参数
python train.py --model_type KAN_CA_CSI --epochs 100 --batch_size 8

# 使用配置文件
python train.py --config_file my_config.json
```

### 4. 批量实验

```bash
# 运行所有预定义实验
python run_experiments.py --all

# 运行特定实验
python run_experiments.py --exp_name baseline_kan
```

### 5. 结果可视化

```bash
# 可视化单个实验结果
python visualize_results.py single runs/20231201_143022/metrics.csv

# 对比多个实验
python visualize_results.py compare exp1.csv exp2.csv --labels KAN_CA_CSI CA_CSI

# 交互模式
python visualize_results.py
```

## 📊 模型架构

### THAT_CSI_Model
- 单流架构，只处理幅度信息
- 适用于计算资源有限的情况

### CA_CSI_Model  
- 双流架构，处理幅度和相位信息
- 使用标准门控残差网络融合特征

### KAN_CA_CSI_Model
- 双流架构，处理幅度和相位信息
- 使用KAN门控残差网络融合特征
- 通常性能最佳

## 🔧 配置说明

### 模型参数
- `hlayers`: 水平Transformer层数
- `vlayers`: 垂直Transformer层数  
- `hheads`: 水平注意力头数
- `vheads`: 垂直注意力头数
- `K`: 高斯核数量
- `sample`: 采样率
- `maxlen`: 最大序列长度
- `embed_dim`: 嵌入维度
- `num_class`: 类别数量

### 训练参数
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `learning_rate`: 学习率

## 📈 实验结果

### 预定义实验配置
- `baseline_kan`: KAN_CA_CSI模型基础配置
- `baseline_ca`: CA_CSI模型基础配置  
- `baseline_that`: THAT_CSI模型基础配置
- `large_model`: 更大的模型配置
- `small_model`: 更小的模型配置
- `lr_experiment`: 不同学习率实验

### 输出文件
每次实验会在 `runs/时间戳/` 目录下生成：
- `config.json`: 实验配置
- `model_structure.txt`: 模型结构信息
- `metrics.csv`: 训练指标记录
- 混淆矩阵和指标曲线图

## 🎨 可视化功能

### 单个实验可视化
- 训练准确率曲线
- 损失曲线
- 精确率、召回率、F1分数曲线
- 统计摘要报告

### 多实验对比
- 验证准确率对比图
- 支持自定义实验标签
- 自动颜色区分

### 训练趋势分析
- 自动分析训练趋势
- 过拟合检测
- 训练稳定性评估

## 🧪 测试

```bash
# 运行测试
pytest test/

# 代码格式化
black .

# 代码检查
flake8 .
```

## 📝 使用示例

### 配置文件示例

创建 `my_config.json`:

```json
{
    "model": {
        "hlayers": 5,
        "vlayers": 1,
        "hheads": 6,
        "vheads": 10,
        "K": 10,
        "sample": 2,
        "maxlen": 2000,
        "embed_dim": 60,
        "num_class": 6
    },
    "training": {
        "batch_size": 4,
        "epochs": 50,
        "learning_rate": 1e-4
    },
    "data": {
        "dataset_type": "npy",
        "root_dir": "./data/",
        "class_names": ["wave", "beckon", "push", "pull", "sitdown", "getdown"]
    },
    "experiment": {
        "model_type": "KAN_CA_CSI",
        "save_dir": "runs"
    }
}
```

### 编程接口

```python
from train import ExperimentConfig, ExperimentManager
from visualization_data import visualize_from_csv

# 创建配置
config = ExperimentConfig({
    'model': {'hlayers': 5, 'vlayers': 1, ...},
    'training': {'batch_size': 4, 'epochs': 50, ...},
    'experiment': {'model_type': 'KAN_CA_CSI'}
})

# 运行实验
experiment_manager = ExperimentManager(config)
history, best_acc = experiment_manager.run_experiment()

# 可视化结果
visualize_from_csv('runs/20231201_143022/metrics.csv')
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [https://github.com/yourusername/KA-CSI](https://github.com/yourusername/KA-CSI)
- 问题反馈: [Issues](https://github.com/yourusername/KA-CSI/issues)

## 🙏 致谢

感谢所有为这个项目做出贡献的研究人员和开发者。

---

**注意**: 这是一个研究项目，主要用于学术研究目的。在生产环境中使用前请进行充分的测试和验证。 