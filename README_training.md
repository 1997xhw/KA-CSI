# CSI WiFi 活动识别训练系统

这是一个重构后的CSI WiFi活动识别训练系统，提供了规范的实验管理和配置系统。

## 系统架构

### 核心组件

1. **ExperimentConfig**: 配置管理类
   - 管理模型参数、训练参数、数据参数和实验参数
   - 支持从JSON文件加载配置
   - 自动保存实验配置

2. **ModelFactory**: 模型工厂类
   - 根据配置创建不同类型的模型
   - 支持THAT_CSI、CA_CSI、KAN_CA_CSI三种模型

3. **Trainer**: 训练器类
   - 封装训练和验证逻辑
   - 自动记录训练历史
   - 支持指标保存和可视化

4. **ExperimentManager**: 实验管理器
   - 管理完整的实验流程
   - 自动创建实验目录
   - 保存模型结构和配置

## 使用方法

### 1. 基本训练

```bash
# 使用默认配置训练
python train.py

# 使用自定义参数训练
python train.py --model_type KAN_CA_CSI --epochs 100 --batch_size 8 --learning_rate 1e-3

# 使用配置文件训练
python train.py --config_file my_config.json
```

### 2. 批量实验

```bash
# 运行所有预定义实验
python run_experiments.py --all

# 运行特定实验
python run_experiments.py --exp_name baseline_kan
```

### 3. 配置文件示例

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

## 预定义实验配置

### 基础实验
- `baseline_kan`: KAN_CA_CSI模型基础配置
- `baseline_ca`: CA_CSI模型基础配置  
- `baseline_that`: THAT_CSI模型基础配置

### 超参数调优实验
- `large_model`: 更大的模型配置
- `small_model`: 更小的模型配置
- `lr_experiment`: 不同学习率实验

## 输出文件

每次实验会在 `runs/时间戳/` 目录下生成：

- `config.json`: 实验配置
- `model_structure.txt`: 模型结构信息
- `metrics.csv`: 训练指标记录
- 混淆矩阵和指标曲线图

## 模型类型说明

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

## 参数说明

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

## 实验管理建议

1. **参数调优顺序**:
   - 先确定最佳模型类型
   - 调整学习率和批次大小
   - 优化模型架构参数
   - 最后进行超参数网格搜索

2. **实验记录**:
   - 每次实验都会自动保存配置和结果
   - 使用有意义的实验名称
   - 记录实验目的和预期结果

3. **结果分析**:
   - 查看 `metrics.csv` 了解训练过程
   - 分析混淆矩阵识别问题类别
   - 比较不同配置的性能

## 故障排除

### 常见问题

1. **内存不足**:
   - 减小 `batch_size`
   - 减小 `maxlen`
   - 使用更小的模型配置

2. **训练不收敛**:
   - 调整学习率
   - 增加训练轮数
   - 检查数据预处理

3. **过拟合**:
   - 增加正则化
   - 减小模型复杂度
   - 使用数据增强

### 调试技巧

- 使用小数据集快速验证配置
- 检查GPU内存使用情况
- 监控训练和验证损失曲线
- 保存中间检查点用于恢复训练 