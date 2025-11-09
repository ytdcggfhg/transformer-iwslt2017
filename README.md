# Transformer for IWSLT2017 英德翻译

本项目从零实现了基于 Transformer 编码器–解码器模型的英德翻译模型，使用 IWSLT2017 数据集进行训练。

## 项目结构

```
.
├── src/
│   ├── config.py          # 配置和参数解析模块
│   ├── utils.py           # 工具函数（随机种子、设备设置、绘图等）
│   ├── data.py            # 数据处理模块（分词器、数据集、词汇表等）
│   ├── model.py           # Transformer 模型架构定义
│   ├── trainer.py         # 训练逻辑（训练循环、评估、早停等）
│   ├── inference.py        # 推理模块（greedy decoder等）
│   └── train.py           # 主训练脚本（整合所有模块）
├── scripts/
│   └── run.sh             # 训练脚本
├── results/               # 训练结果（曲线和指标）
├── requirements.txt       # Python 依赖
└── README.md              # 本文件
```

## 模块说明

### 核心模块

- **`config.py`**: 负责命令行参数解析和配置管理
- **`utils.py`**: 提供工具函数，包括随机种子设置、设备配置、训练曲线绘制、指标保存等
- **`data.py`**: 处理所有数据相关操作，包括分词器加载、数据集加载、词汇表构建、collate function 等
- **`model.py`**: 定义完整的 Transformer 模型架构，包括编码器、解码器、注意力机制等
- **`trainer.py`**: 实现训练循环、模型评估、早停机制、模型保存等训练逻辑
- **`inference.py`**: 提供推理功能，包括贪心解码器和测试翻译
- **`train.py`**: 主入口文件，整合所有模块，作为训练脚本的入口点



## 硬件要求

- **GPU**: 支持 CUDA 的 NVIDIA GPU（推荐：8GB+ 显存）
- **内存**: 推荐 16GB+
- **存储**: 约 5GB（用于数据集和模型）

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 spaCy 语言模型

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```
## 数据集准备

本项目使用 Hugging Face 的 `datasets` 库来处理 IWSLT2017 数据集。无需手动下载此数据集。

当第一次运行训练脚本 (`python src/train.py`) 时，`datasets` 库将会：

1. 自动下载 IWSLT2017 (de-en) 数据集。
2. 将其缓存到本地磁盘（通常位于 `~/.cache/huggingface/datasets`）。

后续所有运行都将直接从这个本地缓存中读取数据，无需再次下载。


## 使用方法

### 快速开始

使用默认参数运行训练：

```bash
bash scripts/run.sh
```

或直接使用 Python 运行：

```bash
python src/train.py
```

### 命令行参数

- `--epochs`: 训练轮数（默认：100）
- `--batch_size`: 批次大小（默认：64）
- `--d_model`: 嵌入维度（默认：256）
- `--d_ff`: 前馈网络维度（默认：1024）
- `--d_k`: K(=Q), V 的维度（默认：32）
- `--n_layers`: 编码器/解码器层数（默认：4）
- `--n_heads`: 多头注意力头数（默认：8）
- `--lr`: 学习率（默认：0.0001）
- `--train_samples`: 训练样本数（默认：40000）
- `--device`: 使用的设备（如 "cuda", "cuda:0", "cpu"）。未指定时自动检测
- `--dropout`: Dropout 率（默认：0.1）
- `--early_stop`: 启用早停机制
- `--patience`: 早停的耐心值（默认：5）
- `--min_delta`: 被视为改进的最小变化量（默认：0.0）
- `--seed`: 随机种子，用于可重现性（默认：42）

### 消融实验参数

- `--no_residual`: 禁用残差连接（消融实验）
- `--no_positional_encoding`: 禁用位置编码（消融实验）
- `--single_head`: 使用单头注意力而非多头注意力（消融实验）

### 使用示例

**基础训练：**

```bash
python src/train.py --epochs 20 --batch_size 64
```

**使用早停机制训练：**

```bash
python src/train.py --epochs 100 --early_stop --patience 10
```

**在指定 GPU 上训练：**

```bash
python src/train.py --device cuda:0 --epochs 20
```

**消融实验示例：**

```bash
python src/train.py --epochs 20 --no_residual --no_positional_encoding
```

## 复现实验

要复现精确的结果，请使用以下命令和指定的随机种子：

### 默认配置

```bash
python src/train.py \
    --epochs 20 \
    --batch_size 64 \
    --d_model 256 \
    --d_ff 2048 \
    --d_k 64 \
    --n_layers 4 \
    --n_heads 8 \
    --lr 0.0001 \
    --train_samples 40000 \
    --dropout 0.1 \
    --seed 42
```

### 使用早停机制

```bash
python src/train.py \
    --epochs 100 \
    --batch_size 64 \
    --d_model 256 \
    --d_ff 2048 \
    --d_k 64 \
    --n_layers 4 \
    --n_heads 8 \
    --lr 0.0001 \
    --train_samples 40000 \
    --dropout 0.1 \
    --early_stop \
    --patience 5 \
    --min_delta 0.0 \
    --seed 42
```

### 更大的模型

```bash
python src/train.py \
    --epochs 20 \
    --batch_size 32 \
    --d_model 512 \
    --d_ff 2048 \
    --d_k 64 \
    --n_layers 6 \
    --n_heads 8 \
    --lr 0.0001 \
    --train_samples 40000 \
    --dropout 0.1 \
    --seed 42
```

## 结果

训练结果保存在 `results/` 目录下，结构如下：

```
results/
└── YYYYMMDD_HHMMSS_ep{epochs}_bs{batch_size}_dm{d_model}_.../
    ├── best_model.pth          # 最佳模型检查点
    ├── training_curves.png     # 训练曲线（损失和困惑度）
    └── metrics.txt             # 详细的训练指标
```

每个实验目录以时间戳和超参数命名，便于识别。

