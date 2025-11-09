#!/bin/bash

# ====================================================================================================
# Transformer 模型训练脚本
# 本脚本用于运行 Transformer 模型的训练，支持所有命令行参数
# ====================================================================================================

# 设置默认参数
EPOCHS=20
BATCH_SIZE=64
D_MODEL=256
D_FF=2048
D_K=64
N_LAYERS=4
N_HEADS=8
LR=0.0001
TRAIN_SAMPLES=40000
DROPOUT=0.1
SEED=42
EARLY_STOP=""
PATIENCE=""
MIN_DELTA=""
DEVICE=""
NO_RESIDUAL=""
NO_POSITIONAL_ENCODING=""
SINGLE_HEAD=""

# 显示使用说明
show_usage() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --epochs NUM                   训练轮数 (默认: 20)"
    echo "  --batch_size NUM               批次大小 (默认: 64)"
    echo "  --d_model NUM                  嵌入维度 (默认: 256)"
    echo "  --d_ff NUM                     前馈网络维度 (默认: 2048)"
    echo "  --d_k NUM                      K(=Q), V 的维度 (默认: 64)"
    echo "  --n_layers NUM                 编码器/解码器层数 (默认: 4)"
    echo "  --n_heads NUM                  多头注意力头数 (默认: 8)"
    echo "  --lr NUM                       学习率 (默认: 0.0001)"
    echo "  --train_samples NUM            训练样本数 (默认: 40000)"
    echo "  --dropout NUM                  Dropout 率 (默认: 0.1)"
    echo "  --seed NUM                     随机种子 (默认: 42)"
    echo "  --device DEVICE                使用的设备 (如: cuda, cuda:0, cpu)"
    echo "  --early_stop                   启用早停机制"
    echo "  --patience NUM                 早停的耐心值 (默认: 5)"
    echo "  --min_delta NUM                被视为改进的最小变化量 (默认: 0.0)"
    echo "  --no_residual                  禁用残差连接 (消融实验)"
    echo "  --no_positional_encoding       禁用位置编码 (消融实验)"
    echo "  --single_head                  使用单头注意力 (消融实验)"
    echo "  --help                         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --epochs 20 --batch_size 64"
    echo "  $0 --epochs 100 --early_stop --patience 10"
    echo "  $0 --device cuda:0 --epochs 20"
    echo "  $0 --epochs 20 --no_residual --no_positional_encoding"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --d_model)
            D_MODEL="$2"
            shift 2
            ;;
        --d_ff)
            D_FF="$2"
            shift 2
            ;;
        --d_k)
            D_K="$2"
            shift 2
            ;;
        --n_layers)
            N_LAYERS="$2"
            shift 2
            ;;
        --n_heads)
            N_HEADS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --train_samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --early_stop)
            EARLY_STOP="--early_stop"
            shift
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --min_delta)
            MIN_DELTA="$2"
            shift 2
            ;;
        --no_residual)
            NO_RESIDUAL="--no_residual"
            shift
            ;;
        --no_positional_encoding)
            NO_POSITIONAL_ENCODING="--no_positional_encoding"
            shift
            ;;
        --single_head)
            SINGLE_HEAD="--single_head"
            shift
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "错误: 未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python src/train.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --d_k $D_K \
    --n_layers $N_LAYERS \
    --n_heads $N_HEADS \
    --lr $LR \
    --train_samples $TRAIN_SAMPLES \
    --dropout $DROPOUT \
    --seed $SEED"

# 添加可选参数
if [ ! -z "$EARLY_STOP" ]; then
    CMD="$CMD $EARLY_STOP"
    if [ ! -z "$PATIENCE" ]; then
        CMD="$CMD --patience $PATIENCE"
    fi
    if [ ! -z "$MIN_DELTA" ]; then
        CMD="$CMD --min_delta $MIN_DELTA"
    fi
fi

if [ ! -z "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ ! -z "$NO_RESIDUAL" ]; then
    CMD="$CMD $NO_RESIDUAL"
fi

if [ ! -z "$NO_POSITIONAL_ENCODING" ]; then
    CMD="$CMD $NO_POSITIONAL_ENCODING"
fi

if [ ! -z "$SINGLE_HEAD" ]; then
    CMD="$CMD $SINGLE_HEAD"
fi

# 打印命令并运行
echo "======================================================================"
echo "Transformer 模型训练"
echo "======================================================================"
echo ""
echo "运行命令:"
echo "$CMD"
echo ""
echo "======================================================================"
echo ""

# 运行训练
eval $CMD
