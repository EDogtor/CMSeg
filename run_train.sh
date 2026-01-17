#!/bin/bash
# CoCoSeg 训练脚本 - CIPA风格训练策略

# 设置数据集路径
DATASET_ROOT="/root/autodl-tmp/CoCoSeg/pkdata/"

# 检查数据集是否存在
if [ ! -d "$DATASET_ROOT" ]; then
    echo "Error: Dataset directory $DATASET_ROOT does not exist!"
    exit 1
fi

echo "========================================"
echo "CoCoSeg Training (CIPA-style Strategy)"
echo "Dataset: $DATASET_ROOT"
echo "Training Strategy:"
echo "  - Step-based LR scheduler (每个batch更新)"
echo "  - Weight decay: 1e-2 (CIPA default)"
echo "  - Mixed precision: Enabled"
echo "  - Architecture improvements: Enabled"
echo "========================================"

python main.py \
  --train \
  --dataset_root "$DATASET_ROOT" \
  --dataset_type pclt20k \
  --model_type vmamba \
  --normalization vmamba \
  --use_gpu \
  --bs 4 \
  --lr 6e-5 \
  --epoch 50 \
  --warmup_epochs 5 \
  --weight_decay 1e-2 \
  --eps 1e-8 \
  --amp \
  --save_freq 10 \
  --val_freq 1 \
  --patience 25 \
  --loss_type combined \
  --dice_weight 0.7 \
  --ce_weight 0.3 \
  --pos_weight 5.0

echo "Training completed!"
