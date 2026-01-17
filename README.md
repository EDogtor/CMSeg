# CoCoSeg: 基于VMamba的CT-PET双模态医疗图像分割模型

## 📋 项目简介

CoCoSeg是一个专门用于CT-PET双模态医疗图像分割的深度学习模型。本项目基于**VMamba (Vision Mamba)**架构，采用双独立编码器和多层级特征融合策略，实现了高质量的多模态医疗图像分割。模型支持512×512像素的CT-PET图像对，专门用于肿瘤等病灶的精确分割任务。

### 核心特点

- ✅ **VMamba架构**: 基于状态空间模型(SSM)的高效视觉编码器，线性复杂度
- ✅ **双独立编码器**: CT和PET各使用独立的VMamba编码器，保持模态特异性
- ✅ **多层级融合**: CRM (Channel Rectify Module) + DCIM (Dual Cross-Modal Interaction) + 自适应融合
- ✅ **MambaDecoder**: 基于VMamba的上采样解码器，保持架构一致性
- ✅ **先进训练策略**: EMA、混合精度训练、阈值扫描、早停机制
- ✅ **多种损失函数**: 支持Tversky+BCE、Dice+CE等多种组合

---

## 🏗️ 模型架构详解

> 📐 **架构图绘制指南**: 详细的架构说明和绘图指南请参考 [ARCHITECTURE_DIAGRAM.md](./ARCHITECTURE_DIAGRAM.md)，包含与CIPA的对比和绘图建议。

### 整体架构概览

CoCoSeg采用**编码器-解码器(Encoder-Decoder)**架构，专门设计用于CT-PET双模态分割：

```
输入层:
├── CT图像 [B, 1, 512, 512] → 复制为3通道 [B, 3, 512, 512]
└── PET图像 [B, 1, 512, 512] → 复制为3通道 [B, 3, 512, 512]

编码器层 (双独立VMamba):
├── CT_VMamba_Encoder → 4层特征: [96, 192, 384, 768] channels
└── PET_VMamba_Encoder → 4层特征: [96, 192, 384, 768] channels

多层级融合 (4个层级):
├── Level 1 (96ch):  CRM → DCIM → Adaptive Fusion
├── Level 2 (192ch): CRM → DCIM → Adaptive Fusion  
├── Level 3 (384ch): CRM → DCIM → Adaptive Fusion
└── Level 4 (768ch): CRM → DCIM → Adaptive Fusion

解码器层 (MambaDecoder):
├── 上采样 + 跳跃连接 (4层)
└── 输出层 → [B, 1, 512, 512]
```

### 1. VMamba编码器架构

#### 1.1 VMamba Backbone

模型提供三种VMamba backbone（与`models/vmamba/dual_vmamba.py`和`models/vmamba/builder.py`一致）：

```python
# VMamba backbone 配置（当前代码实际取值）
backbone = 'sigma_tiny'     # 可选: sigma_tiny / sigma_small / sigma_base
depths = [2, 2, 9, 2]       # sigma_tiny
dims = 96
channels = [96, 192, 384, 768]
pretrained = './pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth'
patch_size = 4
image_size = [512, 512]

# sigma_small: depths=[2, 2, 27, 2], dims=96
# sigma_base:  depths=[2, 2, 27, 2], dims=128
```

**四个Stage的详细结构**（以 `sigma_tiny` 为例）：

| Stage | 输入尺寸 | 输出通道 | Block数 | 分辨率变化 |
|-------|---------|---------|---------|-----------|
| Stage 1 | 512×512 | 96 | 2 | 512×512 → 128×128 |
| Stage 2 | 128×128 | 192 | 2 | 128×128 → 64×64 |
| Stage 3 | 64×64 | 384 | 9 | 64×64 → 32×32 |
| Stage 4 | 32×32 | 768 | 2 | 32×32 → 16×16 |

#### 1.2 SS2D (Selective Scan 2D) 核心模块

VMamba的核心是**SS2D (Selective Scan 2D)**模块，基于状态空间模型(SSM)：

```python
class SS2D(nn.Module):
    """二维选择性扫描模块 - VMamba的核心"""
    def __init__(self, d_model=96, d_state=16, ssm_ratio=2, ...):
        # 输入投影: d_model → d_inner * 2
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        
        # 深度可分离卷积 (可选)
        self.conv2d = nn.Conv2d(d_inner, d_inner, groups=d_inner, ...)
        
        # SSM参数
        self.x_proj_weight  # 状态投影权重
        self.dt_projs_weight  # 时间步投影权重
        self.A_logs, self.Ds  # 状态转移矩阵
        
    def forward_core(self, x):
        # 1. 四方向扫描 (CrossScan)
        xs = CrossScan.apply(x)  # [B, 4, D, H*W]
        # 四个方向: HW, WH, 翻转HW, 翻转WH
        
        # 2. 选择性扫描 (Selective Scan)
        ys = SelectiveScan.apply(xs, dts, As, Bs, Cs, Ds, ...)
        
        # 3. 合并四方向结果 (CrossMerge)
        y = CrossMerge.apply(ys)  # [B, D, H*W]
        return y
```

**SS2D的工作原理**：

1. **CrossScan (交叉扫描)**: 将2D特征图转换为4个方向的1D序列
   - 方向1: 行优先扫描 (H×W)
   - 方向2: 列优先扫描 (W×H)
   - 方向3: 翻转行优先扫描
   - 方向4: 翻转列优先扫描

2. **Selective Scan (选择性扫描)**: 对每个方向的序列应用状态空间模型
   - 状态方程: `h(t) = A * h(t-1) + B * x(t)`
   - 输出方程: `y(t) = C * h(t) + D * x(t)`
   - 线性复杂度: O(N) vs Transformer的O(N²)

3. **CrossMerge (交叉合并)**: 将4个方向的结果合并回2D特征图

**优势**：
- **线性复杂度**: O(N) vs Transformer的O(N²)，适合高分辨率图像
- **长距离依赖**: SSM天然适合建模长序列依赖
- **高效计算**: 相比Self-Attention，计算和内存更高效

#### 1.3 双独立编码器设计

CT和PET使用**完全独立的VMamba编码器**（不共享权重）：

```python
class RGBXTransformer(nn.Module):
    def __init__(self, ...):
        # CT编码器 (独立权重)
        self.vssm_rgb = Backbone_VSSM(
            in_chans=3,  # 单通道复制为3通道以利用预训练权重
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='./pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth'
        )
        
        # PET编码器 (独立权重)
        self.vssm_x = Backbone_VSSM(
            in_chans=3,  # 单通道复制为3通道
            depths=[2, 2, 9, 2],
            dims=96,
            pretrained='./pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth'
        )
```

**设计理念**：
- **模态特异性**: CT和PET具有不同的成像原理和特征分布，独立编码器可以学习各自模态的独特特征
- **预训练权重**: 单通道图像复制为3通道，充分利用ImageNet预训练的VMamba权重
- **特征互补**: 两个编码器提取的特征在后续融合阶段互补，提升分割性能

### 2. 多层级特征融合机制

模型在4个层级进行CT-PET特征融合，每个层级包含三个关键模块：

#### 2.1 CRM (Channel Rectify Module) - 通道校正模块

CRM用于跨模态特征对齐，学习每个通道的重要性权重：

```python
class ChannelRectifyModule(nn.Module):
    """通道校正模块 - 学习CT和PET特征的通道权重"""
    def __init__(self, dim, HW, reduction=16):
        # ChannelWeights: 使用SS1D学习通道权重
        self.channel_weights = ChannelWeights(dim=HW, channel_dim=dim)
    
    def forward(self, x1, x2):
        # x1: CT特征 [B, C, H, W]
        # x2: PET特征 [B, C, H, W]
        
        # 学习通道权重 [2, B, C, 1, 1]
        channel_weights = self.channel_weights(x1, x2)
        
        # 通道加权增强
        out_x1 = x1 + channel_weights[0] * x1  # CT特征增强
        out_x2 = x2 + channel_weights[1] * x2  # PET特征增强
        
        return out_x1, out_x2
```

**ChannelWeights内部结构**：

```python
class ChannelWeights(nn.Module):
    def __init__(self, dim, channel_dim, reduction=4):
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),           # H*W维度归一化
            nn.Linear(dim, 96),          # 降维
            nn.GELU(),
            SS1D(d_model=96, ...),       # 使用SS1D处理序列
            nn.LayerNorm(96),
            nn.Linear(96, 1),            # 输出权重
            nn.Sigmoid()                 # 归一化到[0,1]
        )
    
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # 拼接CT和PET特征
        x = torch.cat([x1, x2], dim=1).view(B, 2*C, H*W)  # [B, 2C, HW]
        # 学习通道权重
        channel_weights = self.mlp(x)  # [B, 2C, 1]
        return channel_weights.reshape(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
```

**功能**：
- **通道对齐**: 自动学习CT和PET特征中哪些通道更重要
- **跨模态交互**: 通过拼接和SS1D处理，实现跨模态的通道权重学习
- **自适应增强**: 对重要通道进行加权增强，抑制噪声通道

#### 2.2 DCIM (Dual Cross-Modal Interaction Module) - 双交叉模态交互模块

DCIM通过区域Mamba实现跨模态的细粒度交互：

```python
# DCIM包含两个组件:
# 1. Region Patch (区域分块)
cross_rgb, cross_x, (H_out, W_out), (H_in, W_in) = self.region_patch[i](
    cross_rgb, cross_x
)

# 2. Channel Attention Mamba (通道注意力Mamba)
attn_output = self.channel_attn_mamba[i](
    cross_rgb.contiguous(), 
    cross_x.contiguous(),
    H_out, W_out, H_in, W_in
).permute(0, 3, 1, 2).contiguous()
```

**Region Patch模块**：
- 将特征图划分为多个小区域(如4×4 patches)
- 每个区域独立处理，保持局部细节
- 输出区域特征和空间维度信息

**Channel Attention Mamba**：
- 使用区域Mamba处理跨模态交互
- 学习CT和PET区域之间的注意力权重
- 输出跨模态增强的特征

#### 2.3 Adaptive Fusion Module - 自适应融合模块

自适应融合模块学习CT和PET的最优融合权重：

```python
class AdaptiveFusionModule(nn.Module):
    """自适应模态融合模块 - 学习CT和PET的最优融合权重"""
    def __init__(self, dim):
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # 全局平均池化
            nn.Conv2d(dim * 2, max(dim // 4, 16), 1),  # 降维
            nn.ReLU(),
            nn.Conv2d(max(dim // 4, 16), 2, 1),  # 输出2个权重
            nn.Softmax(dim=1)                   # 归一化
        )
    
    def forward(self, feat_rgb, feat_x):
        # 拼接特征
        concat_feat = torch.cat([feat_rgb, feat_x], dim=1)  # [B, 2C, H, W]
        
        # 学习融合权重
        weights = self.weight_net(concat_feat)  # [B, 2, 1, 1]
        w_rgb, w_x = weights[:, 0:1], weights[:, 1:2]
        
        # 自适应加权融合
        fused = w_rgb * feat_rgb + w_x * feat_x
        return fused
```

**融合策略**：
```python
# 完整融合流程 (每个层级)
for i in range(4):  # 4个层级
    # 1. CRM: 通道校正
    cross_rgb, cross_x = self.cross_mamba[i](out_rgb, out_x)
    
    # 2. DCIM: 区域交互
    cross_rgb, cross_x, (H_out, W_out), (H_in, W_in) = \
        self.region_patch[i](cross_rgb, cross_x)
    attn_output = self.channel_attn_mamba[i](
        cross_rgb, cross_x, H_out, W_out, H_in, W_in
    )
    
    # 3. Adaptive Fusion: 自适应融合
    adaptive_fused = self.adaptive_fusion[i](out_rgb, out_x)
    
    # 4. 最终融合: 注意力输出 + 自适应融合
    x_fuse = adaptive_fused + attn_output
    outs_fused.append(x_fuse)
```

### 3. MambaDecoder 解码器架构

MambaDecoder采用与编码器一致的VMamba架构，通过上采样逐步恢复分辨率：

```python
class MambaDecoder(nn.Module):
    def __init__(self, 
                 img_size=[512, 512],
                 in_channels=[96, 192, 384, 768],
                 num_classes=1,
                 embed_dim=96,
                 depths=[4, 4, 4, 4]):
        # 4个上采样层
        self.layers_up = nn.ModuleList()
        for i_layer in range(4):
            if i_layer == 0:
                # 第一层: PatchExpand (768 → 384)
                layer_up = PatchExpand(...)
            else:
                # 其他层: Mamba_up (包含CVSSDecoderBlock)
                layer_up = Mamba_up(
                    dim=embed_dim * 2 ** (3 - i_layer),
                    depth=depths[3 - i_layer],
                    upsample=PatchExpand if (i_layer < 3) else None
                )
            self.layers_up.append(layer_up)
        
        # 最终上采样 (4倍)
        self.up = FinalUpsample_X4(...)
        self.output = nn.Conv2d(embed_dim, num_classes, 1)
```

**解码器流程**：

| 层级 | 输入 | 操作 | 输出 | 分辨率 |
|------|------|------|------|--------|
| Layer 0 | 768ch, 16×16 | PatchExpand | 384ch, 32×32 | 2×上采样 |
| Layer 1 | 384ch, 32×32 | Mamba_up + Skip | 192ch, 64×64 | 2×上采样 |
| Layer 2 | 192ch, 64×64 | Mamba_up + Skip | 96ch, 128×128 | 2×上采样 |
| Layer 3 | 96ch, 128×128 | Mamba_up + Skip | 96ch, 128×128 | 保持 |
| Final | 96ch, 128×128 | FinalUpsample_X4 | 1ch, 512×512 | 4×上采样 |

**CVSSDecoderBlock** (解码器中的核心block)：

```python
class CVSSDecoderBlock(nn.Module):
    """VMamba解码器Block"""
    def __init__(self, hidden_dim, ...):
        self.norm1 = norm_layer(hidden_dim)
        self.ssm = SS2D(d_model=hidden_dim, ...)  # SS2D模块
        self.norm2 = norm_layer(hidden_dim)
        self.mlp = Mlp(hidden_dim, ...)  # MLP
        
    def forward(self, x):
        # 残差连接 + SS2D
        x = x + self.ssm(self.norm1(x))
        # 残差连接 + MLP
        x = x + self.mlp(self.norm2(x))
        return x
```

**跳跃连接**：
- 解码器每层与对应层级的融合特征进行跳跃连接
- 保持细节信息，提升分割精度

### 4. 完整数据流

```
输入阶段:
├── CT [B,1,512,512] → repeat(1,3,1,1) → [B,3,512,512]
└── PET [B,1,512,512] → repeat(1,3,1,1) → [B,3,512,512]

编码阶段 (双独立VMamba):
├── CT_VMamba:
│   ├── Stage1: [B,3,512,512] → [B,96,128,128]   (patch_size=4)
│   ├── Stage2: [B,96,128,128] → [B,192,64,64]   (下采样2×)
│   ├── Stage3: [B,192,64,64] → [B,384,32,32]    (下采样2×)
│   └── Stage4: [B,384,32,32] → [B,768,16,16]    (下采样2×)
│
└── PET_VMamba:
    ├── Stage1: [B,3,512,512] → [B,96,128,128]
    ├── Stage2: [B,96,128,128] → [B,192,64,64]
    ├── Stage3: [B,192,64,64] → [B,384,32,32]
    └── Stage4: [B,384,32,32] → [B,768,16,16]

融合阶段 (4个层级):
├── Level 4 (768ch, 16×16):
│   ├── CRM: CT(768) + PET(768) → cross_CT, cross_PET
│   ├── DCIM: Region Patch + Channel Attn Mamba → attn_output
│   └── Adaptive Fusion: CT + PET → fused(768)
│
├── Level 3 (384ch, 32×32):
│   ├── CRM → cross_CT, cross_PET
│   ├── DCIM → attn_output
│   └── Adaptive Fusion → fused(384)
│
├── Level 2 (192ch, 64×64):
│   ├── CRM → cross_CT, cross_PET
│   ├── DCIM → attn_output
│   └── Adaptive Fusion → fused(192)
│
└── Level 1 (96ch, 128×128):
    ├── CRM → cross_CT, cross_PET
    ├── DCIM → attn_output
    └── Adaptive Fusion → fused(96)

解码阶段 (MambaDecoder):
├── Layer 0: fused(768,16×16) → PatchExpand → (384,32×32)
├── Layer 1: (384,32×32) + Skip(384) → Mamba_up → (192,64×64)
├── Layer 2: (192,64×64) + Skip(192) → Mamba_up → (96,128×128)
├── Layer 3: (96,128×128) + Skip(96) → Mamba_up → (96,128×128)
└── Final: (96,128×128) → FinalUpsample_X4 → (1,512×512)

输出:
└── [B, 1, 512, 512] (分割mask)
```

---

## 📊 训练结果

基于训练日志 `logs/20251218_230355/` 的结果：

### 训练配置

```json
{
    "epoch": 50,
    "lr": 6e-05,
    "bs": 4,
    "loss_type": "tversky_bce",
    "tversky_weight": 0.7,
    "bce_weight": 0.3,
    "tversky_alpha": 0.7,  // 更关注假阴性(漏检)
    "tversky_beta": 0.3,   // 假阳性权重
    "model_type": "vmamba",
    "backbone": "sigma_tiny",
    "patience": 25,
    "warmup_epochs": 5,
    "weight_decay": 0.01,
    "amp": true  // 混合精度训练
}
```

### 性能指标

**最佳验证结果** (Epoch 50):
- **最佳Dice**: 0.7755 (阈值扫描)
- **最佳阈值**: 0.35
- **IoU**: ~0.63
- **F1**: ~0.77
- **HD95**: ~20.5mm

**训练曲线趋势**:
- 训练Loss: 0.739 → 0.085 (持续下降)
- 训练Dice: 0.031 → 0.861 (持续上升)
- 验证Dice: 0.203 → 0.775 (稳定提升)
- 验证HD95: ~20-21mm (稳定)

### 模型参数量

- **总参数量**: ~33M (VMamba Tiny)
- **可训练参数**: ~33M
- **模型大小**: ~780MB (best_model.pth)

---

## 🚀 快速开始

### 环境配置

#### 1. 创建虚拟环境

```bash
conda create -n cocoseg python=3.10
conda activate cocoseg
```

#### 2. 安装PyTorch

```bash
# CUDA 12.1/12.4 (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. 安装依赖

```bash
cd CoCoSeg
pip install -r requirements.txt
```

#### 4. 编译Selective Scan CUDA扩展

```bash
cd models/vmamba/selective_scan
python setup.py build_ext --inplace
cd ../../..
```

### 数据集准备

支持PCLT20K格式数据集，目录结构：

```
pkdata/
├── patient_id_1/
│   ├── patient_id_1_slice_001_ct.png
│   ├── patient_id_1_slice_001_pet.png
│   ├── patient_id_1_slice_001_mask.png
│   └── ...
├── patient_id_2/
└── ...
```

### 训练模型

#### 基础训练命令

```bash
python main.py --train --use_gpu \
    --dataset_root ./pkdata/ \
    --dataset_type pclt20k \
    --model_type vmamba \
    --epoch 50 \
    --bs 4 \
    --lr 6e-5 \
    --loss_type tversky_bce \
    --tversky_weight 0.7 \
    --bce_weight 0.3 \
    --tversky_alpha 0.7 \
    --tversky_beta 0.3 \
    --patience 25 \
    --warmup_epochs 5 \
    --weight_decay 0.01 \
    --amp
```

#### 使用配置文件

```bash
python main.py --train --use_gpu \
    --config logs/20251218_230355/config.json \
    --dataset_root ./pkdata/
```

#### 恢复训练

```bash
python main.py --train --use_gpu \
    --dataset_root ./pkdata/ \
    --resume \
    --resume_ckpt logs/20251218_230355/checkpoint_epoch_30.pth
```

### 测试模型

```bash
python test_model.py \
    --ckpt logs/20251218_230355/best_model.pth \
    --use_gpu \
    --mode eval \
    --dataset_root ./pkdata/ \
    --split test \
    --save_predictions
```

---

## 📈 训练配置详解

### 损失函数

#### Tversky + BCE Loss (推荐) ⭐

```bash
--loss_type tversky_bce \
--tversky_weight 0.7 \
--bce_weight 0.3 \
--tversky_alpha 0.7 \  # FN权重 (漏检惩罚)
--tversky_beta 0.3     # FP权重 (误检惩罚)
```

**Tversky Loss公式**:
```
Tversky = TP / (TP + α*FN + β*FP)
Loss = 1 - Tversky
```

**优势**:
- **控制漏检**: α=0.7, β=0.3 更关注假阴性，适合肿瘤分割
- **稳定训练**: BCE提供稳定的梯度
- **最佳性能**: 在验证集上达到0.775 Dice

#### 其他损失函数

| 损失函数 | 命令 | 适用场景 |
|---------|------|---------|
| Combined (Dice+CE) | `--loss_type combined` | 通用分割 |
| Dice | `--loss_type dice` | 小目标分割 |
| IoU | `--loss_type iou` | 直接优化IoU |
| Focal | `--loss_type focal` | 难样本多 |

### 优化器配置

```python
# AdamW优化器 (与CIPA一致)
optimizer = torch.optim.AdamW(
    param_groups,  # 参数分组
    lr=6e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# 参数分组策略
# - Linear/Conv: 使用weight_decay
# - BatchNorm/LayerNorm: weight_decay=0.0
```

### 学习率调度

```python
# CIPA风格: 基于step数的余弦退火
# - Warm-up: 5 epochs, 起始lr = lr * 1e-3
# - Cosine Annealing: 最终lr = lr * 1e-6
# - 每个batch更新一次
```

### 训练技巧

1. **EMA (Exponential Moving Average)**: 衰减率0.999，提升模型稳定性
2. **混合精度训练 (AMP)**: 加速训练，节省显存
3. **阈值扫描**: 验证时自动寻找最佳二值化阈值
4. **早停机制**: Patience=25，防止过拟合

---

## 📂 目录结构

```
CoCoSeg/
├── main.py                    # 主训练/测试脚本
├── test_model.py              # 模型评估脚本
├── requirements.txt            # 依赖列表
├── requirements-dev.txt        # 开发/全量依赖列表
├── README.md                  # 本文档
│
├── models/                     # 模型相关代码
│   ├── __init__.py            # 模块初始化文件
│   │
│   ├── vmamba/                # VMamba核心模块目录
│   │   ├── __init__.py        # VMamba模块初始化
│   │   │
│   │   ├── builder.py          # EncoderDecoder构建器
│   │   │                       # - 负责构建完整的编码器-解码器架构
│   │   │                       # - 根据配置选择backbone (sigma_tiny/sigma_small/sigma_base)
│   │   │                       # - 集成编码器和解码器
│   │   │
│   │   ├── dual_vmamba.py      # 双模态VMamba编码器
│   │   │                       # - RGBXTransformer: 双独立编码器主类
│   │   │                       # - AdaptiveFusionModule: 自适应融合模块
│   │   │                       # - vssm_tiny/small/base: 不同规模的backbone
│   │   │                       # - 实现CRM + DCIM + Adaptive Fusion流程
│   │   │
│   │   ├── vmamba.py           # VMamba核心实现
│   │   │                       # - SS2D: 二维选择性扫描核心模块
│   │   │                       # - VSSBlock: VMamba基础block
│   │   │                       # - Backbone_VSSM: VMamba backbone实现
│   │   │                       # - CrossMambaFusionBlock: 跨模态融合block
│   │   │                       # - CVSSDecoderBlock: 解码器block
│   │   │                       # - 包含CrossScan, CrossMerge等核心操作
│   │   │
│   │   ├── MambaDecoder.py     # Mamba解码器
│   │   │                       # - MambaDecoder: 主解码器类
│   │   │                       # - PatchExpand: Patch扩展上采样
│   │   │                       # - Mamba_up: 上采样层
│   │   │                       # - FinalUpsample_X4: 最终4倍上采样
│   │   │                       # - 实现跳跃连接和特征融合
│   │   │
│   │   ├── mamba_net_utils.py  # CRM等工具模块
│   │   │                       # - ChannelRectifyModule (CRM): 通道校正模块
│   │   │                       # - ChannelWeights: 通道权重学习
│   │   │                       # - SS1D: 一维选择性扫描（用于CRM）
│   │   │
│   │   ├── local_vmamba/       # 局部Mamba实现（用于DCIM）
│   │   │   ├── region_mamba.py # 区域Mamba模块
│   │   │   │                   # - SS2D_Region: 区域选择性扫描
│   │   │   │                   # - Region_global_Block: 区域全局block
│   │   │   │                   # - 用于DCIM中的区域交互
│   │   │   │
│   │   │   └── local_scan.py   # 局部扫描实现
│   │   │                       # - local_scan: 局部扫描函数
│   │   │                       # - local_reverse: 局部反向扫描
│   │   │                       # - 基于Triton的CUDA实现
│   │   │
│   │   └── selective_scan/     # Selective Scan CUDA扩展
│   │       ├── setup.py        # CUDA扩展编译脚本
│   │       ├── csrc/           # CUDA源码目录
│   │       │   └── selective_scan/
│   │       │       ├── selective_scan_core.cu    # 核心CUDA实现
│   │       │       ├── selective_scan_fwd_kernel.cuh  # 前向kernel
│   │       │       ├── selective_scan_bwd_kernel.cuh  # 反向kernel
│   │       │       └── ...
│   │       │
│   │       ├── selective_scan/ # Python接口
│   │       │   └── selective_scan_interface.py  # SelectiveScanFn接口
│   │       │
│   │       └── build/          # 编译输出目录
│   │
│   ├── segmentation_loss.py   # 分割损失函数实现
│   │                           # - DiceLoss: Dice损失
│   │                           # - TverskyLoss: Tversky损失（控制漏检）
│   │                           # - CombinedSegLoss: Dice+CE组合损失
│   │                           # - FocalLoss: Focal损失
│   │                           # - IoULoss: IoU损失
│   │
│   ├── model.py               # 传统基线模型（保留对比）
│   │                           # - Vgg19_Encoder: VGG19编码器
│   │                           # - UNetEncoder: 基础UNet编码器
│   │                           # - DualIndependentEncoderUNet: 双独立编码器UNet
│   │                           # - 主项目使用VMamba，此文件保留用于参考
│   │
│   ├── train_tasks.py         # 训练任务相关（旧版训练代码）
│   │                           # - 包含旧的训练循环实现
│   │                           # - 本项目使用main.py进行训练
│   │
│   ├── measure_model.py       # 模型参数量/FLOPs测量工具
│   │                           # - 使用torchstat统计模型参数和计算量
│   │
│   └── P_loss.py              # VGG19感知损失（图像重建任务用）
│                               # - 本项目为分割任务，不使用此文件
│
├── data/
│   └── dataset.py              # PCLT20K数据集加载器
│
├── utils/
│   ├── ema.py                 # EMA实现
│   ├── early_stopping.py      # 早停机制
│   └── ...
│
├── logs/                      # 训练日志
│   └── 20251218_230355/
│       ├── best_model.pth     # 最佳模型
│       ├── latest.pth         # 最新模型
│       ├── checkpoint_epoch_*.pth  # 定期checkpoint
│       ├── history.json       # 训练历史
│       ├── config.json        # 训练配置
│       └── tensorboard/       # TensorBoard日志
│
└── pretrained/
    └── vmamba/
        └── vssmtiny_dp01_ckpt_epoch_292.pth  # VMamba预训练权重
```

---

## 📁 模型文件详解

### 核心架构文件

#### 1. `models/vmamba/builder.py` - 模型构建器

**作用**: 构建完整的编码器-解码器架构，是模型的入口点。

**主要类**:
- `EncoderDecoder`: 完整的编码器-解码器模型类
  - 根据配置选择backbone (sigma_tiny / sigma_small / sigma_base)
  - 集成双模态编码器与 `MambaDecoder`
  - 当前默认 `deep_supervision=False`（与CIPA一致）
  - 处理模型初始化和权重加载

**使用流程**:
```python
from models.vmamba.builder import EncoderDecoder
model = EncoderDecoder(cfg=config, criterion=loss_fn)
```

#### 2. `models/vmamba/dual_vmamba.py` - 双模态编码器

**作用**: 实现CT-PET双模态特征提取和多层级融合。

**主要类**:
- `RGBXTransformer`: 双模态编码器主类
  - 包含两个独立的VMamba编码器（CT和PET）
  - 实现4个层级的特征融合（CRM + DCIM + Adaptive Fusion）
  - 输出融合后的多尺度特征
  
- `AdaptiveFusionModule`: 自适应融合模块
  - 学习CT和PET的最优融合权重
  - 基于全局平均池化和全连接层

- `vssm_tiny / vssm_small / vssm_base`: 不同规模的backbone配置

**数据流**:
```
CT输入 → CT_VMamba → [96,192,384,768]特征
PET输入 → PET_VMamba → [96,192,384,768]特征
         ↓
    4层融合 (CRM → DCIM → Adaptive Fusion)
         ↓
    融合特征 [96,192,384,768]
```

#### 3. `models/vmamba/vmamba.py` - VMamba核心实现

**作用**: VMamba架构的核心模块，包含所有基础组件。

**主要类和函数**:
- `SS2D`: 二维选择性扫描模块（VMamba的核心）
  - 实现CrossScan → SelectiveScan → CrossMerge流程
  - 线性复杂度O(N)的状态空间模型
  
- `VSSBlock`: VMamba基础block
  - 包含SS2D + MLP + 残差连接
  - 类似Transformer的block结构

- `Backbone_VSSM`: VMamba backbone实现
  - 4个stage的层次化特征提取
  - 支持不同规模配置（tiny/small/base）

- `CrossMambaFusionBlock`: 跨模态融合block
  - 用于编码器中的跨模态交互

- `CVSSDecoderBlock`: 解码器block
  - 用于MambaDecoder中的特征上采样

**关键操作**:
- `CrossScan`: 将2D特征转换为4个方向的1D序列
- `SelectiveScan`: 状态空间模型的前向传播
- `CrossMerge`: 将4个方向的结果合并回2D

#### 4. `models/vmamba/MambaDecoder.py` - Mamba解码器

**作用**: 基于VMamba的上采样解码器，逐步恢复分辨率。

**主要类**:
- `MambaDecoder`: 主解码器类
  - 4个上采样层，逐步从16×16恢复到128×128
  - 与编码器特征进行跳跃连接
  - 最终4倍上采样到512×512

- `PatchExpand`: Patch扩展上采样
  - 通过线性变换和reshape实现2倍上采样

- `Mamba_up`: 上采样层
  - 包含CVSSDecoderBlock和可选的PatchExpand

- `FinalUpsample_X4`: 最终4倍上采样
  - 使用双线性插值快速上采样到原始分辨率

**解码流程**:
```
融合特征[768,16×16] → PatchExpand → [384,32×32]
    ↓ + Skip(384)
Mamba_up → [192,64×64]
    ↓ + Skip(192)
Mamba_up → [96,128×128]
    ↓ + Skip(96)
Mamba_up → [96,128×128]
    ↓
FinalUpsample_X4 → [1,512×512]
```

#### 5. `models/vmamba/mamba_net_utils.py` - 融合工具模块

**作用**: 实现CRM (Channel Rectify Module)等融合相关工具。

**主要类**:
- `ChannelRectifyModule (CRM)`: 通道校正模块
  - 学习CT和PET特征的通道重要性权重
  - 对重要通道进行加权增强
  
- `ChannelWeights`: 通道权重学习
  - 使用SS1D处理通道序列
  - 输出每个通道的重要性权重

- `SS1D`: 一维选择性扫描
  - 用于CRM中的通道权重学习
  - 轻量级的状态空间模型

**CRM工作流程**:
```
CT特征 + PET特征 → 拼接 → SS1D处理 → 通道权重
    ↓
加权增强 → 输出校正后的CT和PET特征
```

#### 6. `models/vmamba/local_vmamba/` - 局部Mamba

**作用**: 实现DCIM中的区域级交互。

**主要文件**:
- `region_mamba.py`: 区域Mamba实现
  - `SS2D_Region`: 区域选择性扫描
  - `Region_global_Block`: 区域全局交互block
  - 将特征图划分为多个区域，每个区域独立处理

- `local_scan.py`: 局部扫描实现
  - `local_scan`: 局部窗口扫描
  - 基于Triton的CUDA加速实现

**DCIM工作流程**:
```
CT特征 + PET特征 → Region Patch (分块)
    ↓
每个区域独立处理 → Channel Attention Mamba
    ↓
跨模态区域交互 → 输出增强特征
```

#### 7. `models/vmamba/selective_scan/` - CUDA扩展

**作用**: Selective Scan的CUDA加速实现，提升计算效率。

**关键文件**:
- `setup.py`: 编译脚本，用于构建CUDA扩展
- `csrc/selective_scan/`: CUDA源码
  - `selective_scan_core.cu`: 核心CUDA实现
  - `selective_scan_fwd_kernel.cuh`: 前向传播kernel
  - `selective_scan_bwd_kernel.cuh`: 反向传播kernel
- `selective_scan/selective_scan_interface.py`: Python接口

**编译方法**:
```bash
cd models/vmamba/selective_scan
python setup.py build_ext --inplace
```

### 损失函数文件

#### `models/segmentation_loss.py` - 分割损失函数

**作用**: 实现各种分割任务常用的损失函数。

**主要类**:
- `DiceLoss`: Dice损失
  - 适用于二值分割，对小目标友好
  - 按样本计算，与CIPA保持一致

- `TverskyLoss`: Tversky损失 ⭐
  - 可控制假阴性(FN)和假阳性(FP)的权重
  - 适合肿瘤分割（更关注漏检）
  - 参数: alpha (FN权重), beta (FP权重)

- `CombinedSegLoss`: 组合损失
  - Dice + CrossEntropy的组合
  - 平衡重叠度和分类准确性

- `FocalLoss`: Focal损失
  - 关注难样本，降低易样本权重

- `IoULoss`: IoU损失
  - 直接优化IoU指标

**使用示例**:
```python
from models.segmentation_loss import TverskyLoss
loss_fn = TverskyLoss(alpha=0.7, beta=0.3, smooth=1e-6)
loss = loss_fn(pred, target)
```

### 辅助文件

#### `models/model.py` - 传统基线模型

**作用**: 包含VGG/UNet相关基线模型，保留用于对比实验。

**主要类**:
- `Vgg19_Encoder`: VGG19编码器（灰度输入）
  - 提取三层特征（64/128/256 channels）
- `UNetEncoder`: 基础UNet编码器
- `DualIndependentEncoderUNet`: 双独立编码器UNet（CT/PET各自编码器）

**注意**: 当前主模型为VMamba架构，此文件保留用于参考和对比。

#### `models/train_tasks.py` - 旧版训练代码

**作用**: 旧的训练循环实现，本项目已迁移到`main.py`。

**注意**: 当前训练使用`main.py`，此文件保留用于参考。

#### `models/measure_model.py` - 模型测量工具

**作用**: 统计模型的参数量和FLOPs。

**使用示例**:
```python
from models.measure_model import measure_model
params, flops = measure_model(model, input_size=(1, 3, 512, 512))
```

#### `models/P_loss.py` - 感知损失

**作用**: VGG19感知损失，用于图像重建任务。

**注意**: 本项目为分割任务，不使用此文件。

---

## 🔬 模型创新点

### 1. VMamba架构的优势

- **线性复杂度**: O(N) vs Transformer的O(N²)
- **长距离依赖**: SSM天然适合建模长序列
- **高效计算**: 适合高分辨率医疗图像(512×512)

### 2. 双独立编码器设计

- **模态特异性**: CT和PET独立学习各自特征
- **预训练利用**: 单通道→3通道，充分利用ImageNet预训练
- **特征互补**: 融合阶段实现互补增强

### 3. 多层级融合机制

- **CRM**: 通道级别的跨模态对齐
- **DCIM**: 区域级别的细粒度交互
- **Adaptive Fusion**: 自适应学习最优融合权重

### 4. 先进训练策略

- **EMA**: 提升模型稳定性
- **阈值扫描**: 自动寻找最佳阈值
- **混合精度**: 加速训练，节省显存

---

## 📊 评估指标

模型支持以下评估指标：

- **Dice系数**: 衡量重叠度，范围[0,1]，越大越好
- **IoU (Intersection over Union)**: 交并比，范围[0,1]
- **F1分数**: 精确率和召回率的调和平均
- **HD95 (95% Hausdorff距离)**: 边界精度，单位mm，越小越好
- **准确率**: 正确像素比例

---

## 🔧 超参数调优建议

### 学习率

- **初始学习率**: 6e-5 (VMamba Tiny推荐)
- **Warm-up**: 5 epochs
- **调度策略**: 余弦退火

### 批次大小

根据GPU显存选择：
- **4GB**: bs=2
- **8GB**: bs=4 (推荐)
- **16GB+**: bs=8

### 损失函数权重

- **Tversky权重**: 0.7 (推荐，更关注漏检)
- **BCE权重**: 0.3
- **Alpha/Beta**: 0.7/0.3 (肿瘤分割推荐)

### 数据增强

- **随机裁剪**: 512×512
- **随机翻转**: 水平/垂直
- **随机旋转**: ±15度
- **亮度/对比度调整**: ±20%

---

## 📦 依赖项

### 核心依赖

- **PyTorch**: >=2.1.0 (推荐CUDA 12.1+)
- **torchvision**: >=0.16.0
- **numpy**: >=1.24.0,<2.0.0
- **opencv-python**: >=4.8.0
- **pillow**: >=10.0.0

### VMamba相关

- **einops**: >=0.7.0 (张量操作)
- **timm**: >=0.9.0 (模型工具)
- **selective_scan**: CUDA扩展 (需编译)

### 训练工具

- **tensorboard**: >=2.14.0
- **tqdm**: >=4.66.0
- **albumentations**: >=1.3.0 (数据增强)
- **scipy**: >=1.11.0 (HD95计算)

完整依赖列表请查看 `requirements.txt`。

---

## 📝 最佳实践

1. ✅ **使用Tversky+BCE损失**: 适合肿瘤分割，控制漏检
2. ✅ **启用混合精度训练**: 加速训练，节省显存
3. ✅ **使用EMA**: 提升模型稳定性
4. ✅ **阈值扫描**: 验证时自动寻找最佳阈值
5. ✅ **监控HD95**: 边界精度的重要指标
6. ✅ **早停机制**: 防止过拟合
7. ✅ **固定随机种子**: 确保可复现

---

## 🤝 贡献

本项目基于VMamba和CIPA架构修改，欢迎提出改进建议。

## 📄 许可证

本项目遵循MIT许可证。

## 🙏 致谢

- **VMamba**: Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
- **CIPA**: 双模态医疗图像分割框架
- **Mamba**: Efficient Language Modeling with State Space Models

## 📧 联系方式

如有问题或建议，请提交Issue。

---

## 📚 参考文献

1. Liu, Y., et al. "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model." arXiv preprint (2024).
2. Gu, A., & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv preprint (2023).
3. CIPA: 双模态医疗图像分割框架 (内部项目)

---

**最后更新**: 2024-12-18  
**模型版本**: VMamba-Tiny + MambaDecoder  
**最佳性能**: Dice=0.7755 @ 阈值=0.35
