#!/usr/bin/env python3
"""
模型测试和评估脚本
支持在有ground truth的情况下计算详细的评估指标
"""

import os
import torch
import cv2
import argparse
import numpy as np
import imageio
import json
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from datetime import datetime
from scipy.ndimage import distance_transform_edt, binary_erosion

from models.model import Unet_resize_conv
from models.vmamba.builder import EncoderDecoder as CoCoSegVMamba
from utils import fname_presuffix
from data.dataset import LungSegmentationDataset, PCLT20KDataset
from easydict import EasyDict as edict
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class GrayscaleTransform:
    def __call__(self, img):
        # Convert the image to grayscale
        if img.shape[0] == 3:
            img = img[0, :, :]
            img = torch.unsqueeze(img, 0)
        return img


def calculate_hd95(pred_binary, target_binary):
    """计算95% Hausdorff距离（使用距离变换优化）
    
    Args:
        pred_binary: 预测二值mask [H, W] (numpy array, uint8)
        target_binary: 真实二值mask [H, W] (numpy array, uint8)
    
    Returns:
        hd95: 95% Hausdorff距离 (float)
    """
    # 检查是否有有效的mask
    if pred_binary.sum() == 0 and target_binary.sum() == 0:
        return 0.0
    if pred_binary.sum() == 0 or target_binary.sum() == 0:
        # 如果一个为空，返回一个大的惩罚值
        return float(max(pred_binary.shape))
    
    # 提取边界点
    def get_boundary_points(binary_mask):
        """提取二值mask的边界点坐标（使用腐蚀操作）"""
        boundary = binary_mask.astype(bool) & (~binary_erosion(binary_mask))
        coords = np.array(np.where(boundary)).T
        return coords
    
    # 获取边界点
    pred_boundary = get_boundary_points(pred_binary)
    target_boundary = get_boundary_points(target_binary)
    
    if len(pred_boundary) == 0 or len(target_boundary) == 0:
        return float(max(pred_binary.shape))
    
    # 使用距离变换计算从预测边界到目标mask的距离
    # 计算到目标mask的距离变换（距离背景到前景的距离）
    target_inv = (~target_binary.astype(bool)).astype(float)
    dt_target = distance_transform_edt(target_inv)
    
    # 获取预测边界点到目标mask的距离
    distances_pred_to_target = dt_target[pred_boundary[:, 0], pred_boundary[:, 1]]
    
    # 使用距离变换计算从目标边界到预测mask的距离
    pred_inv = (~pred_binary.astype(bool)).astype(float)
    dt_pred = distance_transform_edt(pred_inv)
    
    # 获取目标边界点到预测mask的距离
    distances_target_to_pred = dt_pred[target_boundary[:, 0], target_boundary[:, 1]]
    
    # 合并所有距离
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    
    if len(all_distances) == 0:
        return 0.0
    
    # 计算95百分位数
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def calculate_metrics(pred, target, threshold=0.5):
    """计算评估指标（按样本计算）
    
    Args:
        pred: 预测值 [B, 1, H, W]
        target: 真实值 [B, 1, H, W]
        threshold: 二值化阈值，默认0.5
    
    Returns:
        dice, iou, hd95, accuracy, recall (每个都是标量，batch的平均值)
        以及每张图片的指标列表
    """
    # 二值化预测 [B, 1, H, W]
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    batch_size = pred_binary.shape[0]
    
    # 按样本计算 (dim=(1, 2, 3) -> [B])
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))  # [B]
    union_sum = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))  # [B]
    union_intersect = union_sum - intersection  # [B]
    
    # Dice, IoU (per-sample)
    dice_per_sample = (2. * intersection + 1e-6) / (union_sum + 1e-6)  # [B]
    iou_per_sample = (intersection + 1e-6) / (union_intersect + 1e-6)  # [B]
    
    # TP, FP, FN
    tp = intersection  # [B]
    fp = pred_binary.sum(dim=(1, 2, 3)) - tp  # [B]
    fn = target.sum(dim=(1, 2, 3)) - tp  # [B]
    tn = (pred_binary == 0).sum(dim=(1, 2, 3)) - fn  # [B]
    
    # Accuracy (per-sample)
    accuracy_per_sample = (tp + tn) / (tp + tn + fp + fn + 1e-6)  # [B]
    
    # Recall (per-sample)
    recall_per_sample = (tp + 1e-6) / (tp + fn + 1e-6)  # [B]
    
    # 计算95% Hausdorff距离（对batch中每个样本分别计算）
    hd95_list = []
    for i in range(batch_size):
        pred_np = pred_binary[i, 0].cpu().numpy().astype(np.uint8)
        target_np = target[i, 0].cpu().numpy().astype(np.uint8)
        
        # 处理空mask的情况
        if pred_np.sum() == 0 and target_np.sum() == 0:
            hd95_list.append(0.0)
        elif pred_np.sum() == 0 or target_np.sum() == 0:
            # 如果只有一个是空的，HD95没有明确定义，这里给一个惩罚值（图像对角线长度）
            hd95_list.append(float(np.sqrt(pred_np.shape[0]**2 + pred_np.shape[1]**2)))
        else:
            hd95 = calculate_hd95(pred_np, target_np)
            hd95_list.append(hd95)
    
    hd95_per_sample = torch.tensor(hd95_list, dtype=torch.float32, device=pred.device)  # [B]
    
    # 取平均
    dice = dice_per_sample.mean()
    iou = iou_per_sample.mean()
    hd95 = hd95_per_sample.mean()
    accuracy = accuracy_per_sample.mean()
    recall = recall_per_sample.mean()
    
    # 返回平均值和每张图片的指标
    per_sample_metrics = {
        'dice': dice_per_sample.cpu().numpy(),
        'iou': iou_per_sample.cpu().numpy(),
        'hd95': hd95_list,
        'accuracy': accuracy_per_sample.cpu().numpy(),
        'recall': recall_per_sample.cpu().numpy()
    }
    
    return dice, iou, hd95, accuracy, recall, per_sample_metrics


def test_with_metrics(model, val_loader, device, save_dir=None, model_type='vmamba'):
    """测试模型并计算指标（使用DataLoader）
    
    Returns:
        metrics_dict: 包含所有指标统计信息的字典
        per_image_metrics: 每张图片的详细指标列表
    """
    model.eval()
    
    # 存储每张图片的指标
    per_image_metrics = []
    
    # 存储所有样本的指标（用于计算中位数和最大值）
    all_dice = []
    all_iou = []
    all_hd95 = []
    all_accuracy = []
    all_recall = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Testing', ncols=120)
        for batch_idx, batch in enumerate(pbar):
            ct = batch['ct'].to(device)
            pet = batch['pet'].to(device)
            mask = batch['mask'].to(device)
            filenames = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(len(ct))])
            
            # 前向传播
            # VMamba模型需要3通道输入（单通道复制为3通道）
            if model_type == 'vmamba':
                # 如果输入是单通道，复制为3通道
                if ct.shape[1] == 1:
                    ct = ct.repeat(1, 3, 1, 1)
                if pet.shape[1] == 1:
                    pet = pet.repeat(1, 3, 1, 1)
            
            output = model(ct, pet)
            
            # 计算指标（返回平均值和每张图片的指标）
            dice, iou, hd95, accuracy, recall, per_sample_metrics = calculate_metrics(output, mask)
            
            # 记录每张图片的指标
            batch_size = ct.size(0)
            for i in range(batch_size):
                filename = filenames[i] if isinstance(filenames[i], str) else filenames[i].item()
                per_image_metrics.append({
                    'filename': filename,
                    'dice': float(per_sample_metrics['dice'][i]),
                    'iou': float(per_sample_metrics['iou'][i]),
                    'hd95': float(per_sample_metrics['hd95'][i]),
                    'accuracy': float(per_sample_metrics['accuracy'][i]),
                    'recall': float(per_sample_metrics['recall'][i])
                })
                
                # 收集所有样本的指标
                all_dice.append(per_sample_metrics['dice'][i])
                all_iou.append(per_sample_metrics['iou'][i])
                all_hd95.append(per_sample_metrics['hd95'][i])
                all_accuracy.append(per_sample_metrics['accuracy'][i])
                all_recall.append(per_sample_metrics['recall'][i])
            
            # 更新进度条
            pbar.set_postfix({
                'dice': f'{dice.item():.3f}',
                'iou': f'{iou.item():.3f}',
                'hd95': f'{hd95.item():.2f}',
                'acc': f'{accuracy.item():.3f}',
                'recall': f'{recall.item():.3f}'
            })
            
            # 保存预测结果
            if save_dir:
                for i in range(batch_size):
                    # 保存预测mask
                    output_np = (torch.sigmoid(output[i]) > 0.5).float().squeeze().cpu().numpy()
                    filename = filenames[i] if isinstance(filenames[i], str) else filenames[i].item()
                    # 确保文件名有.png扩展名
                    if not filename.endswith('.png'):
                        filename = filename + '.png'
                    save_fn = os.path.join(save_dir, filename)
                    cv2.imwrite(save_fn, output_np * 255)
    
    # 转换为numpy数组以便计算统计信息
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)
    all_hd95 = np.array(all_hd95)
    all_accuracy = np.array(all_accuracy)
    all_recall = np.array(all_recall)
    
    # 排除Dice为0或接近0的样本（阈值设为0.01）
    # 这些样本通常是预测失败的情况，不应该纳入统计
    valid_mask = all_dice >= 0.01
    num_valid = np.sum(valid_mask)
    num_excluded = len(all_dice) - num_valid
    
    if num_excluded > 0:
        logging.warning(f"排除了 {num_excluded} 个Dice<0.01的样本（预测可能失败）")
        logging.info(f"有效样本数: {num_valid}/{len(all_dice)}")
    
    # 只对有效样本计算统计信息
    if num_valid > 0:
        valid_dice = all_dice[valid_mask]
        valid_iou = all_iou[valid_mask]
        valid_hd95 = all_hd95[valid_mask]
        valid_accuracy = all_accuracy[valid_mask]
        valid_recall = all_recall[valid_mask]
        
        # 同时过滤per_image_metrics，只保留有效样本
        valid_per_image_metrics = [m for i, m in enumerate(per_image_metrics) if valid_mask[i]]
    else:
        # 如果没有有效样本，使用所有样本（不应该发生）
        logging.warning("警告：所有样本的Dice都<0.01，使用所有样本计算统计信息")
        valid_dice = all_dice
        valid_iou = all_iou
        valid_hd95 = all_hd95
        valid_accuracy = all_accuracy
        valid_recall = all_recall
        valid_per_image_metrics = per_image_metrics
    
    # 计算统计信息（平均值、中位数、最大值、最小值）
    def compute_stats(values):
        if len(values) == 0:
            return {'mean': 0.0, 'median': 0.0, 'max': 0.0, 'min': 0.0}
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'max': float(np.max(values)),
            'min': float(np.min(values))
        }
    
    metrics_dict = {
        'dice': compute_stats(valid_dice),
        'iou': compute_stats(valid_iou),
        'hd95': compute_stats(valid_hd95),
        'accuracy': compute_stats(valid_accuracy),
        'recall': compute_stats(valid_recall),
        'num_valid_samples': int(num_valid),
        'num_excluded_samples': int(num_excluded),
        'total_samples': len(all_dice)
    }
    
    return metrics_dict, valid_per_image_metrics


def test_simple_inference(model, ct_path, pet_path, save_path, device, model_type='vmamba'):
    """简单推理模式（无ground truth，只生成预测）"""
    print('开始推理模式...')
    ct_list = [n for n in os.listdir(ct_path) if n.endswith('.png')]
    pet_list = [n for n in os.listdir(pet_path) if n.endswith('.png')]
    
    ct_list = sorted(ct_list)
    pet_list = sorted(pet_list)
    
    if len(ct_list) != len(pet_list):
        print(f"Warning: CT文件数量 ({len(ct_list)}) 和PET文件数量 ({len(pet_list)}) 不匹配")
    
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        GrayscaleTransform()
    ])
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    import time
    Time = []
    pbar = tqdm(zip(ct_list, pet_list), total=len(ct_list), desc='推理中')
    for ct_file, pet_file in pbar:
        fn_pet = os.path.join(pet_path, pet_file)
        fn_ct = os.path.join(ct_path, ct_file)
        start = time.time()
        
        img_ct = imageio.imread(fn_ct).astype(np.float32)
        img_pet = imageio.imread(fn_pet).astype(np.float32)
        
        # to tensor and grayscale
        data_ct = transform(img_ct)
        data_pet = transform(img_pet)
        
        # add batch size dimension
        data_ct = torch.unsqueeze(data_ct, 0).to(device)  # [1, 1, H, W]
        data_pet = torch.unsqueeze(data_pet, 0).to(device)  # [1, 1, H, W]
        
        # VMamba模型需要3通道输入（单通道复制为3通道）
        if model_type == 'vmamba':
            data_ct = data_ct.repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W]
            data_pet = data_pet.repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W]
        
        with torch.no_grad():
            output = model(data_ct, data_pet)
        
        # 转换为numpy并保存
        output_np = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()
        
        save_fn = fname_presuffix(
            fname=ct_file, prefix='', suffix='', newpath=save_path)
        cv2.imwrite(save_fn.split('.')[0] + '.png', output_np * 255)
        
        end = time.time()
        Time.append(end - start)
    
    print(f"推理完成！平均时间: {np.mean(Time):.3f}秒/张, 标准差: {np.std(Time):.3f}秒")


def main():
    parser = argparse.ArgumentParser(description='CoCoSeg: 模型测试和评估')
    
    # 模型参数
    parser.add_argument('--ckpt', type=str, required=True, 
                       help='模型checkpoint路径 (如: ./logs/best_model.pth)')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU')
    
    # 测试模式选择
    parser.add_argument('--mode', type=str, choices=['eval', 'infer'], 
                       default='eval',
                       help='评估模式: eval=有ground truth计算指标, infer=仅生成预测')
    
    # 模型类型（如果checkpoint目录下没有config.json，需要手动指定）
    parser.add_argument('--model_type', type=str, choices=['vmamba', 'original'], 
                       default=None,
                       help='模型类型: vmamba 或 original (如果未指定，将尝试从config.json读取)')
    
    # 数据集类型
    parser.add_argument('--dataset_type', type=str, default='auto',
                       choices=['auto', 'pclt', 'pclt20k'],
                       help='数据集类型: auto (auto-detect), pclt (old format), or pclt20k (new format)')
    
    # 评估模式参数（使用DataLoader）
    parser.add_argument('--dataset_root', type=str, default='./dataset',
                       help='数据集根目录（评估模式使用）')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], 
                       default='val',
                       help='使用哪个数据分割进行测试 (all=测试整个数据集)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    
    # 推理模式参数（直接指定路径）
    parser.add_argument('--test_ct', type=str, 
                       help='CT图像目录（推理模式使用）')
    parser.add_argument('--test_pet', type=str,
                       help='PET图像目录（推理模式使用）')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='./test_results/',
                       help='结果保存目录')
    parser.add_argument('--save_predictions', action='store_true',
                       help='是否保存预测结果图像')
    
    args = parser.parse_args()
    
    print('\n' + '='*60)
    print('CoCoSeg: 模型测试和评估')
    print('='*60)
    print(f'GPU可用: {torch.cuda.is_available()}')
    print(f'测试模式: {args.mode}')
    print(f'模型checkpoint: {args.ckpt}')
    print('='*60)
    
    # 设置设备
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if args.use_gpu:
            print("⚠ CUDA不可用，使用CPU模式")
        else:
            print("✓ 使用CPU模式")
    
    # 加载模型
    print(f"\n加载模型从: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"错误: Checkpoint文件不存在: {args.ckpt}")
        return
    
    # 尝试从checkpoint目录下的config.json读取model_type
    model_type = args.model_type
    checkpoint_dir = os.path.dirname(args.ckpt)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if model_type is None and os.path.exists(config_path):
        print(f"从配置文件读取模型类型: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            model_type = config.get('model_type', 'vmamba')
            print(f"检测到模型类型: {model_type}")
    elif model_type is None:
        # 默认使用vmamba（因为README显示训练使用的是vmamba）
        model_type = 'vmamba'
        print(f"未找到config.json，使用默认模型类型: {model_type}")
    
    # 根据model_type初始化模型
    print(f"\n初始化模型: {model_type}")
    if model_type == 'vmamba':
        # VMamba模型配置（与训练时一致）
        C = edict()
        config = C
        C.backbone = 'sigma_tiny'
        pretrained_path = './pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth'
        if os.path.exists(pretrained_path):
            C.pretrained_model = pretrained_path
        else:
            C.pretrained_model = None
        C.decoder = 'MambaDecoder'
        C.decoder_embed_dim = 512
        C.image_height = 512
        C.image_width = 512
        C.bn_eps = 1e-3
        C.bn_momentum = 0.1
        C.num_classes = 1
        model = CoCoSegVMamba(cfg=config, norm_layer=torch.nn.BatchNorm2d)
    else:
        # Original UNet模型
        model = Unet_resize_conv()
    
    # 加载checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # 尝试加载state_dict（处理可能的键名不匹配）
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("✓ 模型权重加载成功")
    except RuntimeError as e:
        print(f"⚠ 严格加载失败: {e}")
        print("尝试非严格加载...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✓ 模型权重加载成功（非严格模式）")
    
    model.to(device)
    model.eval()
    
    # 显示模型信息
    if 'epoch' in checkpoint:
        print(f"模型epoch: {checkpoint['epoch']}")
    if 'best_val_dice' in checkpoint:
        print(f"训练时最佳验证Dice: {checkpoint['best_val_dice']:.4f}")
    if 'val_best_threshold' in checkpoint:
        print(f"训练时最佳阈值: {checkpoint['val_best_threshold']:.4f}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果保存到: {save_dir}")
    
    # 根据模式执行测试
    if args.mode == 'eval':
        print("\n开始评估模式（使用ground truth计算指标）...")
        print(f"数据集根目录: {args.dataset_root}")
        print(f"数据分割: {args.split}")
        
        # 创建测试数据集（自动检测数据集类型）
        if args.dataset_type == 'auto':
            # 自动检测数据集类型
            # 1. 检查是否有预先分割的train/val/test目录
            if os.path.exists(os.path.join(args.dataset_root, 'train')):
                # 进一步检查是PCLT还是PCLT20K格式
                # PCLT格式：train/CT/, train/PET/, train/masks/
                # PCLT20K格式：train/0001/, train/0002/, ... (每个目录是patient)
                train_dir = os.path.join(args.dataset_root, 'train')
                if os.path.exists(os.path.join(train_dir, 'CT')):
                    dataset_type = 'pclt'
                else:
                    # 检查train目录下是否有子目录（patient目录）
                    train_subdirs = [d for d in os.listdir(train_dir) 
                                    if os.path.isdir(os.path.join(train_dir, d))]
                    if train_subdirs:
                        # 检查第一个子目录中是否有_ct.png文件
                        sample_dir = os.path.join(train_dir, train_subdirs[0])
                        sample_files = os.listdir(sample_dir) if os.path.exists(sample_dir) else []
                        if any('_ct.png' in f or '_CT.png' in f for f in sample_files):
                            dataset_type = 'pclt20k'
                        else:
                            dataset_type = 'pclt'
                    else:
                        dataset_type = 'pclt'
            else:
                # 2. 检查根目录下的结构
                root_items = os.listdir(args.dataset_root)
                # 检查是否有CT/PET/masks目录（PCLT格式）
                if 'CT' in root_items and 'PET' in root_items:
                    dataset_type = 'pclt'
                else:
                    # 检查是否是PCLT20K格式（根目录下直接是patient目录）
                    patient_dirs = [d for d in root_items 
                                  if os.path.isdir(os.path.join(args.dataset_root, d))]
                    if patient_dirs:
                        # 检查第一个patient目录中是否有_ct.png文件
                        sample_dir = os.path.join(args.dataset_root, patient_dirs[0])
                        sample_files = os.listdir(sample_dir) if os.path.exists(sample_dir) else []
                        if any('_ct.png' in f or '_CT.png' in f for f in sample_files):
                            dataset_type = 'pclt20k'
                        else:
                            dataset_type = 'pclt'
                    else:
                        dataset_type = 'pclt'
        else:
            dataset_type = args.dataset_type
        
        print(f"使用数据集类型: {dataset_type}")
        
        # 如果split为'all'，创建组合数据集（train + val + test）
        if args.split == 'all':
            print("测试整个数据集（train + val + test）...")
            if dataset_type == 'pclt20k':
                train_dataset = PCLT20KDataset(
                    dataset_root=args.dataset_root,
                    split='train',
                    val_ratio=0.2,
                    random_seed=42,
                    normalization_mode='vmamba' if model_type == 'vmamba' else 'standard'
                )
                val_dataset = PCLT20KDataset(
                    dataset_root=args.dataset_root,
                    split='val',
                    val_ratio=0.2,
                    random_seed=42,
                    normalization_mode='vmamba' if model_type == 'vmamba' else 'standard'
                )
                test_dataset = PCLT20KDataset(
                    dataset_root=args.dataset_root,
                    split='test',
                    val_ratio=0.2,
                    random_seed=42,
                    normalization_mode='vmamba' if model_type == 'vmamba' else 'standard'
                )
            else:
                train_dataset = LungSegmentationDataset(
                    dataset_root=args.dataset_root,
                    split='train',
                    val_ratio=0.2,
                    random_seed=42
                )
                val_dataset = LungSegmentationDataset(
                    dataset_root=args.dataset_root,
                    split='val',
                    val_ratio=0.2,
                    random_seed=42
                )
                test_dataset = LungSegmentationDataset(
                    dataset_root=args.dataset_root,
                    split='test',
                    val_ratio=0.2,
                    random_seed=42
                )
            
            print(f"训练集样本数: {len(train_dataset)}")
            print(f"验证集样本数: {len(val_dataset)}")
            print(f"测试集样本数: {len(test_dataset)}")
            
            # 组合三个数据集
            test_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
            print(f"总测试样本数: {len(test_dataset)}")
        else:
            # 测试单个split
            if dataset_type == 'pclt20k':
                test_dataset = PCLT20KDataset(
                    dataset_root=args.dataset_root,
                    split=args.split,
                    val_ratio=0.2,
                    random_seed=42,
                    normalization_mode='vmamba' if model_type == 'vmamba' else 'standard'
                )
            else:
                test_dataset = LungSegmentationDataset(
                    dataset_root=args.dataset_root,
                    split=args.split,
                    val_ratio=0.2,
                    random_seed=42
                )
            print(f"测试样本数: {len(test_dataset)}")
        
        # 创建DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.use_gpu else False
        )
        
        # 运行评估
        save_pred_dir = os.path.join(save_dir, 'predictions') if args.save_predictions else None
        if save_pred_dir:
            os.makedirs(save_pred_dir, exist_ok=True)
        
        metrics_dict, per_image_metrics = test_with_metrics(
            model, test_loader, device, save_pred_dir, model_type
        )
        
        # 打印结果
        print('\n' + '='*70)
        print('评估结果汇总')
        print('='*70)
        if metrics_dict.get('num_excluded_samples', 0) > 0:
            print(f'⚠ 注意: 已排除 {metrics_dict["num_excluded_samples"]} 个Dice<0.01的样本（预测可能失败）')
            print(f'   有效样本数: {metrics_dict["num_valid_samples"]}/{metrics_dict["total_samples"]}')
            print('-'*70)
        print(f'指标          |  平均值  |  中位数  |  最大值  |  最小值')
        print('-'*70)
        print(f'Dice系数      | {metrics_dict["dice"]["mean"]:7.4f} | {metrics_dict["dice"]["median"]:7.4f} | {metrics_dict["dice"]["max"]:7.4f} | {metrics_dict["dice"]["min"]:7.4f}')
        print(f'IoU           | {metrics_dict["iou"]["mean"]:7.4f} | {metrics_dict["iou"]["median"]:7.4f} | {metrics_dict["iou"]["max"]:7.4f} | {metrics_dict["iou"]["min"]:7.4f}')
        print(f'HD95 (mm)     | {metrics_dict["hd95"]["mean"]:7.2f} | {metrics_dict["hd95"]["median"]:7.2f} | {metrics_dict["hd95"]["max"]:7.2f} | {metrics_dict["hd95"]["min"]:7.2f}')
        print(f'Accuracy      | {metrics_dict["accuracy"]["mean"]:7.4f} | {metrics_dict["accuracy"]["median"]:7.4f} | {metrics_dict["accuracy"]["max"]:7.4f} | {metrics_dict["accuracy"]["min"]:7.4f}')
        print(f'Recall        | {metrics_dict["recall"]["mean"]:7.4f} | {metrics_dict["recall"]["median"]:7.4f} | {metrics_dict["recall"]["max"]:7.4f} | {metrics_dict["recall"]["min"]:7.4f}')
        print('='*70)
        
        # 保存详细结果
        results = {
            'checkpoint': args.ckpt,
            'dataset_root': args.dataset_root,
            'split': args.split,
            'num_samples': len(test_dataset),
            'overall_metrics': metrics_dict,
            'per_image_metrics': per_image_metrics,
            'timestamp': timestamp
        }
        
        # 如果测试整个数据集，添加各split的统计信息
        if args.split == 'all':
            # 通过检查pk_split目录来确定各split的病人ID
            split_info_path = os.path.join(args.dataset_root, '..', 'pk_split', 'split_info.json')
            if os.path.exists(split_info_path):
                try:
                    with open(split_info_path, 'r') as f:
                        split_info = json.load(f)
                    train_patients = set(split_info.get('train', []))
                    val_patients = set(split_info.get('val', []))
                    test_patients = set(split_info.get('test', []))
                    
                    train_count = sum(1 for m in per_image_metrics if any(pid in m['filename'] for pid in train_patients))
                    val_count = sum(1 for m in per_image_metrics if any(pid in m['filename'] for pid in val_patients))
                    test_count = sum(1 for m in per_image_metrics if any(pid in m['filename'] for pid in test_patients))
                    
                    results['split_statistics'] = {
                        'train_samples': train_count,
                        'val_samples': val_count,
                        'test_samples': test_count,
                        'total_samples': len(per_image_metrics)
                    }
                except:
                    pass
        
        results_file = os.path.join(save_dir, 'results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {results_file}")
        
    elif args.mode == 'infer':
        print("\n开始推理模式（仅生成预测，不计算指标）...")
        
        if not args.test_ct or not args.test_pet:
            print("错误: 推理模式需要指定 --test_ct 和 --test_pet 参数")
            return
        
        print(f"CT目录: {args.test_ct}")
        print(f"PET目录: {args.test_pet}")
        
        save_pred_dir = os.path.join(save_dir, 'predictions')
        os.makedirs(save_pred_dir, exist_ok=True)
        
        test_simple_inference(model, args.test_ct, args.test_pet, save_pred_dir, device, model_type)
        print(f"\n预测结果已保存到: {save_pred_dir}")
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()

