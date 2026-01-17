#!/usr/bin/env python
"""
按病人ID生成数据集分割txt文件脚本

功能：
1. 扫描数据集目录，获取所有病人ID
2. 按病人ID分割成训练集、验证集、测试集（8:1:1比例）
3. 生成train.txt、val.txt和test.txt文件（包含该病人的所有切片）

用法：
    python generate_split_txt.py --data_dir /root/autodl-tmp/CIPA/data/PCLT20k/
"""

import os
import argparse
import random
from collections import defaultdict
from pathlib import Path


def get_all_patients(data_dir):
    """获取所有病人ID和对应的切片列表
    
    Args:
        data_dir: 数据集根目录（包含病人文件夹）
        
    Returns:
        patient_dict: {patient_id: [slice_list]} 字典
    """
    patient_dict = defaultdict(list)
    
    # 遍历所有病人文件夹
    for patient_dir in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_dir)
        
        # 跳过非目录和txt文件
        if not os.path.isdir(patient_path) or patient_dir.endswith('.txt'):
            continue
        
        patient_id = patient_dir
        
        # 获取该病人的所有切片文件（通过查找_ct.png或_CT.png文件，支持大小写）
        if os.path.isdir(patient_path):
            for filename in os.listdir(patient_path):
                filename_lower = filename.lower()
                # 支持 _ct.png 或 _CT.png（大小写不敏感）
                if filename_lower.endswith('_ct.png'):
                    # 提取切片ID（格式：病人ID_切片编号）
                    # 移除后缀，保留原始大小写
                    if filename.endswith('_CT.png'):
                        slice_id = filename.replace('_CT.png', '')
                    else:
                        slice_id = filename.replace('_ct.png', '')
                    
                    # 验证对应的pet和mask文件是否存在（尝试多种大小写组合）
                    base_path = os.path.join(patient_path, slice_id)
                    # 尝试不同的后缀组合
                    pet_variants = [f"{base_path}_PET.png", f"{base_path}_pet.png"]
                    mask_variants = [f"{base_path}_mask.png", f"{base_path}_Mask.png"]
                    
                    pet_exists = any(os.path.exists(v) for v in pet_variants)
                    mask_exists = any(os.path.exists(v) for v in mask_variants)
                    
                    if pet_exists and mask_exists:
                        patient_dict[patient_id].append(slice_id)
        
        # 按切片编号排序
        patient_dict[patient_id].sort()
    
    return patient_dict


def split_patients(patient_dict, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """按病人ID分割数据集（8:1:1比例）
    
    Args:
        patient_dict: {patient_id: [slice_list]} 字典
        train_ratio: 训练集比例（默认0.8）
        val_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.1）
        random_seed: 随机种子
        
    Returns:
        train_slices: 训练集切片列表
        val_slices: 验证集切片列表
        test_slices: 测试集切片列表
        train_patients: 训练集病人ID列表
        val_patients: 验证集病人ID列表
        test_patients: 测试集病人ID列表
    """
    patient_ids = sorted(patient_dict.keys())
    
    # 固定随机种子
    random.seed(random_seed)
    random.shuffle(patient_ids)
    
    total_patients = len(patient_ids)
    
    # 按8:1:1比例分割
    train_end = int(total_patients * train_ratio)
    val_end = train_end + int(total_patients * val_ratio)
    
    train_patients = patient_ids[:train_end]
    val_patients = patient_ids[train_end:val_end]
    test_patients = patient_ids[val_end:]
    
    # 收集所有切片
    train_slices = []
    for patient_id in train_patients:
        train_slices.extend(patient_dict[patient_id])
    
    val_slices = []
    for patient_id in val_patients:
        val_slices.extend(patient_dict[patient_id])
    
    test_slices = []
    for patient_id in test_patients:
        test_slices.extend(patient_dict[patient_id])
    
    return train_slices, val_slices, test_slices, train_patients, val_patients, test_patients


def save_split_files(output_dir, train_slices, val_slices, test_slices):
    """保存分割文件到train.txt、val.txt和test.txt
    
    Args:
        output_dir: 输出目录
        train_slices: 训练集切片列表
        val_slices: 验证集切片列表
        test_slices: 测试集切片列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w') as f:
        for slice_id in train_slices:
            f.write(f"{slice_id}\n")
    print(f"✓ 训练集: {len(train_slices)} 个切片 -> {train_file}")
    
    # 保存验证集
    val_file = os.path.join(output_dir, 'val.txt')
    with open(val_file, 'w') as f:
        for slice_id in val_slices:
            f.write(f"{slice_id}\n")
    print(f"✓ 验证集: {len(val_slices)} 个切片 -> {val_file}")
    
    # 保存测试集
    test_file = os.path.join(output_dir, 'test.txt')
    with open(test_file, 'w') as f:
        for slice_id in test_slices:
            f.write(f"{slice_id}\n")
    print(f"✓ 测试集: {len(test_slices)} 个切片 -> {test_file}")


def main():
    parser = argparse.ArgumentParser(description='按病人ID生成数据集分割txt文件（8:1:1比例）')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据集根目录（包含病人文件夹）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认0.8）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例（默认0.1）')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例（默认0.1）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子（默认42）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认与data_dir相同）')
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"警告: 比例总和为 {total_ratio:.2f}，不等于1.0，将自动归一化")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    print("=" * 60)
    print("数据集分割脚本 - 按病人ID分割（8:1:1比例）")
    print("=" * 60)
    print(f"数据集目录: {args.data_dir}")
    print(f"分割比例: 训练集={args.train_ratio:.1%}, 验证集={args.val_ratio:.1%}, 测试集={args.test_ratio:.1%}")
    print(f"随机种子: {args.random_seed}")
    print("=" * 60)
    
    # 获取所有病人
    print(f"\n正在扫描数据集目录...")
    patient_dict = get_all_patients(args.data_dir)
    print(f"✓ 找到 {len(patient_dict)} 个病人")
    
    # 统计切片数量
    total_slices = sum(len(slices) for slices in patient_dict.values())
    print(f"✓ 总共 {total_slices} 个切片")
    
    # 分割数据集
    print(f"\n正在按病人ID分割数据集...")
    train_slices, val_slices, test_slices, train_patients, val_patients, test_patients = \
        split_patients(patient_dict, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed)
    
    print(f"\n分割结果:")
    print(f"  训练集: {len(train_patients)} 个病人, {len(train_slices)} 个切片")
    print(f"  验证集: {len(val_patients)} 个病人, {len(val_slices)} 个切片")
    print(f"  测试集: {len(test_patients)} 个病人, {len(test_slices)} 个切片")
    
    # 显示部分病人ID
    print(f"\n病人ID分布:")
    print(f"  训练集病人ID示例: {train_patients[:5]}{'...' if len(train_patients) > 5 else ''}")
    print(f"  验证集病人ID示例: {val_patients[:5]}{'...' if len(val_patients) > 5 else ''}")
    print(f"  测试集病人ID示例: {test_patients[:5]}{'...' if len(test_patients) > 5 else ''}")
    
    # 保存分割文件
    output_dir = args.output_dir if args.output_dir else args.data_dir
    print(f"\n正在保存分割文件到: {output_dir}")
    save_split_files(output_dir, train_slices, val_slices, test_slices)
    
    print(f"\n{'=' * 60}")
    print("✓ 分割完成！")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

