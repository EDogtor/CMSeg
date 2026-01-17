import h5py
from torch.utils.data import Dataset
from os.path import splitext, join, exists
from os import listdir
import os
import numpy as np
from glob import glob
import torch
import logging
from PIL import Image
import random
from collections import defaultdict
import albumentations as A
import cv2


class TrainDataSet(Dataset):
    """旧版本的H5数据集加载器（保留用于兼容）"""
    def __init__(self, dataset=None, arg=None):
        super(TrainDataSet, self).__init__()
        self.arg = arg
        data = h5py.File(dataset, 'r')
        data = data['data'][:]
        np.random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = ((self.data[idx] - 0.5) / 0.5).astype(np.float32)
        return data


class LungSegmentationDataset(Dataset):
    """肺癌CT-PET-Mask分割数据集
    
    支持两种数据格式：
    1. 预先分割的数据集（推荐）:
       - dataset_root/train/CT/, train/PET/, train/masks/
       - dataset_root/val/CT/, val/PET/, val/masks/
       - dataset_root/test/CT/, test/PET/, test/masks/
    
    2. 未分割的数据集（兼容模式）:
       - dataset_root/CT/patient_id_slice_001.png
       - dataset_root/PET/patient_id_slice_001.png
       - dataset_root/masks/patient_id_slice_001.png
    
    按病人ID分割，避免数据泄漏
    """
    
    def __init__(self, dataset_root, split='train', val_ratio=0.2, random_seed=42, 
                 scale=1.0, augment=False, normalization_mode='standard'):
        """
        Args:
            dataset_root: 数据集根目录
            split: 'train' 或 'val' 或 'test'
            val_ratio: 验证集比例
            random_seed: 随机种子
            scale: 图像缩放比例（0-1）
            augment: 是否进行数据增强
            normalization_mode: 'standard' ([-1, 1]) 或 'vmamba' ([-1.6, 1.6])
        """
        super(LungSegmentationDataset, self).__init__()
        
        self.scale = scale
        self.augment = augment
        self.normalization_mode = normalization_mode
        
        # 初始化增强管道
        if self.augment:
            self.transform = self._get_augmentation_pipeline()
            logging.info("数据增强已启用")
        else:
            self.transform = None
        
        # 检查是否为预先分割的数据集
        split_dir = join(dataset_root, split)
        if exists(split_dir) and os.path.isdir(split_dir):
            # 预先分割模式：直接使用对应split目录
            logging.info(f"使用预先分割的数据集模式: {split_dir}")
            self.ct_dir = join(split_dir, 'CT')
            self.pet_dir = join(split_dir, 'PET')
            self.mask_dir = join(split_dir, 'masks')
            self._load_pre_split_data()
        else:
            # 运行时分割模式：从根目录分割
            logging.info(f"使用运行时分割模式: {dataset_root}")
            self.ct_dir = join(dataset_root, 'CT')
            self.pet_dir = join(dataset_root, 'PET')
            self.mask_dir = join(dataset_root, 'masks')
            self._load_and_split_data(split, val_ratio, random_seed)
    
    def _load_pre_split_data(self):
        """加载预先分割的数据集"""
        # 获取所有图像文件
        if not exists(self.ct_dir):
            raise FileNotFoundError(f"CT目录不存在: {self.ct_dir}")
        
        ct_files = [f for f in listdir(self.ct_dir) 
                   if f.endswith('.png') and not f.startswith('.')]
        
        # 构建数据列表
        self.data_list = []
        for ct_file in ct_files:
            ct_path = join(self.ct_dir, ct_file)
            pet_path = join(self.pet_dir, ct_file)
            mask_path = join(self.mask_dir, ct_file)
            
            # 验证文件是否存在
            if not glob(pet_path):
                logging.warning(f"PET file not found: {pet_path}")
                continue
            if not glob(mask_path):
                logging.warning(f"Mask file not found: {mask_path}")
                continue
            
            self.data_list.append({
                'ct': ct_path,
                'pet': pet_path,
                'mask': mask_path,
                'name': splitext(ct_file)[0]
            })
        
        logging.info(f"Loaded {len(self.data_list)} samples from pre-split dataset")
    
    def _load_and_split_data(self, split, val_ratio, random_seed):
        """加载并运行时分割数据集（向后兼容）"""
        # 获取所有图像文件并提取病人ID
        if not exists(self.ct_dir):
            raise FileNotFoundError(f"CT目录不存在: {self.ct_dir}")
        
        ct_files = [f for f in listdir(self.ct_dir) 
                   if f.endswith('.png') and not f.startswith('.')]
        
        # 按病人ID分组
        patient_dict = defaultdict(list)
        for ct_file in ct_files:
            # 假设命名格式: patient_id_slice_001.png
            filename = splitext(ct_file)[0]
            if '_slice_' in filename:
                patient_id = filename.rsplit('_slice_', 1)[0]
            else:
                # 如果没有slice编号，整个文件名作为patient_id
                patient_id = filename.split('_')[0]
            patient_dict[patient_id].append(ct_file)
        
        # 按病人ID排序并分割数据集
        patient_ids = sorted(patient_dict.keys())
        
        # 固定随机种子以确保可重复性
        random.seed(random_seed)
        random.shuffle(patient_ids)
        
        # 按比例分割
        total_patients = len(patient_ids)
        val_split_idx = int(total_patients * (1 - val_ratio))
        
        if split == 'train':
            self.patient_ids = patient_ids[:val_split_idx]
        elif split == 'val':
            self.patient_ids = patient_ids[val_split_idx:]
        elif split == 'test':
            # 如果没有test目录，将test等同于val
            logging.warning("运行时分割模式不支持test，使用val代替")
            self.patient_ids = patient_ids[val_split_idx:]
        else:
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split}")
        
        # 为选中的病人收集所有切片
        self.data_list = []
        for patient_id in self.patient_ids:
            for ct_file in patient_dict[patient_id]:
                ct_path = join(self.ct_dir, ct_file)
                pet_file = ct_file.replace('.png', '.png')  # CT和PET文件名相同
                pet_path = join(self.pet_dir, pet_file)
                mask_path = join(self.mask_dir, ct_file)
                
                # 验证文件是否存在
                if not glob(pet_path):
                    logging.warning(f"PET file not found: {pet_path}")
                    continue
                if not glob(mask_path):
                    logging.warning(f"Mask file not found: {mask_path}")
                    continue
                
                self.data_list.append({
                    'ct': ct_path,
                    'pet': pet_path,
                    'mask': mask_path,
                    'name': splitext(ct_file)[0]
                })
        
        logging.info(f"Created {split} dataset with {len(self.patient_ids)} patients, "
                    f"{len(self.data_list)} slices")
        logging.info(f"Patient IDs in {split} set: {self.patient_ids[:5]}..." 
                    if len(self.patient_ids) > 5 else f"Patient IDs: {self.patient_ids}")
    
    def __len__(self):
        return len(self.data_list)
    
    def _get_augmentation_pipeline(self):
        """获取数据增强管道
        
        与CIPA保持一致的数据增强策略：
        1. 几何变换（同步）
        2. 弹性形变（同步）
        3. 像素/噪声变换（仅CT和PET）
        
        注意：随机裁剪（randomcrop）将在__getitem__中单独实现，以匹配CIPA的行为
        """
        # 使用additional_targets定义多个输入
        return A.Compose([
            # 1. 几何变换 - 应对位置、角度变化（所有图像同步）
            A.HorizontalFlip(p=0.5),  # 50%概率水平翻转
            A.VerticalFlip(p=0.5),    # 50%概率垂直翻转
            A.Affine(
                rotate=(-15, 15),     # ±15度旋转
                translate_percent=(0, 0.05),  # 5%平移
                scale=(0.9, 1.1),     # 90%-110%缩放
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                fit_output=False      # 不调整输出尺寸
            ),
            
            # 2. 弹性形变 - 对抗过拟合的利器（所有图像同步）
            A.ElasticTransform(
                p=0.3,
                alpha=120,
                sigma=120 * 0.05,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            
            # 3. 像素/噪声变换 - 应对CT/PET成像差异（只对CT和PET应用）
            A.RandomBrightnessContrast(
                brightness_limit=0.1,   # ±10%亮度
                contrast_limit=0.1,     # ±10%对比度
                p=0.3
            ),
            A.GaussNoise(
                std_range=(0.04, 0.2),  # 高斯噪声标准差范围（归一化到[0,1]）
                p=0.2
            ),
        ], additional_targets={'image1': 'image'})  # 定义PET作为image1，与image（CT）同步变换
    
    def _random_crop_cipa_style(self, ct_hwc, pet_hwc, mask_hwc, u=0.5):
        """CIPA风格的随机裁剪：70%-90%裁剪后resize回原尺寸
        
        Args:
            ct_hwc: CT图像 [H, W]
            pet_hwc: PET图像 [H, W]
            mask_hwc: Mask图像 [H, W]
            u: 应用概率，默认0.5
        
        Returns:
            ct_cropped, pet_cropped, mask_cropped
        """
        if np.random.random() >= u:
            return ct_hwc, pet_hwc, mask_hwc
        
        h, w = ct_hwc.shape[:2]
        crop_rate = np.random.uniform(0.7, 0.9)  # CIPA: 70%-90%
        crop_h = int(h * crop_rate)
        crop_w = int(w * crop_rate)
        
        # 随机选择裁剪起始位置
        y = np.random.randint(0, h - crop_h + 1)
        x = np.random.randint(0, w - crop_w + 1)
        
        # 裁剪
        ct_crop = ct_hwc[y:y+crop_h, x:x+crop_w]
        pet_crop = pet_hwc[y:y+crop_h, x:x+crop_w]
        mask_crop = mask_hwc[y:y+crop_h, x:x+crop_w]
        
        # Resize回原尺寸（CIPA使用CUBIC插值）
        ct_cropped = cv2.resize(ct_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        pet_cropped = cv2.resize(pet_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        mask_cropped = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return ct_cropped, pet_cropped, mask_cropped
    
    @staticmethod
    def preprocess(img, scale=1.0, is_mask=False):
        """预处理图像"""
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 如果是单通道灰度图，增加通道维度
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        # 缩放（图片和 mask 都要按各自的插值方式缩放）
        if scale != 1.0:
            from PIL import Image
            new_w = int(img_array.shape[1] * scale)
            new_h = int(img_array.shape[0] * scale)
            if is_mask:
                # 最近邻避免产生灰阶
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.NEAREST)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
            else:
                # 双线性插值用于图像
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.BILINEAR)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
        
        # 转换为CHW格式
        img_trans = img_array.transpose((2, 0, 1))
        
        # 归一化（mask不需要归一化）
        if not is_mask and img_trans.max() > 1:
            img_trans = img_trans / 255.0
        
        return img_trans
    
    def __getitem__(self, idx):
        """获取单个样本"""
        data_info = self.data_list[idx]
        
        # 加载图像
        try:
            ct_img = Image.open(data_info['ct']).convert('L')
            pet_img = Image.open(data_info['pet']).convert('L')
            mask_img = Image.open(data_info['mask']).convert('L')
        except Exception as e:
            logging.error(f"Error loading images: {data_info['name']}, {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self.data_list))
        
        # 验证图像尺寸一致性
        assert ct_img.size == pet_img.size == mask_img.size, \
            f"Image sizes mismatch for {data_info['name']}"
        
        # 预处理 (得到 [0, 1] 的 float32)
        ct_array = self.preprocess(ct_img, self.scale, is_mask=False)
        pet_array = self.preprocess(pet_img, self.scale, is_mask=False)
        mask_array = self.preprocess(mask_img, self.scale, is_mask=True)
        
        # 归一化
        if self.normalization_mode == 'vmamba':
            # VMamba 预训练权重的归一化方式: x * 3.2 - 1.6 (from CIPA)
            # 原图是 [0, 1], 变换后范围 [-1.6, 1.6]
            ct_array = ct_array * 3.2 - 1.6
            pet_array = pet_array * 3.2 - 1.6
        else:
            # 标准归一化: (x - 0.5) / 0.5 (from CoCoSeg)
            # 原图是 [0, 1], 变换后范围 [-1, 1]
            ct_array = (ct_array - 0.5) / 0.5
            pet_array = (pet_array - 0.5) / 0.5
        
        # Mask二值化处理
        mask_array = (mask_array > 0.5).astype(np.float32)
        
        # 应用数据增强（如果需要）
        if self.transform is not None:
            # 将CHW转换为HWC以便Albumentations处理
            ct_hwc = ct_array.transpose(1, 2, 0).squeeze()  # (H, W)
            pet_hwc = pet_array.transpose(1, 2, 0).squeeze()  # (H, W)
            mask_hwc = mask_array.transpose(1, 2, 0).squeeze()  # (H, W)
            
            # 转回 [0, 255] uint8
            if self.normalization_mode == 'vmamba':
                # [-1.6, 1.6] -> [0, 1] -> [0, 255]
                ct_hwc_uint8 = np.clip(((ct_hwc + 1.6) / 3.2 * 255.0), 0, 255).astype(np.uint8)
                pet_hwc_uint8 = np.clip(((pet_hwc + 1.6) / 3.2 * 255.0), 0, 255).astype(np.uint8)
            else:
                # [-1, 1] -> [0, 1] -> [0, 255]
                ct_hwc_uint8 = np.clip(((ct_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
                pet_hwc_uint8 = np.clip(((pet_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
                
            mask_hwc_uint8 = np.clip(mask_hwc * 255.0, 0, 255).astype(np.uint8)
            
            # 先应用CIPA风格的随机裁剪（在Albumentations之前）
            ct_hwc_uint8, pet_hwc_uint8, mask_hwc_uint8 = self._random_crop_cipa_style(
                ct_hwc_uint8, pet_hwc_uint8, mask_hwc_uint8, u=0.5
            )
            
            # 应用其他增强（Albumentations）
            augmented = self.transform(
                image=ct_hwc_uint8,
                image1=pet_hwc_uint8,
                mask=mask_hwc_uint8
            )
            
            ct_aug = augmented['image']
            pet_aug = augmented['image1']
            mask_aug = augmented['mask']
            
            # 转回 float
            if self.normalization_mode == 'vmamba':
                ct_hwc = ct_aug.astype(np.float32) / 255.0 * 3.2 - 1.6
                pet_hwc = pet_aug.astype(np.float32) / 255.0 * 3.2 - 1.6
            else:
                ct_hwc = ct_aug.astype(np.float32) / 255.0 * 2.0 - 1.0
                pet_hwc = pet_aug.astype(np.float32) / 255.0 * 2.0 - 1.0
            
            mask_hwc = mask_aug.astype(np.float32) / 255.0
            
            # 转回CHW格式
            ct_array = np.expand_dims(ct_hwc, axis=0)
            pet_array = np.expand_dims(pet_hwc, axis=0)
            mask_array = np.expand_dims(mask_hwc, axis=0).astype(np.float32)
        
        # 转换为tensor
        ct_tensor = torch.from_numpy(ct_array).type(torch.FloatTensor)
        pet_tensor = torch.from_numpy(pet_array).type(torch.FloatTensor)
        mask_tensor = torch.from_numpy(mask_array).type(torch.FloatTensor)
        
        return {
            'ct': ct_tensor,
            'pet': pet_tensor,
            'mask': mask_tensor,
            'name': data_info['name']
        }


# 保留旧代码用于兼容
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        img = (img - 0.5) / 0.5
        mask = (mask - 0.5) / 0.5

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }


class MaskDataset(Dataset):
    """旧版本的双模态数据集（保留用于兼容）"""
    def __init__(self, vis_dir, ir_dir, masks_dir, scale=1):
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(vis_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        # resize

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        vis_file = glob(self.vis_dir + idx + '.*')
        ir_file = glob(self.ir_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(ir_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {ir_file}'
        assert len(vis_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {vis_file}'
        mask = Image.open(mask_file[0])
        img_vis = Image.open(vis_file[0]).convert('L')
        img_ir = Image.open(ir_file[0]).convert('L')

        assert img_vis.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img_vis.size} and {mask.size}'

        img_vis = self.preprocess(img_vis, self.scale)
        mask = self.preprocess(mask, self.scale)
        img_ir = self.preprocess(img_ir, self.scale)

        img_vis = (img_vis - 0.5) / 0.5
        # mask = (mask - 0.5)/0.5
        img_ir = (img_ir - 0.5) / 0.5

        return {
            'vis': torch.from_numpy(img_vis).type(torch.FloatTensor),
            'ir': torch.from_numpy(img_ir).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }


class PCLT20KDataset(Dataset):
    """PCLT20K数据集加载器（仅支持TXT文件模式）
    
    数据集结构：
    - dataset_root/
      - train.txt, val.txt, test.txt  (TXT文件，每行一个样本ID)
      - 0001/
        - 0001_001_ct.png
        - 0001_001_pet.png
        - 0001_001_mask.png
        - 0001_002_ct.png
        - ...
      - 0002/
        - ...
    
    每个病人一个文件夹，文件中包含该病人的所有CT、PET和mask图片。
    文件命名格式：病人id_切片编号_模态.png
    
    数据加载方式：
    - 从TXT文件加载（类似CIPA）:
      - dataset_root/train.txt, val.txt, test.txt
      - 每行一个样本ID（如：200908508_42）
    
    按病人ID分割，避免数据泄漏
    """
    
    def __init__(self, dataset_root, split='train', val_ratio=0.2, test_ratio=0.2, 
                 random_seed=42, scale=1.0, augment=False, normalization_mode='standard'):
        """
        Args:
            dataset_root: 数据集根目录
            split: 'train' 或 'val' 或 'test'
            val_ratio: 验证集比例（已废弃，保留以兼容接口）
            test_ratio: 测试集比例（已废弃，保留以兼容接口）
            random_seed: 随机种子（已废弃，保留以兼容接口）
            scale: 图像缩放比例（0-1）
            augment: 是否进行数据增强
            normalization_mode: 'standard' or 'vmamba'
        """
        super(PCLT20KDataset, self).__init__()
        
        self.scale = scale
        self.augment = augment
        self.dataset_root = dataset_root
        self.normalization_mode = normalization_mode
        
        # 初始化增强管道
        if self.augment:
            self.transform = self._get_augmentation_pipeline()
            logging.info("数据增强已启用")
        else:
            self.transform = None
        
        # 从TXT文件加载数据
        txt_file = join(dataset_root, f'{split}.txt')
        if not exists(txt_file):
            raise FileNotFoundError(
                f"TXT文件不存在: {txt_file}\n"
                f"请确保数据集目录 {dataset_root} 下存在 {split}.txt 文件。\n"
                f"可以使用 generate_split_txt.py 脚本生成TXT文件。"
            )
        
        logging.info(f"使用TXT文件模式加载数据: {txt_file}")
        self._load_from_txt(txt_file)
    
    def _load_from_txt(self, txt_file):
        """从TXT文件加载数据（类似CIPA的方式）
        
        Args:
            txt_file: TXT文件路径（如：train.txt, val.txt, test.txt）
        """
        # 读取样本ID列表
        with open(txt_file, 'r') as f:
            slice_ids = [x.strip() for x in f if x.strip()]
        
        logging.info(f"从 {txt_file} 读取到 {len(slice_ids)} 个样本")
        
        # 构建数据列表
        self.data_list = []
        missing_count = 0
        
        for slice_id in slice_ids:
            # 提取病人ID（第一个下划线前的部分）
            patient_id = slice_id.split('_')[0]
            
            # 构建文件路径（支持大小写）
            patient_dir = join(self.dataset_root, patient_id)
            
            # 尝试不同的文件命名格式（大小写组合）
            ct_variants = [
                join(patient_dir, f"{slice_id}_CT.png"),   # 大写
                join(patient_dir, f"{slice_id}_ct.png"),   # 小写
            ]
            pet_variants = [
                join(patient_dir, f"{slice_id}_PET.png"),   # 大写
                join(patient_dir, f"{slice_id}_pet.png"),   # 小写
            ]
            mask_variants = [
                join(patient_dir, f"{slice_id}_mask.png"),  # 小写
                join(patient_dir, f"{slice_id}_Mask.png"),  # 首字母大写
            ]
            
            # 查找存在的文件路径
            ct_path = None
            pet_path = None
            mask_path = None
            
            for variant in ct_variants:
                if exists(variant):
                    ct_path = variant
                    break
            
            for variant in pet_variants:
                if exists(variant):
                    pet_path = variant
                    break
            
            for variant in mask_variants:
                if exists(variant):
                    mask_path = variant
                    break
            
            # 验证所有文件是否存在
            if ct_path and pet_path and mask_path:
                self.data_list.append({
                    'ct': ct_path,
                    'pet': pet_path,
                    'mask': mask_path,
                    'name': slice_id
                })
            else:
                missing = []
                if not ct_path:
                    missing.append('CT')
                if not pet_path:
                    missing.append('PET')
                if not mask_path:
                    missing.append('mask')
                logging.warning(f"样本 {slice_id} 缺少文件: {', '.join(missing)}")
                missing_count += 1
        
        if missing_count > 0:
            logging.warning(f"有 {missing_count} 个样本文件缺失")
        
        logging.info(f"成功加载 {len(self.data_list)} 个样本（从 {len(slice_ids)} 个样本ID）")
    
    def __len__(self):
        return len(self.data_list)
    
    def _get_augmentation_pipeline(self):
        """获取数据增强管道（与LungSegmentationDataset相同）"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                rotate=(-15, 15),
                translate_percent=(0, 0.05),
                scale=(0.9, 1.1),
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                fit_output=False
            ),
            A.ElasticTransform(
                p=0.3,
                alpha=120,
                sigma=120 * 0.05,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.GaussNoise(
                std_range=(0.04, 0.2),
                p=0.2
            ),
        ], additional_targets={'image1': 'image'})
    
    def _random_crop_cipa_style(self, ct_hwc, pet_hwc, mask_hwc, u=0.5):
        """CIPA风格的随机裁剪（与LungSegmentationDataset相同）"""
        if np.random.random() >= u:
            return ct_hwc, pet_hwc, mask_hwc
        
        h, w = ct_hwc.shape[:2]
        crop_rate = np.random.uniform(0.7, 0.9)
        crop_h = int(h * crop_rate)
        crop_w = int(w * crop_rate)
        
        y = np.random.randint(0, h - crop_h + 1)
        x = np.random.randint(0, w - crop_w + 1)
        
        ct_crop = ct_hwc[y:y+crop_h, x:x+crop_w]
        pet_crop = pet_hwc[y:y+crop_h, x:x+crop_w]
        mask_crop = mask_hwc[y:y+crop_h, x:x+crop_w]
        
        ct_cropped = cv2.resize(ct_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        pet_cropped = cv2.resize(pet_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        mask_cropped = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        
        return ct_cropped, pet_cropped, mask_cropped
    
    @staticmethod
    def preprocess(img, scale=1.0, is_mask=False):
        """预处理图像（与LungSegmentationDataset相同）"""
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        if scale != 1.0:
            from PIL import Image
            new_w = int(img_array.shape[1] * scale)
            new_h = int(img_array.shape[0] * scale)
            if is_mask:
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.NEAREST)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
            else:
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.BILINEAR)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
        
        img_trans = img_array.transpose((2, 0, 1))
        
        if not is_mask and img_trans.max() > 1:
            img_trans = img_trans / 255.0
        
        return img_trans
    
    def __getitem__(self, idx):
        """获取单个样本（与LungSegmentationDataset相同）"""
        data_info = self.data_list[idx]
        
        # 加载图像
        try:
            ct_img = Image.open(data_info['ct']).convert('L')
            pet_img = Image.open(data_info['pet']).convert('L')
            mask_img = Image.open(data_info['mask']).convert('L')
        except Exception as e:
            logging.error(f"Error loading images: {data_info['name']}, {e}")
            return self.__getitem__((idx + 1) % len(self.data_list))
        
        # 验证图像尺寸一致性
        assert ct_img.size == pet_img.size == mask_img.size, \
            f"Image sizes mismatch for {data_info['name']}"
        
        # 预处理
        ct_array = self.preprocess(ct_img, self.scale, is_mask=False)
        pet_array = self.preprocess(pet_img, self.scale, is_mask=False)
        mask_array = self.preprocess(mask_img, self.scale, is_mask=True)
        
        # 归一化
        if self.normalization_mode == 'vmamba':
            ct_array = ct_array * 3.2 - 1.6
            pet_array = pet_array * 3.2 - 1.6
        else:
            ct_array = (ct_array - 0.5) / 0.5
            pet_array = (pet_array - 0.5) / 0.5
        
        # Mask二值化处理
        mask_array = (mask_array > 0.5).astype(np.float32)
        
        # 应用数据增强（如果需要）
        if self.transform is not None:
            ct_hwc = ct_array.transpose(1, 2, 0).squeeze()
            pet_hwc = pet_array.transpose(1, 2, 0).squeeze()
            mask_hwc = mask_array.transpose(1, 2, 0).squeeze()
            
            if self.normalization_mode == 'vmamba':
                ct_hwc_uint8 = np.clip(((ct_hwc + 1.6) / 3.2 * 255.0), 0, 255).astype(np.uint8)
                pet_hwc_uint8 = np.clip(((pet_hwc + 1.6) / 3.2 * 255.0), 0, 255).astype(np.uint8)
            else:
                ct_hwc_uint8 = np.clip(((ct_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
                pet_hwc_uint8 = np.clip(((pet_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
            
            mask_hwc_uint8 = np.clip(mask_hwc * 255.0, 0, 255).astype(np.uint8)
            
            # 先应用CIPA风格的随机裁剪（在Albumentations之前）
            ct_hwc_uint8, pet_hwc_uint8, mask_hwc_uint8 = self._random_crop_cipa_style(
                ct_hwc_uint8, pet_hwc_uint8, mask_hwc_uint8, u=0.5
            )
            
            # 应用其他增强（Albumentations）
            augmented = self.transform(
                image=ct_hwc_uint8,
                image1=pet_hwc_uint8,
                mask=mask_hwc_uint8
            )
            
            ct_aug = augmented['image']
            pet_aug = augmented['image1']
            mask_aug = augmented['mask']
            
            if self.normalization_mode == 'vmamba':
                ct_hwc = ct_aug.astype(np.float32) / 255.0 * 3.2 - 1.6
                pet_hwc = pet_aug.astype(np.float32) / 255.0 * 3.2 - 1.6
            else:
                ct_hwc = ct_aug.astype(np.float32) / 255.0 * 2.0 - 1.0
                pet_hwc = pet_aug.astype(np.float32) / 255.0 * 2.0 - 1.0
            
            mask_hwc = mask_aug.astype(np.float32) / 255.0
            
            ct_array = np.expand_dims(ct_hwc, axis=0)
            pet_array = np.expand_dims(pet_hwc, axis=0)
            mask_array = np.expand_dims(mask_hwc, axis=0).astype(np.float32)
        
        # 转换为tensor
        ct_tensor = torch.from_numpy(ct_array).type(torch.FloatTensor)
        pet_tensor = torch.from_numpy(pet_array).type(torch.FloatTensor)
        mask_tensor = torch.from_numpy(mask_array).type(torch.FloatTensor)
        
        return {
            'ct': ct_tensor,
            'pet': pet_tensor,
            'mask': mask_tensor,
            'name': data_info['name']
        }
