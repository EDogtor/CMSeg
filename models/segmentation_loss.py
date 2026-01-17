"""
医疗图像分割专用损失函数
包含Dice Loss, CrossEntropy Loss等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice损失函数
    适用于二值分割任务，对小目标友好
    修改为按样本计算（与CIPA一致）
    """
    
    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth: 平滑因子，避免分母为0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, 1, H, W] (logits或经过sigmoid)
            target: 真实值 [B, 1, H, W] (0或1)
        
        Returns:
            dice loss (按样本平均)
        """
        # 将logits转换为概率
        pred = torch.sigmoid(pred)
        
        # 按样本计算（与CIPA一致）
        # pred: [B, 1, H, W], target: [B, 1, H, W]
        # 对每个样本分别计算dice，然后取平均
        i = target.sum(dim=(1, 2, 3))  # [B] 每个样本的target sum
        j = pred.sum(dim=(1, 2, 3))    # [B] 每个样本的pred sum
        intersection = (pred * target).sum(dim=(1, 2, 3))  # [B] 每个样本的intersection
        
        # 每个样本的Dice系数
        dice_per_sample = (2. * intersection + self.smooth) / (i + j + self.smooth)
        
        # 返回平均Dice loss
        return 1 - dice_per_sample.mean()


class CombinedSegLoss(nn.Module):
    """组合损失函数：Dice Loss + CrossEntropy Loss
    
    这是医疗图像分割中最常用的组合
    - Dice Loss: 处理类别不平衡，对小目标友好
    - CE Loss: 稳定梯度，帮助收敛
    
    修改：移除pos_weight，使用按样本计算的Dice
    """
    
    def __init__(self, 
                 dice_weight=0.5, 
                 ce_weight=0.5,
                 use_class_weights=False,
                 pos_weight=None):
        """
        Args:
            dice_weight: Dice损失的权重
            ce_weight: CrossEntropy损失的权重
            use_class_weights: 是否使用类别权重（已废弃）
            pos_weight: 正样本权重（已移除，不再使用）
        """
        super(CombinedSegLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Dice Loss (按样本计算)
        self.dice_loss = DiceLoss()
        
        # BCE with Logits Loss (不使用pos_weight)
        self.ce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, 1, H, W]
            target: 真实值 [B, 1, H, W]
        
        Returns:
            total_loss, dice_loss, ce_loss
        """
        # 计算两个损失
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        # 组合损失
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        return total_loss, dice, ce


class CombinedTverskyBCELoss(nn.Module):
    """组合损失函数：Tversky Loss + BCE Loss
    
    推荐用于医疗图像分割，特别是肿瘤分割任务
    - Tversky Loss: 可以控制假阴性/假阳性的权重，对漏检更敏感
    - BCE Loss: 稳定梯度，帮助收敛
    
    默认alpha=0.7, beta=0.3，更关注假阴性（漏检）
    """
    
    def __init__(self, 
                 tversky_weight=0.7, 
                 bce_weight=0.3,
                 alpha=0.7,
                 beta=0.3):
        """
        Args:
            tversky_weight: Tversky损失的权重
            bce_weight: BCE损失的权重
            alpha: Tversky损失中假阴性的权重（FN）
            beta: Tversky损失中假阳性的权重（FP）
        """
        super(CombinedTverskyBCELoss, self).__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        
        # Tversky Loss (按样本计算)
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
        
        # BCE with Logits Loss
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, 1, H, W]
            target: 真实值 [B, 1, H, W]
        
        Returns:
            total_loss, tversky_loss, bce_loss
        """
        # 计算两个损失
        tversky = self.tversky_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        # 组合损失
        total_loss = self.tversky_weight * tversky + self.bce_weight * bce
        
        return total_loss, tversky, bce


class IoULoss(nn.Module):
    """IoU损失函数
    与Dice类似，对小目标友好
    修改为按样本计算
    """
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 将logits转换为概率
        pred = torch.sigmoid(pred)
        
        # 按样本计算
        intersection = (pred * target).sum(dim=(1, 2, 3))  # [B]
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))  # [B]
        iou_per_sample = (intersection + self.smooth) / (union - intersection + self.smooth)
        
        return 1 - iou_per_sample.mean()


class TverskyLoss(nn.Module):
    """Tversky损失
    Dice Loss的泛化版本，可以调整假阳性/假阴性的权重
    修改为按样本计算
    
    对于医疗图像分割，通常alpha=0.7, beta=0.3更关注假阴性（漏检）
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Args:
            alpha: 假阴性的权重（FN），越大越关注漏检
            beta: 假阳性的权重（FP），越大越关注误检
            smooth: 平滑因子
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 将logits转换为概率
        pred = torch.sigmoid(pred)
        
        # 按样本计算
        # TP: intersection
        intersection = (pred * target).sum(dim=(1, 2, 3))  # [B]
        # FP: pred为1但target为0
        fp = (pred * (1 - target)).sum(dim=(1, 2, 3))  # [B]
        # FN: target为1但pred为0
        fn = ((1 - pred) * target).sum(dim=(1, 2, 3))  # [B]
        
        # 每个样本的Tversky系数
        tversky_per_sample = (intersection + self.smooth) / (intersection + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1 - tversky_per_sample.mean()


class FocalLoss(nn.Module):
    """Focal Loss
    解决难易样本不平衡问题
    """
    
    def __init__(self, alpha=1.0, gamma=2.0):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数，越大越关注难样本
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        
        # Focal term
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Focal Loss
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()


def get_loss_function(loss_type='combined', **kwargs):
    """获取损失函数的便捷函数
    
    Args:
        loss_type: 损失类型
            - 'dice': Dice Loss
            - 'ce': CrossEntropy Loss
            - 'combined': Dice + BCE (不使用pos_weight)
            - 'tversky_bce': Tversky + BCE (推荐用于医疗图像分割)
            - 'iou': IoU Loss
            - 'tversky': Tversky Loss
            - 'focal': Focal Loss
        
    Returns:
        loss function
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'ce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == 'combined':
        # 移除pos_weight参数（如果传入）
        kwargs.pop('pos_weight', None)
        return CombinedSegLoss(**kwargs)
    elif loss_type == 'tversky_bce':
        # Tversky + BCE组合损失（推荐）
        tversky_weight = kwargs.pop('tversky_weight', 0.7)
        bce_weight = kwargs.pop('bce_weight', 0.3)
        alpha = kwargs.pop('alpha', 0.7)
        beta = kwargs.pop('beta', 0.3)
        return CombinedTverskyBCELoss(
            tversky_weight=tversky_weight,
            bce_weight=bce_weight,
            alpha=alpha,
            beta=beta
        )
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


