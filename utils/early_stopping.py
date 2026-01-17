import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation Dice doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, mode='max', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation Dice improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation Dice improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): 'max' for Dice (higher is better), 'min' for loss (lower is better)
                            Default: 'max'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.trace_func = trace_func
        
        # 用于记录最佳指标值
        if mode == 'max':
            self.best_metric = -np.Inf
        else:
            self.best_metric = np.Inf

    def __call__(self, val_metric, model):
        """
        Args:
            val_metric: Validation metric to monitor (Dice coefficient for mode='max', loss for mode='min')
            model: The model being trained
        """
        if self.mode == 'max':
            # For Dice: higher is better
            score = val_metric
            is_better = score > self.best_metric + self.delta
        else:
            # For loss: lower is better
            score = -val_metric
            is_better = score > self.best_metric + self.delta

        if self.best_score is None:
            self.best_score = score
            self.best_metric = val_metric
            self.counter = 0
        elif not is_better:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metric = val_metric
            self.counter = 0
            if self.verbose:
                self.trace_func(f'Validation {self.mode} improved to {val_metric:.6f}')

