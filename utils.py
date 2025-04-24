import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """计算各种评估指标"""
    # 将预测概率转换为二进制标签
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # 计算每个类别的precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average=None
    )
    
    # 计算每个类别的AUC-ROC
    auc_scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:  # 确保标签中有正负样本
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': np.array(auc_scores)
    }

def plot_confusion_matrices(y_true, y_pred, class_names, threshold=0.5):
    """绘制每个类别的混淆矩阵"""
    y_pred_binary = (y_pred > threshold).astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, class_name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, idx], y_pred_binary[:, idx])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix - {class_name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    return fig

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0 