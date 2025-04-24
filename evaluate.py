import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns

from config import Config
from model import get_model
from dataset import BrainHemorrhageDataset, apply_window_settings
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_best_model(model_path):
    """加载最佳模型"""
    model = get_model(use_sequence_model=False)
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, test_loader, device, class_names, thresholds=None):
    """评估模型在测试集上的性能"""
    model.eval()
    all_targets = []
    all_predictions = []
    all_raw_predictions = []
    
    print("在测试集上评估模型...")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            # 处理不同格式的数据加载
            if len(batch_data) == 3:  # 如果返回了合成样本标识
                data, target, _ = batch_data
            else:
                data, target = batch_data
                
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            all_targets.extend(target.cpu().numpy())
            all_raw_predictions.extend(outputs.cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_raw_predictions = np.array(all_raw_predictions)
    
    # 如果没有提供阈值，则使用默认的0.5
    if thresholds is None:
        thresholds = [0.5] * len(class_names)
    
    # 对每个类别使用不同的阈值
    all_predictions = np.zeros_like(all_raw_predictions)
    for i, threshold in enumerate(thresholds):
        all_predictions[:, i] = (all_raw_predictions[:, i] > threshold).astype(float)
    
    # 计算每个类别的精确率、召回率和F1分数
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    
    print("\n测试集评估结果:")
    print("-" * 70)
    print(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'AUC':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        precision = precision_score(all_targets[:, i], all_predictions[:, i], zero_division=0)
        recall = recall_score(all_targets[:, i], all_predictions[:, i], zero_division=0)
        f1 = f1_score(all_targets[:, i], all_predictions[:, i], zero_division=0)
        
        # 只有当类别中有正样本时才计算AUC
        if np.sum(all_targets[:, i]) > 0:
            auc = roc_auc_score(all_targets[:, i], all_raw_predictions[:, i])
        else:
            auc = 0.0
            
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        aucs.append(auc)
        
        print(f"{class_name:<15} {precision:.4f}     {recall:.4f}     {f1:.4f}     {auc:.4f}")
    
    # 计算宏平均和微平均
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)
    macro_auc = np.mean(aucs)
    
    micro_precision = precision_score(all_targets, all_predictions, average='micro', zero_division=0)
    micro_recall = recall_score(all_targets, all_predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
    
    print("-" * 70)
    print(f"{'宏平均':<15} {macro_precision:.4f}     {macro_recall:.4f}     {macro_f1:.4f}     {macro_auc:.4f}")
    print(f"{'微平均':<15} {micro_precision:.4f}     {micro_recall:.4f}     {micro_f1:.4f}")
    
    return {
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'aucs': aucs,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_auc': macro_auc,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'all_targets': all_targets,
        'all_predictions': all_predictions,
        'all_raw_predictions': all_raw_predictions
    }

def plot_confusion_matrices(class_names, all_targets, all_predictions, output_dir):
    """为每个类别绘制混淆矩阵"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 12))
    for i, class_name in enumerate(class_names):
        plt.subplot(2, 3, i+1)
        cm = confusion_matrix(all_targets[:, i], all_predictions[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['阴性', '阳性'], 
                   yticklabels=['阴性', '阳性'])
        plt.title(f'{class_name}混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    print(f"混淆矩阵已保存到 {os.path.join(output_dir, 'confusion_matrices.png')}")

def plot_roc_curves(class_names, all_targets, all_raw_predictions, output_dir):
    """为每个类别绘制ROC曲线"""
    from sklearn.metrics import roc_curve
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        if np.sum(all_targets[:, i]) > 0:  # 只有当有正样本时才绘制ROC曲线
            fpr, tpr, _ = roc_curve(all_targets[:, i], all_raw_predictions[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc_score(all_targets[:, i], all_raw_predictions[:, i]):.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    print(f"ROC曲线已保存到 {os.path.join(output_dir, 'roc_curves.png')}")

def find_optimal_thresholds(all_targets, all_raw_predictions, class_names):
    """为每个类别找到最佳阈值（以F1分数为优化目标）"""
    from sklearn.metrics import f1_score
    
    optimal_thresholds = []
    
    print("\n为每个类别寻找最优阈值...")
    print("-" * 70)
    print(f"{'类别':<15} {'最优阈值':<10} {'优化后F1':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        best_threshold = 0.5
        best_f1 = 0.0
        
        # 测试不同的阈值
        for threshold in np.arange(0.1, 1.0, 0.05):
            predictions = (all_raw_predictions[:, i] > threshold).astype(float)
            f1 = f1_score(all_targets[:, i], predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"{class_name:<15} {best_threshold:.4f}     {best_f1:.4f}")
    
    return optimal_thresholds

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 类别名称
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    # 询问用户要评估哪个模型
    print("\n可用的模型:")
    print("1. best_model_initial.pth (分类头训练的最佳模型)")
    print("2. best_model_fine_tune.pth (部分微调的最佳模型)")
    print("3. best_model_full_tune.pth (全模型微调的最佳模型)")
    print("4. final_model.pth (最终保存的模型)")
    
    model_choice = input("请选择要评估的模型 (1-4): ")
    if model_choice == '1':
        model_path = 'best_model_initial.pth'
    elif model_choice == '2':
        model_path = 'best_model_fine_tune.pth'
    elif model_choice == '3':
        model_path = 'best_model_full_tune.pth'
    else:
        model_path = 'training_results/final_model.pth'
    
    # 加载模型
    try:
        model = load_best_model(model_path)
        model = model.to(device)
        print(f"模型已从 {model_path} 加载")
    except Exception as e:
        print(f"加载模型错误: {str(e)}")
        return
    
    # 创建测试数据集
    test_dataset = BrainHemorrhageDataset(
        csv_file=Config.TEST_CSV,
        image_dir=Config.TEST_IMAGES_DIR,
        transform=None,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    print(f"测试集大小: {len(test_dataset)} 样本")
    
    # 询问是否要使用自定义阈值
    use_custom_thresholds = input("\n是否要使用自定义预测阈值? (y/n): ").lower() == 'y'
    
    if use_custom_thresholds:
        # 从Config中读取预定义的阈值
        if hasattr(Config, 'PREDICTION_THRESHOLDS'):
            thresholds = [
                Config.PREDICTION_THRESHOLDS['epidural'],
                Config.PREDICTION_THRESHOLDS['intraparenchymal'],
                Config.PREDICTION_THRESHOLDS['intraventricular'],
                Config.PREDICTION_THRESHOLDS['subarachnoid'],
                Config.PREDICTION_THRESHOLDS['subdural']
            ]
            print(f"使用配置中的预定义阈值: {thresholds}")
        else:
            # 手动输入阈值
            thresholds = []
            for class_name in class_names:
                threshold = float(input(f"请输入{class_name}的预测阈值 (0.0-1.0): "))
                thresholds.append(threshold)
    else:
        # 使用默认阈值0.5
        thresholds = [0.5] * len(class_names)
    
    # 评估模型
    results = evaluate_model(model, test_loader, device, class_names, thresholds)
    
    # 询问是否要寻找最优阈值
    find_optimal = input("\n是否要寻找每个类别的最优阈值? (y/n): ").lower() == 'y'
    if find_optimal:
        optimal_thresholds = find_optimal_thresholds(
            results['all_targets'], 
            results['all_raw_predictions'],
            class_names
        )
        
        # 使用最优阈值重新评估
        print("\n使用最优阈值重新评估...")
        results = evaluate_model(model, test_loader, device, class_names, optimal_thresholds)
    
    # 绘制混淆矩阵
    plot_confusion_matrices(
        class_names, 
        results['all_targets'], 
        results['all_predictions'], 
        'evaluation_results'
    )
    
    # 绘制ROC曲线
    plot_roc_curves(
        class_names, 
        results['all_targets'], 
        results['all_raw_predictions'], 
        'evaluation_results'
    )
    
    print("\n评估完成! 结果已保存到 evaluation_results 目录")

if __name__ == "__main__":
    main() 