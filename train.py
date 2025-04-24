import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import platform
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
import os
import json
import datetime
import argparse

from config import Config
from dataset import get_train_val_dataloaders
from model import get_model, EnhancedFocalLoss, get_optimizer
from utils import calculate_metrics, plot_confusion_matrices, EarlyStopping
from data_balance import apply_smote, check_balanced_dataset_exists

# 设置中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
elif platform.system() == 'Linux':
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux系统使用文泉驿微米黑
elif platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统使用Arial Unicode MS
    
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def calculate_batch_metrics(targets, predictions, threshold=0.5):
    """计算单个批次的评估指标"""
    predictions_binary = (predictions > threshold).astype(np.float32)
    precision = precision_score(targets, predictions_binary, average='macro', zero_division=0)
    recall = recall_score(targets, predictions_binary, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions_binary, average='macro', zero_division=0)
    return precision, recall, f1

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler=None, total_epochs=None):
    model.train()
    total_loss = 0
    all_targets = []
    all_predictions = []
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    
    # 使用传入的total_epochs，如果没有则默认使用Config.EPOCHS
    epochs_to_display = total_epochs if total_epochs is not None else Config.EPOCHS
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs_to_display} [Train]')
    
    for batch_idx, batch_data in enumerate(pbar):
        if len(batch_data) == 3:
            data, target, is_synthetic = batch_data
            is_synthetic = is_synthetic.to(device)
        else:
            data, target = batch_data
            is_synthetic = None
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 添加调试信息
        if batch_idx == 0:
            print(f"\n调试信息 - 第一个批次:")
            print(f"输入形状: {data.shape}")
            print(f"目标形状: {target.shape}")
            if is_synthetic is not None:
                print(f"合成样本标识形状: {is_synthetic.shape}")
                print(f"合成样本数量: {is_synthetic.sum().item()}/{len(is_synthetic)}")
        
        output = model(data)
        
        # 添加调试信息
        if batch_idx == 0:
            print(f"模型输出形状: {output.shape}")
            print(f"模型类: {model.__class__.__name__}")
            print(f"目标类别数: {Config.NUM_CLASSES}")
            
        # 使用更新后的损失函数，传入合成样本标识
        loss = criterion(output, target, is_synthetic)
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(epoch + batch_idx / len(train_loader))
        
        batch_targets = target.cpu().numpy()
        batch_predictions = output.detach().cpu().numpy()
        precision, recall, f1 = calculate_batch_metrics(batch_targets, batch_predictions)
        
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        running_precision = (running_precision * batch_idx + precision) / (batch_idx + 1)
        running_recall = (running_recall * batch_idx + recall) / (batch_idx + 1)
        running_f1 = (running_f1 * batch_idx + f1) / (batch_idx + 1)
        
        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'prec': f'{running_precision:.4f}',
            'rec': f'{running_recall:.4f}',
            'f1': f'{running_f1:.4f}'
        })
        
        total_loss += loss.item()
        all_targets.extend(batch_targets)
        all_predictions.extend(batch_predictions)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, np.array(all_targets), np.array(all_predictions)

def validate(model, val_loader, criterion, device, epoch, total_epochs=None):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    
    # 使用传入的total_epochs，如果没有则默认使用Config.EPOCHS
    epochs_to_display = total_epochs if total_epochs is not None else Config.EPOCHS
    
    # 创建验证进度条
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs_to_display} [Valid]')
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                data, target, is_synthetic = batch_data
                is_synthetic = is_synthetic.to(device)
            else:
                data, target = batch_data
                is_synthetic = None
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 使用更新后的损失函数，传入合成样本标识
            loss = criterion(output, target, is_synthetic)
            
            # 计算当前批次的指标
            batch_targets = target.cpu().numpy()
            batch_predictions = output.cpu().numpy()
            precision, recall, f1 = calculate_batch_metrics(batch_targets, batch_predictions)
            
            # 更新累积指标
            running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
            running_precision = (running_precision * batch_idx + precision) / (batch_idx + 1)
            running_recall = (running_recall * batch_idx + recall) / (batch_idx + 1)
            running_f1 = (running_f1 * batch_idx + f1) / (batch_idx + 1)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{running_loss:.4f}',
                'prec': f'{running_precision:.4f}',
                'rec': f'{running_recall:.4f}',
                'f1': f'{running_f1:.4f}'
            })
            
            total_loss += loss.item()
            all_targets.extend(batch_targets)
            all_predictions.extend(batch_predictions)
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, np.array(all_targets), np.array(all_predictions)

def train_model(model, train_loader, val_loader, device, criterion, stage='initial', num_epochs=None, results_dir='training_results'):
    if num_epochs is None:
        num_epochs = Config.EPOCHS
    
    optimizer = get_optimizer(model, stage)
    
    if stage == 'initial':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=Config.SCHEDULER_PATIENCE,
            factor=Config.SCHEDULER_FACTOR
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
    
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    best_val_loss = float('inf')
    best_model = None
    train_losses = []
    val_losses = []
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n开始{stage}阶段训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        train_loss, train_targets, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scheduler, total_epochs=num_epochs
        )
        train_losses.append(train_loss)
        
        # 验证阶段
        val_loss, val_targets, val_preds = validate(
            model, val_loader, criterion, device, epoch, total_epochs=num_epochs
        )
        val_losses.append(val_loss)
        
        # 计算指标
        train_metrics = calculate_metrics(train_targets, train_preds)
        val_metrics = calculate_metrics(val_targets, val_preds)
        
        # 打印当前epoch的详细信息
        print(f'\n当前学习率: {optimizer.param_groups[0]["lr"]:.2e}')
        print(f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
        print('\n验证集指标:')
        print(f'精确率: {val_metrics["precision"]}')
        print(f'召回率: {val_metrics["recall"]}')
        print(f'F1分数: {val_metrics["f1"]}')
        print(f'AUC: {val_metrics["auc"]}')
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            # 将模型保存到results_dir目录
            torch.save(model.state_dict(), os.path.join(results_dir, f'best_model_{stage}.pth'))
            print(f'\n保存最佳模型 (阶段: {stage}, 验证损失: {val_loss:.4f}) 到 {results_dir}')
        
        # 早停
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n{stage}阶段触发早停机制")
            break
    
    return best_model, train_losses, val_losses

def prepare_dataset(args):
    """准备数据集，交互式询问用户是否使用数据平衡"""
    # 获取原始数据集路径
    original_train_csv = Config.TRAIN_CSV
    print(f"原始数据集路径: {original_train_csv}")
    
    # 确保balanced_dataset目录存在
    os.makedirs(os.path.dirname(Config.BALANCED_DATASET_PATH), exist_ok=True)
    
    dataset_to_use = original_train_csv  # 默认使用原始数据集
    
    # 交互式询问用户是否使用数据平衡
    use_balance = input("\n是否需要使用data_balance.py进行数据平衡处理? (y/n): ").strip().lower() == 'y'
    
    # 如果配置为使用数据平衡且用户同意使用
    if Config.USE_DATA_BALANCE and use_balance:
        # 检查平衡数据集是否已存在且有效
        balance_exists = check_balanced_dataset_exists(Config.BALANCED_DATASET_PATH)
        
        # 如果平衡数据集已存在，询问是否需要重新生成
        force_rebalance = False
        if balance_exists:
            force_rebalance = input("\n平衡数据集已存在，是否需要重新生成? (y/n): ").strip().lower() == 'y'
        
        # 如果需要强制重新生成或平衡数据集不存在/无效
        if force_rebalance or not balance_exists:
            print(f"\n{'重新' if balance_exists else ''}生成平衡数据集...")
            apply_smote(
                original_train_csv, 
                Config.BALANCED_DATASET_PATH, 
                target_ratio=Config.TARGET_RATIO, 
                max_ratio=Config.MAX_RATIO
            )
            print(f"平衡数据集已保存至: {Config.BALANCED_DATASET_PATH}")
        
        # 使用平衡后的数据集
        dataset_to_use = Config.BALANCED_DATASET_PATH
        print(f"\n使用平衡后的数据集: {dataset_to_use}")
    else:
        print(f"\n使用原始数据集: {dataset_to_use}")
    
    # 获取数据加载器
    train_loader, val_loader = get_train_val_dataloaders(
        dataset_to_use,
        Config.TRAIN_IMAGES_DIR,
        Config.BATCH_SIZE,
        val_split=0.2
    )
    
    return train_loader, val_loader, dataset_to_use

def save_results(results_dir, train_losses1, val_losses1, train_losses2=None, val_losses2=None, train_losses3=None, val_losses3=None, dataset_info=None):
    """保存训练结果和可视化"""
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存训练配置和数据路径信息
    if dataset_info:
        with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
    
    # 绘制训练曲线
    if train_losses2 is None:
        # 只有一个阶段
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses1, label='训练损失')
        plt.plot(val_losses1, label='验证损失')
        plt.title('模型训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    else:
        # 多阶段训练
        plt.figure(figsize=(15, 5))
        
        # 第一阶段
        plt.subplot(131)
        plt.plot(train_losses1, label='训练损失')
        plt.plot(val_losses1, label='验证损失')
        plt.title('阶段1：分类头训练')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        # 第二阶段
        plt.subplot(132)
        plt.plot(train_losses2, label='训练损失')
        plt.plot(val_losses2, label='验证损失')
        plt.title('阶段2：部分微调')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        # 如果有第三阶段
        if train_losses3 is not None:
            plt.subplot(133)
            plt.plot(train_losses3, label='训练损失')
            plt.plot(val_losses3, label='验证损失')
            plt.title('阶段3：完全微调', fontsize=10)
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel('损失', fontsize=10)
            plt.legend(prop={'size': 8})
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    
    print(f"已保存训练曲线图到 {os.path.join(results_dir, 'training_curves.png')}")
    print(f"\n所有结果已保存到 {results_dir} 目录")

def save_multi_stage_results(results_dir, train_losses, val_losses, dataset_info=None, use_sequence=False):
    """保存多阶段训练的结果和可视化"""
    
    # 确保结果目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存训练配置和数据路径信息
    if dataset_info:
        with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
            json.dump(dataset_info, f, indent=4)
    
    # 绘制训练曲线 - 多子图
    if use_sequence:
        # 使用序列模型时有5个阶段
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        stages = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']
        titles = ['阶段1：分类头训练', '阶段2：序列模型训练', 
                 '阶段3：集成模型训练', '阶段4：部分微调', 
                 '阶段5：全模型微调']
        
        for i, (stage, title) in enumerate(zip(stages, titles)):
            if stage in train_losses and stage in val_losses:
                ax = axes[i]
                ax.plot(train_losses[stage], label='训练损失')
                ax.plot(val_losses[stage], label='验证损失')
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('损失')
                ax.legend()
        
        # 如果有第六个子图位置，用于显示模型架构
        if len(axes) > len(stages):
            axes[-1].axis('off')  # 关闭坐标轴
            axes[-1].text(0.5, 0.5, '多模态模型架构:\nViT + BiGRU + Ensemble', 
                         ha='center', va='center', fontsize=12)
    else:
        # 不使用序列模型时只有3个阶段
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        stages = ['stage1', 'stage4', 'stage5']
        titles = ['阶段1：分类头训练', '阶段4：部分微调', '阶段5：全模型微调']
        
        for i, (stage, title) in enumerate(zip(stages, titles)):
            if stage in train_losses and stage in val_losses:
                ax = axes[i]
                ax.plot(train_losses[stage], label='训练损失')
                ax.plot(val_losses[stage], label='验证损失')
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('损失')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    
    # 保存每个阶段的损失数据
    loss_data = {'train_losses': train_losses, 'val_losses': val_losses}
    with open(os.path.join(results_dir, 'loss_data.json'), 'w') as f:
        # 将numpy数组转换为列表以进行JSON序列化
        json_data = {}
        for key, value in loss_data.items():
            json_data[key] = {}
            for stage, losses in value.items():
                json_data[key][stage] = losses.tolist() if isinstance(losses, np.ndarray) else losses
        json.dump(json_data, f, indent=4)
    
    print(f"已保存训练曲线图到 {os.path.join(results_dir, 'training_curves.png')}")
    print(f"\n所有结果已保存到 {results_dir} 目录")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练脑出血检测模型')
    parser.add_argument('--single_stage', action='store_true', help='只运行第一阶段训练')
    parser.add_argument('--results_dir', type=str, default='training_results', help='结果保存目录')
    parser.add_argument('--use_sequence', action='store_true', help='使用序列模型（需要连续切片数据）')
    parser.add_argument('--sequence_length', type=int, default=24, help='序列长度')
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"当前GPU显存使用情况：")
        print(f"总显存：{torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB")
        print(f"当前占用：{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"当前缓存：{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    else:
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU训练")
    
    print("\n正在初始化...")
    
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
    torch.cuda.set_device(0)  # 使用第一个GPU
    
    # 更新配置，使用命令行参数覆盖默认配置
    use_sequence = args.use_sequence or Config.USE_SEQUENCE_MODEL
    sequence_length = args.sequence_length or Config.SEQUENCE_LENGTH
    
    # 准备数据集
    train_loader, val_loader, dataset_path = prepare_dataset(args)
    
    # 检查是否使用多阶段训练
    use_multi_stage = Config.MULTI_STAGE_TRAINING and not args.single_stage
    
    # 初始化模型
    model = get_model(use_sequence_model=use_sequence)
    model = model.to(device)
    print(f"模型训练设备: {next(model.parameters()).device}")
    print(f"是否使用序列模型: {use_sequence}")
    if use_sequence:
        print(f"序列长度: {sequence_length}")
    
    # 使用增强型Focal Loss
    criterion = EnhancedFocalLoss(
        alpha=1, 
        gamma=2
        # 不再传入synthetic_weight和class_weights参数，使用Config中的值
    ).to(device)
    
    # 保存训练信息
    dataset_info = {
        "original_data": Config.TRAIN_CSV,
        "used_data": dataset_path,
        "training_time": f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "balance_method": Config.BALANCE_METHOD if "enhanced" in dataset_path else "none",
        "target_ratio": Config.TARGET_RATIO,
        "max_ratio": Config.MAX_RATIO,
        "val_split": 0.2,
        "use_sequence_model": use_sequence,
        "sequence_length": sequence_length if use_sequence else 0,
        "multi_stage_training": use_multi_stage
    }
    
    # 记录各阶段的训练损失
    all_train_losses = {}
    all_val_losses = {}
    best_models = {}
    
    if not use_multi_stage:
        # 单阶段训练 - 仅训练分类头
        print("\n使用单阶段训练...")
        model, train_losses1, val_losses1 = train_model(
            model, train_loader, val_loader, device, criterion,
            stage='initial', num_epochs=Config.EPOCHS,
            results_dir=args.results_dir  # 传入结果目录
        )
        
        # 保存结果
        all_train_losses['stage1'] = train_losses1
        all_val_losses['stage1'] = val_losses1
        best_models['stage1'] = model
        
        # 保存训练结果
        save_results(args.results_dir, train_losses1, val_losses1, dataset_info=dataset_info)
    else:
        # 多阶段渐进式训练
        print("\n使用多阶段渐进式训练...")
        
        # 阶段1：仅训练ViT分类头
        print("\n阶段1: 训练ViT分类头")
        model, train_losses1, val_losses1 = train_model(
            model, train_loader, val_loader, device, criterion,
            stage='initial', num_epochs=Config.STAGE1_EPOCHS,
            results_dir=args.results_dir  # 传入结果目录
        )
        
        all_train_losses['stage1'] = train_losses1
        all_val_losses['stage1'] = val_losses1
        best_models['stage1'] = copy.deepcopy(model)
        
        if use_sequence:
            # 阶段2：训练序列模型
            print("\n阶段2: 训练序列模型")
            model, train_losses2, val_losses2 = train_model(
                model, train_loader, val_loader, device, criterion,
                stage='sequence', num_epochs=Config.STAGE2_EPOCHS,
                results_dir=args.results_dir  # 传入结果目录
            )
            
            all_train_losses['stage2'] = train_losses2
            all_val_losses['stage2'] = val_losses2
            best_models['stage2'] = copy.deepcopy(model)
            
            # 阶段3：训练集成模型
            print("\n阶段3: 训练集成模型")
            model, train_losses3, val_losses3 = train_model(
                model, train_loader, val_loader, device, criterion,
                stage='ensemble', num_epochs=Config.STAGE3_EPOCHS,
                results_dir=args.results_dir  # 传入结果目录
            )
            
            all_train_losses['stage3'] = train_losses3
            all_val_losses['stage3'] = val_losses3
            best_models['stage3'] = copy.deepcopy(model)
        
        # 阶段4：解冻部分层进行微调
        print("\n阶段4: 部分层微调")
        model.unfreeze_layers(num_layers=4)  # 解冻最后4层
        model, train_losses4, val_losses4 = train_model(
            model, train_loader, val_loader, device, criterion,
            stage='fine_tune', num_epochs=Config.STAGE4_EPOCHS,
            results_dir=args.results_dir  # 传入结果目录
        )
        
        all_train_losses['stage4'] = train_losses4
        all_val_losses['stage4'] = val_losses4
        best_models['stage4'] = copy.deepcopy(model)
        
        # 阶段5：全模型微调
        print("\n阶段5: 全模型微调")
        model.unfreeze_backbone()
        model, train_losses5, val_losses5 = train_model(
            model, train_loader, val_loader, device, criterion,
            stage='full_tune', num_epochs=Config.STAGE5_EPOCHS,
            results_dir=args.results_dir  # 传入结果目录
        )
        
        all_train_losses['stage5'] = train_losses5
        all_val_losses['stage5'] = val_losses5
        best_models['stage5'] = model
        
        # 确保结果目录存在
        os.makedirs(args.results_dir, exist_ok=True)
        
        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(args.results_dir, 'final_model.pth'))
        
        # 保存训练结果
        save_multi_stage_results(
            args.results_dir, 
            all_train_losses, 
            all_val_losses,
            dataset_info=dataset_info,
            use_sequence=use_sequence
        )
    
    print("\n训练完成！保存模型和可视化结果...")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("训练完成！")

if __name__ == '__main__':
    main() 