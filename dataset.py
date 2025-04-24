import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pydicom
from config import Config
from sklearn.model_selection import train_test_split

def apply_window_settings(image, window_center, window_width):
    """应用窗宽窗位设置"""
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    image = np.clip(image, min_value, max_value)
    image = ((image - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    return image

class BrainHemorrhageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.is_train = is_train
        
        # 使用CT图像特定的归一化参数
        self.normalize = transforms.Normalize(
            mean=Config.NORMALIZE_MEAN,
            std=Config.NORMALIZE_STD
        )
        
        # 基础转换（调整大小和转换为tensor）
        self.resize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                antialias=True,
                interpolation=transforms.InterpolationMode.BILINEAR  # 使用双线性插值
            ),
            self.normalize
        ])
        
        # 训练时的数据增强
        if is_train:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.ColorJitter(
                    brightness=Config.BRIGHTNESS,
                    contrast=Config.CONTRAST,
                    saturation=Config.SATURATION,
                    hue=Config.HUE
                )
            ])
        else:
            self.aug_transform = None
            
        # 获取出血类型标签
        self.labels = self.data[['epidural', 'intraparenchymal', 
                               'intraventricular', 'subarachnoid', 'subdural']].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.image_dir, img_name)
        
        # 检查是否为SMOTE生成的合成样本
        is_synthetic = 'SMOTE_' in img_name
        
        if is_synthetic:
            # 对于SMOTE样本，我们需要找到它基于的模板样本
            # 从文件名中获取出血类型
            bleed_type = img_name.split('_')[1]
            
            # 查找相同出血类型的真实样本
            real_samples = self.data[
                (self.data[bleed_type] == 1) & 
                (~self.data['filename'].str.contains('SMOTE_'))
            ]
            
            if len(real_samples) == 0:
                print(f"警告: 找不到与SMOTE样本 {img_name} 匹配的真实样本")
                # 返回一个全零的图像、标签和合成样本标识
                labels = torch.FloatTensor(self.labels[idx])
                return torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)), labels, torch.tensor(is_synthetic)
            
            # 随机选择一个真实样本作为替代
            template_idx = real_samples.index[np.random.randint(len(real_samples))]
            template_name = self.data.loc[template_idx, 'filename']
            img_path = os.path.join(self.image_dir, template_name)
        
        try:
            # 读取DICOM文件
            dcm = pydicom.dcmread(img_path)
            image = dcm.pixel_array.astype(float)
            
            # 应用不同的窗宽窗位设置
            images = []
            for window_center, window_width in Config.WINDOW_SETTINGS:
                windowed_image = apply_window_settings(image.copy(), window_center, window_width)
                images.append(windowed_image)
            
            # 将三个窗口设置的图像组合成三通道图像
            image = np.stack(images, axis=-1)
            
            # 应用基础转换（调整大小和标准化）
            image = self.resize_transform(image)
            
            # 如果是SMOTE样本，增加额外的数据增强以增加多样性
            if is_synthetic and self.is_train:
                # 对SMOTE样本应用更强的数据增强
                strong_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.7),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1)
                    ),
                    transforms.ColorJitter(
                        brightness=Config.BRIGHTNESS * 1.5,
                        contrast=Config.CONTRAST * 1.5,
                        saturation=Config.SATURATION * 1.5,
                        hue=Config.HUE * 1.5
                    )
                ])
                image = strong_aug(image)
            # 如果是训练模式且启用了数据增强，则应用增强
            elif self.is_train and self.aug_transform is not None:
                image = self.aug_transform(image)
            
            labels = torch.FloatTensor(self.labels[idx])
            return image, labels, torch.tensor(is_synthetic)
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回一个全零的图像、标签和合成样本标识
            return torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)), torch.zeros(len(self.labels[idx])), torch.tensor(is_synthetic)

def get_train_val_dataloaders(train_csv, train_dir, batch_size, val_split=0.2, synthetic_ratio=0.2):
    """创建训练集和验证集的数据加载器
    
    参数:
        train_csv: 训练CSV文件路径
        train_dir: 训练图像目录
        batch_size: 批次大小
        val_split: 验证集比例
        synthetic_ratio: 训练集中使用的合成样本比例（已不再使用）
    """
    # 读取数据集
    df = pd.read_csv(train_csv)
    
    # 验证是否成功加载了增强后的数据集
    print(f"加载的数据集: {train_csv}")
    print(f"数据集总样本数: {len(df)}")
    
    # 统计各类别在总体数据集中的分布
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    print("\n加载的数据集中各出血类型分布:")
    for class_name in class_names:
        count = df[class_name].sum()
        ratio = count / len(df)
        print(f"{class_name}: {count} 样本, 占比 {ratio*100:.2f}%")
    
    # 识别真实样本和SMOTE生成的合成样本
    real_indices = df[~df['filename'].str.contains('SMOTE_')].index.tolist()
    synthetic_indices = df[df['filename'].str.contains('SMOTE_')].index.tolist()
    
    print(f"数据集中真实样本数量: {len(real_indices)}")
    print(f"数据集中SMOTE生成的合成样本数量: {len(synthetic_indices)}")
    
    # *** 全新的数据集分割逻辑 - 所有样本（真实+合成）按比例分割 ***
    train_indices = []
    val_indices = []
    
    # 处理每个类别的样本
    for class_name in class_names:
        print(f"\n处理 {class_name} 类别:")
        
        # 获取该类别的所有样本（真实+合成）
        class_indices = df[df[class_name] == 1].index.tolist()
        class_real_indices = [idx for idx in class_indices if idx in real_indices]
        class_synthetic_indices = [idx for idx in class_indices if idx in synthetic_indices]
        
        print(f"- 总样本数: {len(class_indices)}")
        print(f"- 真实样本数: {len(class_real_indices)}")
        print(f"- 合成样本数: {len(class_synthetic_indices)}")
        
        # 将所有样本（真实+合成）随机打乱后按比例分割
        np.random.shuffle(class_indices)
        split_idx = int(len(class_indices) * (1 - val_split))
        class_train = class_indices[:split_idx]
        class_val = class_indices[split_idx:]
        
        # 统计训练集和验证集中的真实样本和合成样本数量
        train_real = [idx for idx in class_train if idx in real_indices]
        train_synthetic = [idx for idx in class_train if idx in synthetic_indices]
        val_real = [idx for idx in class_val if idx in real_indices]
        val_synthetic = [idx for idx in class_val if idx in synthetic_indices]
        
        # 打印分割后的样本数量
        print(f"- 训练集: {len(train_real)} 真实样本 + {len(train_synthetic)} 合成样本 = {len(class_train)} 总样本 ({len(class_train)/len(class_indices)*100:.1f}%)")
        print(f"- 验证集: {len(val_real)} 真实样本 + {len(val_synthetic)} 合成样本 = {len(class_val)} 总样本 ({len(class_val)/len(class_indices)*100:.1f}%)")
        
        # 添加到总的训练和验证索引
        train_indices.extend(class_train)
        val_indices.extend(class_val)
    
    # 处理没有任何出血的负样本
    negative_indices = df[df['any'] == 0].index.tolist()
    negative_real_indices = [idx for idx in negative_indices if idx in real_indices]
    negative_synthetic_indices = [idx for idx in negative_indices if idx in synthetic_indices]
    
    print(f"\n处理无出血的负样本:")
    print(f"- 真实负样本数: {len(negative_real_indices)}")
    print(f"- 合成负样本数: {len(negative_synthetic_indices)}")
    
    if len(negative_indices) > 0:
        # 负样本也按8:2比例分割
        np.random.shuffle(negative_indices)
        split_idx = int(len(negative_indices) * (1 - val_split))
        neg_train_indices = negative_indices[:split_idx]
        neg_val_indices = negative_indices[split_idx:]
        
        # 统计真实和合成的负样本数量
        neg_train_real = [idx for idx in neg_train_indices if idx in real_indices]
        neg_train_synthetic = [idx for idx in neg_train_indices if idx in synthetic_indices]
        neg_val_real = [idx for idx in neg_val_indices if idx in real_indices]
        neg_val_synthetic = [idx for idx in neg_val_indices if idx in synthetic_indices]
        
        print(f"- 训练集添加: {len(neg_train_real)} 真实负样本 + {len(neg_train_synthetic)} 合成负样本 = {len(neg_train_indices)} 总样本")
        print(f"- 验证集添加: {len(neg_val_real)} 真实负样本 + {len(neg_val_synthetic)} 合成负样本 = {len(neg_val_indices)} 总样本")
        
        # 添加到总的训练和验证索引
        train_indices.extend(neg_train_indices)
        val_indices.extend(neg_val_indices)
    
    # 去除重复的索引
    train_indices = list(set(train_indices))
    val_indices = list(set(val_indices))
    
    print(f"\n最终训练集样本数: {len(train_indices)}")
    print(f"最终验证集样本数: {len(val_indices)}")
    print(f"训练集占比: {len(train_indices)/(len(train_indices)+len(val_indices))*100:.1f}%")
    print(f"验证集占比: {len(val_indices)/(len(train_indices)+len(val_indices))*100:.1f}%")
    
    # 创建训练集和验证集
    train_dataset = BrainHemorrhageDataset(train_csv, train_dir, is_train=True)
    val_dataset = BrainHemorrhageDataset(train_csv, train_dir, is_train=False)
    
    # 创建训练数据加载器的子集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # 打印各类别在训练集和验证集中的分布
    print("\n各出血类型在训练集中的分布:")
    train_labels = df.iloc[train_indices][class_names].values
    for i, class_name in enumerate(class_names):
        count = np.sum(train_labels[:, i])
        ratio = count / len(train_indices)
        print(f"{class_name}: {count} 样本, 占比 {ratio*100:.2f}%")
    
    print("\n各出血类型在验证集中的分布:")
    val_labels = df.iloc[val_indices][class_names].values
    for i, class_name in enumerate(class_names):
        count = np.sum(val_labels[:, i])
        ratio = count / len(val_indices)
        print(f"{class_name}: {count} 样本, 占比 {ratio*100:.2f}%")
    
    # 检查各类别样本在训练集和验证集中的分布
    print("\n各类别在训练集和验证集中的比例:")
    for class_name in class_names:
        # 获取该类别在训练集和验证集中的样本数
        train_class_count = np.sum(df.iloc[train_indices][class_name])
        val_class_count = np.sum(df.iloc[val_indices][class_name])
        total_class_count = train_class_count + val_class_count
        
        if total_class_count > 0:
            train_ratio = train_class_count / total_class_count
            val_ratio = val_class_count / total_class_count
            print(f"{class_name}: 训练集 {train_class_count}/{total_class_count} ({train_ratio*100:.1f}%), "
                  f"验证集 {val_class_count}/{total_class_count} ({val_ratio*100:.1f}%)")
    
    # 创建训练数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        drop_last=True
    )
    
    # 创建验证数据加载器
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR
    )
    
    return train_loader, val_loader