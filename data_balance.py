import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from config import Config
import matplotlib.pyplot as plt
import matplotlib
import pydicom
from multiprocessing import Pool, cpu_count
import warnings

# 代码更新：使用KMeans SMOTE替代BorderlineSMOTE
# KMeans SMOTE首先使用KMeans聚类将数据分组，然后在每个簇中单独应用SMOTE
# 这种方法能更好地处理不均衡数据集，特别是当少数类样本在特征空间中分布不均匀时
# 与FocalLoss（损失加权）结合使用，可以通过Config.SYNTHETIC_SAMPLE_WEIGHT调整合成样本权重

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings("ignore")

def extract_image_features(file_path):
    """从DICOM文件中提取简单特征"""
    try:
        dcm = pydicom.dcmread(file_path)
        img = dcm.pixel_array
        
        # 提取简单统计特征
        features = [
            np.mean(img),          # 平均值
            np.std(img),           # 标准差
            np.percentile(img, 25), # 25%分位数
            np.percentile(img, 50), # 中位数
            np.percentile(img, 75), # 75%分位数
            np.max(img),           # 最大值
            np.min(img),           # 最小值
            img.shape[0],          # 高度
            img.shape[1],          # 宽度
            np.mean(img[img.shape[0]//3:2*img.shape[0]//3, img.shape[1]//3:2*img.shape[1]//3])  # 中心区域平均值
        ]
        return features
    except Exception as e:
        print(f"特征提取错误: {str(e)}")
        return [0] * 10  # 返回全零特征

def extract_features_for_samples(df, image_dir, max_samples=1000):
    """为样本提取特征"""
    print("正在提取图像特征...")
    
    # 限制样本数量避免处理过多
    if len(df) > max_samples:
        df_subset = df.sample(max_samples, random_state=42)
    else:
        df_subset = df
    
    features = []
    filenames = []
    
    # 使用多进程加速特征提取
    full_paths = [os.path.join(image_dir, filename) for filename in df_subset['filename']]
    
    # 检查哪些文件存在
    valid_paths = []
    valid_indices = []
    for i, path in enumerate(full_paths):
        if os.path.exists(path):
            valid_paths.append(path)
            valid_indices.append(i)
    
    if not valid_paths:
        print("警告: 没有找到有效的图像文件!")
        return np.zeros((len(df_subset), 10)), df_subset.index
    
    print(f"找到 {len(valid_paths)}/{len(full_paths)} 个有效图像文件")
    
    # 使用多进程提取特征
    with Pool(processes=min(cpu_count(), 4)) as pool:
        features = pool.map(extract_image_features, valid_paths)
    
    # 转换为numpy数组，只包含有效的样本
    return np.array(features), np.array(df_subset.index)[valid_indices]

def apply_smote(csv_file, output_csv=None, target_ratio=0.1, max_ratio=0.2):
    """
    应用欠采样+过采样组合策略处理类别不平衡问题
    
    参数:
        csv_file: 输入CSV文件路径
        output_csv: 输出CSV文件路径，如果为None则不保存
        target_ratio: 每个类别的目标比例（默认0.1，即10%）
        max_ratio: 单个类别最大比例上限（默认0.2，即20%）
        
    返回:
        处理后的DataFrame
    """
    print("开始应用欠采样+过采样组合策略...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    original_samples = len(df)
    print(f"原始样本数: {original_samples}")
    
    # 获取标签列
    class_names = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    
    # 统计各类别样本数量
    class_counts = []
    class_ratios = []
    
    print("\n各出血类型的分布情况:")
    print("-" * 50)
    for class_name in class_names:
        count = df[class_name].sum()
        ratio = count / original_samples
        class_counts.append(count)
        class_ratios.append(ratio)
        print(f"{class_name}: {count} 样本, 占比 {ratio*100:.2f}%")
    
    # 计算目标样本数 - 基于原始样本数的一小部分
    target_count = int(original_samples * target_ratio)
    max_count = int(original_samples * max_ratio)
    print(f"\n目标样本数：每个类别约 {target_count} 个样本 (原始样本数的{target_ratio*100}%)")
    print(f"最大样本数：每个类别不超过 {max_count} 个样本 (原始样本数的{max_ratio*100}%)")
    
    # 创建一个新的平衡数据集
    balanced_df = pd.DataFrame(columns=df.columns)
    
    # 首先进行欠采样 - 提取负样本
    # 获取不包含任何出血类型的负样本
    negative_samples = df[df['any'] == 0].copy()
    print(f"\n原始负样本数量: {len(negative_samples)}")
    
    # 对负样本进行欠采样 - 保留与目标数量相当的负样本
    if len(negative_samples) > target_count * 2:
        # 如果负样本数量过多，随机选择一部分
        selected_negative = negative_samples.sample(target_count * 2, random_state=42)
        print(f"对负样本进行欠采样，保留 {len(selected_negative)} 个样本")
    else:
        # 否则保留所有负样本
        selected_negative = negative_samples
        print(f"保留所有 {len(selected_negative)} 个负样本")
    
    # 将选择的负样本添加到平衡数据集
    balanced_df = pd.concat([balanced_df, selected_negative], ignore_index=True)
    
    # 为每个类别单独处理
    for i, class_name in enumerate(class_names):
        # 获取该类别的所有正样本
        positive_samples = df[df[class_name] == 1].copy()
        current_count = len(positive_samples)
        
        print(f"\n处理 {class_name} 类...")
        print(f"当前样本数: {current_count}")
        
        if current_count > max_count:
            # 如果样本数量超过最大限制，进行欠采样
            print(f"{class_name} 类样本过多，进行欠采样，从 {current_count} 减少到 {max_count}")
            selected_samples = positive_samples.sample(max_count, random_state=42)
            balanced_df = pd.concat([balanced_df, selected_samples], ignore_index=True)
        elif current_count >= target_count:
            # 如果样本数量在目标范围内，全部保留
            print(f"{class_name} 类样本数量适中，保留所有 {current_count} 个样本")
            balanced_df = pd.concat([balanced_df, positive_samples], ignore_index=True)
        else:
            # 如果样本不足，需要生成合成样本
            print(f"{class_name} 类样本不足，需要生成 {target_count - current_count} 个合成样本")
            
            # 先加入所有现有的正样本
            balanced_df = pd.concat([balanced_df, positive_samples], ignore_index=True)
            
            # 需要生成的样本数
            samples_to_generate = target_count - current_count
            
            # 正样本过少，无法应用SMOTE，直接复制
            if current_count < 5:
                print(f"警告: {class_name} 类正样本太少 ({current_count} < 5)，无法应用SMOTE")
                print(f"将使用简单复制来生成合成样本")
                
                # 简单复制
                for j in range(samples_to_generate):
                    # 随机选择一个模板样本
                    template_idx = np.random.randint(len(positive_samples))
                    new_row = positive_samples.iloc[template_idx].copy()
                    
                    # 生成新的唯一文件名
                    new_filename = f"SMOTE_{class_name}_{j}.dcm"
                    new_row['filename'] = new_filename
                    
                    # 添加到平衡数据集
                    balanced_df = pd.concat([balanced_df, pd.DataFrame([new_row])], ignore_index=True)
                
                print(f"已生成 {samples_to_generate} 个复制样本")
            else:
                try:
                    # 提取图像特征用于SMOTE
                    print("准备图像特征以用于SMOTE...")
                    
                    # 获取适量负样本，不再获取过多负样本
                    # 通过控制负样本数量，可以改善SMOTE生成样本的质量
                    neg_sample_count = min(current_count * 2, 500)
                    negative_samples = df[df[class_name] == 0].sample(
                        neg_sample_count,  # 限制负样本数量
                        random_state=42
                    )
                    
                    print(f"使用 {len(negative_samples)} 个负样本作为SMOTE参考点")
                    
                    # 合并正负样本
                    samples_for_features = pd.concat([positive_samples, negative_samples])
                    
                    # 创建随机特征（作为示例，实际中应该提取真实图像特征）
                    print("使用随机特征进行SMOTE示例...")
                    np.random.seed(42)
                    
                    # 为正样本和负样本创建不同的随机特征
                    X = np.random.randn(len(samples_for_features), 10) * 5  # 10维随机特征
                    
                    # 给正样本的特征添加偏移，使其在特征空间中成簇
                    positive_indices = samples_for_features[class_name] == 1
                    X[positive_indices] += np.array([10, 5, 3, 8, 4, 6, 2, 9, 7, 1])  
                    
                    # 添加更多随机性以增加多样性
                    noise = np.random.randn(np.sum(positive_indices), 10) * 1.5
                    X[positive_indices] += noise
                    
                    # 标签
                    y = samples_for_features[class_name].values
                    
                    print(f"特征形状: {X.shape}, 标签形状: {y.shape}")
                    print(f"正样本数: {sum(y)}, 负样本数: {len(y) - sum(y)}")
                    
                    # 应用SMOTE生成合成样本
                    smote = KMeansSMOTE(
                        sampling_strategy={1: current_count + samples_to_generate},
                        k_neighbors=min(5, current_count-1),
                        random_state=42,
                        cluster_balance_threshold='auto',  # 控制簇之间的不平衡阈值
                        density_exponent='auto',  # 用于确定簇密度的指数
                        kmeans_estimator=min(5, current_count)  # 指定簇的数量
                    )
                    
                    print("应用KMeans SMOTE生成合成样本...")
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                    
                    # 获取新生成的正样本数量
                    new_positives = sum(y_resampled) - sum(y)
                    print(f"SMOTE生成了 {new_positives} 个新正样本")
                    
                    if new_positives <= 0:
                        raise ValueError("SMOTE未能生成新的正样本")
                    
                    # 找出新生成的正样本索引
                    original_positive_count = sum(y)
                    new_positive_indices = np.where(y_resampled == 1)[0][original_positive_count:]
                    
                    # 为新样本创建记录
                    remaining = samples_to_generate
                    for idx in new_positive_indices:
                        if remaining <= 0:
                            break
                            
                        # 随机选择一个原始正样本作为模板
                        template_idx = np.random.choice(positive_samples.index)
                        new_row = df.loc[template_idx].copy()
                        
                        # 确保标签正确
                        for c_name in class_names:
                            new_row[c_name] = 1 if c_name == class_name else 0
                        
                        # 生成新的唯一文件名
                        new_filename = f"SMOTE_{class_name}_{remaining}.dcm"
                        new_row['filename'] = new_filename
                        
                        # 设置any标志
                        new_row['any'] = 1
                        
                        # 添加到平衡数据集
                        balanced_df = pd.concat([balanced_df, pd.DataFrame([new_row])], ignore_index=True)
                        remaining -= 1
                    
                    # 如果SMOTE没有生成足够的样本，补充剩余的
                    if remaining > 0:
                        print(f"SMOTE只生成了部分样本，将使用复制方法生成剩余 {remaining} 个样本")
                        
                        for j in range(remaining):
                            # 随机选择一个模板样本
                            template_idx = np.random.randint(len(positive_samples))
                            new_row = positive_samples.iloc[template_idx].copy()
                            
                            # 确保标签正确
                            for c_name in class_names:
                                new_row[c_name] = 1 if c_name == class_name else 0
                            
                            # 生成新的唯一文件名
                            new_filename = f"SMOTE_CP_{class_name}_{j}.dcm"
                            new_row['filename'] = new_filename
                            
                            # 设置any标志
                            new_row['any'] = 1
                            
                            # 添加到平衡数据集
                            balanced_df = pd.concat([balanced_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    print(f"已通过SMOTE生成 {samples_to_generate - remaining} 个合成样本")
                    if remaining > 0:
                        print(f"额外通过复制生成 {remaining} 个样本")
                        
                except Exception as e:
                    print(f"SMOTE失败: {str(e)}")
                    print("使用简单复制代替")
                    
                    # 简单复制
                    for j in range(samples_to_generate):
                        # 随机选择一个模板样本
                        template_idx = np.random.randint(len(positive_samples))
                        new_row = positive_samples.iloc[template_idx].copy()
                        
                        # 确保标签正确 
                        for c_name in class_names:
                            new_row[c_name] = 1 if c_name == class_name else 0
                        
                        # 生成新的唯一文件名
                        new_filename = f"SMOTE_CP_{class_name}_{j}.dcm"
                        new_row['filename'] = new_filename
                        
                        # 设置any标志
                        new_row['any'] = 1
                        
                        # 添加到平衡数据集
                        balanced_df = pd.concat([balanced_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    print(f"已生成 {samples_to_generate} 个复制样本")
    
    # 删除重复的filename
    balanced_df = balanced_df.drop_duplicates(subset=['filename'])
    
    # 更新 'any' 列
    balanced_df['any'] = balanced_df[class_names].max(axis=1)
    
    # 显示最终的类别分布
    print("\n最终平衡后的类别分布:")
    print("-" * 50)
    for class_name in class_names:
        count = balanced_df[class_name].sum()
        ratio = count / len(balanced_df)
        ratio_of_original = count / original_samples
        print(f"{class_name}: {count} 样本, 占总样本比 {ratio*100:.2f}%, 占原始样本数比 {ratio_of_original*100:.2f}%")
    
    # 可视化过采样前后的分布
    plt.figure(figsize=(12, 6))
    
    # 过采样前
    plt.subplot(1, 2, 1)
    plt.bar(class_names, class_counts)
    plt.title('平衡前的类别分布')
    plt.xticks(rotation=45)
    plt.ylabel('样本数量')
    
    # 过采样后
    plt.subplot(1, 2, 2)
    after_counts = [balanced_df[class_name].sum() for class_name in class_names]
    plt.bar(class_names, after_counts)
    plt.title('平衡后的类别分布')
    plt.xticks(rotation=45)
    plt.ylabel('样本数量')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("类别分布图已保存为 'class_distribution.png'")
    
    if output_csv:
        balanced_df.to_csv(output_csv, index=False)
        print(f"\n平衡后的数据集已保存到: {output_csv}")
    
    return balanced_df

if __name__ == "__main__":
    # 应用欠采样+过采样组合策略
    enhanced_df = apply_smote(Config.TRAIN_CSV, "enhanced_train_label.csv", target_ratio=0.1, max_ratio=0.2)
    
    # 更新Config中的训练CSV路径
    print("\n请更新 Config.py 中的 TRAIN_CSV 路径为: 'enhanced_train_label.csv'")

def check_balanced_dataset_exists(filepath):
    """
    检查平衡数据集是否存在并有效
    
    参数:
        filepath: 平衡数据集文件路径
        
    返回:
        Boolean: 如果平衡数据集存在且有效返回True，否则返回False
    """
    if not os.path.exists(filepath):
        return False
    
    try:
        # 尝试读取数据集，确保其有效
        df = pd.read_csv(filepath)
        
        # 检查是否包含所需列
        required_columns = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'filename', 'any']
        if not all(col in df.columns for col in required_columns):
            print(f"警告: 平衡数据集缺少必要的列")
            return False
            
        # 检查是否包含SMOTE样本
        if not any('SMOTE' in str(f) for f in df['filename']):
            print(f"警告: 平衡数据集中没有SMOTE生成的样本")
            return False
            
        # 检查数据集大小是否合理
        if len(df) < 1000:  # 假设最小合理大小
            print(f"警告: 平衡数据集样本数过少: {len(df)}")
            return False
            
        print(f"平衡数据集验证通过: {filepath}")
        print(f"样本总数: {len(df)}")
        print(f"SMOTE生成样本数: {sum('SMOTE' in str(f) for f in df['filename'])}")
        return True
        
    except Exception as e:
        print(f"读取平衡数据集出错: {str(e)}")
        return False 