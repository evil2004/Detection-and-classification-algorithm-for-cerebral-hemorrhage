import torch
import torch.nn as nn
import timm
from config import Config
import numpy as np
import torch.nn.functional as F

class BiGRUModule(nn.Module):
    """双向GRU序列模型，用于捕捉CT切片间的空间连续性"""
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 正交初始化以提高训练稳定性
        for name, param in self.bigru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
                
        # 输出特征维度 = 隐藏层大小 * 2（双向）
        self.output_size = hidden_size * 2
        
    def forward(self, x):
        """
        输入: [batch_size, seq_len, input_size]
        输出: [batch_size, seq_len, hidden_size*2]
        """
        output, _ = self.bigru(x)
        return output

class SelfAttention(nn.Module):
    """自注意力机制，用于聚合序列信息"""
    def __init__(self, input_size):
        super().__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.scale = np.sqrt(input_size)
        
        # 添加Kaiming初始化
        nn.init.kaiming_normal_(self.query.weight)
        nn.init.kaiming_normal_(self.key.weight)
        nn.init.kaiming_normal_(self.value.weight)
        
    def forward(self, x):
        """
        输入: [batch_size, seq_len, input_size]
        输出: [batch_size, input_size]
        """
        # 计算注意力分数
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # 注意力权重
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attn_weights, v)
        
        # 对序列维度求均值，得到全局表示
        return context.mean(dim=1)

class MultiModalBrainHemorrhageModel(nn.Module):
    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES, pretrained=True, 
                 sequence_length=24, use_sequence_model=True):
        super().__init__()
        self.use_sequence_model = use_sequence_model
        self.sequence_length = sequence_length
        
        # 加载预训练ViT模型作为特征提取器
        self.vit_model = timm.create_model(model_name, pretrained=pretrained)
        
        # 获取特征提取器的输出维度
        if hasattr(self.vit_model, 'head'):
            self.in_features = self.vit_model.head.in_features
            # 移除原有的分类头
            self.vit_model.head = nn.Identity()
        
        # 2D分类头 - 处理单切片 (简化网络结构)
        self.classifier_2d = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 从0.5降低到0.3
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # 从0.5降低到0.3
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        
        # 添加Kaiming初始化
        for m in self.classifier_2d.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        
        # 序列处理模块（BiGRU）
        if use_sequence_model:
            self.sequence_model = BiGRUModule(
                input_size=self.in_features,
                hidden_size=256,
                num_layers=2,
                dropout=0.3  # 从0.3保持不变
            )
            
            # 序列级别分类头 (简化结构)
            self.classifier_sequence = nn.Sequential(
                nn.Linear(self.sequence_model.output_size, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),  # 从0.5降低到0.3
                nn.Linear(256, num_classes),
                nn.Sigmoid()
            )
            
            # Kaiming初始化
            for m in self.classifier_sequence.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
            
            # 自注意力层用于聚合序列信息
            self.attention = SelfAttention(self.sequence_model.output_size)
            
            # 集成模块输入维度: ViT特征 + 2D分类结果 + 序列特征
            ensemble_input_size = self.in_features + num_classes + self.sequence_model.output_size
            
            # 集成分类头 (简化结构)
            self.classifier_ensemble = nn.Sequential(
                nn.Linear(ensemble_input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),  # 从0.5降低到0.3
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
            
            # Kaiming初始化
            for m in self.classifier_ensemble.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
        
        # 冻结ViT主干网络参数
        self.freeze_backbone()
    
    def freeze_backbone(self):
        """冻结ViT主干网络参数"""
        for param in self.vit_model.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻ViT主干网络参数"""
        for param in self.vit_model.parameters():
            param.requires_grad = True
    
    def unfreeze_layers(self, num_layers):
        """渐进式解冻，从最后一层开始解冻指定数量的层"""
        if hasattr(self.vit_model, 'blocks'):
            total_layers = len(self.vit_model.blocks)
            for i, block in enumerate(self.vit_model.blocks):
                for param in block.parameters():
                    param.requires_grad = i >= (total_layers - num_layers)
    
    def forward_single_image(self, x):
        """处理单张图像"""
        # ViT特征提取
        vit_features = self.vit_model(x)
        
        # 2D分类头预测
        logits_2d = self.classifier_2d(vit_features)
        
        return vit_features, logits_2d
    
    def forward_sequence(self, sequence):
        """
        处理图像序列
        sequence: [batch_size, sequence_length, 3, height, width]
        """
        batch_size, seq_len = sequence.shape[0], sequence.shape[1]
        
        # 重塑为 [batch_size*seq_len, 3, height, width]
        flat_sequence = sequence.view(-1, sequence.shape[2], sequence.shape[3], sequence.shape[4])
        
        # 批量提取ViT特征
        vit_features, logits_2d = self.forward_single_image(flat_sequence)
        
        # 重新整形为序列格式 [batch_size, seq_len, feature_dim]
        vit_features = vit_features.view(batch_size, seq_len, -1)
        logits_2d = logits_2d.view(batch_size, seq_len, -1)
        
        # 应用BiGRU序列模型
        sequence_features = self.sequence_model(vit_features)
        
        # 使用自注意力聚合序列特征
        attended_features = self.attention(sequence_features)
        
        # 序列级别预测
        logits_sequence = self.classifier_sequence(attended_features)
        
        # 获取中心切片的特征和预测（第seq_len/2个）
        center_idx = seq_len // 2
        center_features = vit_features[:, center_idx, :]
        center_logits = logits_2d[:, center_idx, :]
        
        # 集成不同模块的结果
        ensemble_input = torch.cat([center_features, center_logits, attended_features], dim=1)
        logits_ensemble = self.classifier_ensemble(ensemble_input)
        
        return {
            'logits_2d': center_logits,
            'logits_sequence': logits_sequence,
            'logits_ensemble': logits_ensemble
        }
    
    def forward(self, x):
        """根据输入类型选择合适的前向传播方法"""
        if not self.use_sequence_model or len(x.shape) == 4:  # 单张图像 [batch, channel, height, width]
            _, logits = self.forward_single_image(x)
            return logits
        elif len(x.shape) == 5:  # 图像序列 [batch, seq_len, channel, height, width]
            results = self.forward_sequence(x)
            # 在训练模式下返回所有输出，在评估模式下返回集成结果
            if self.training:
                return results
            else:
                return results['logits_ensemble']
        else:
            raise ValueError(f"不支持的输入形状: {x.shape}")

class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, synthetic_weight=None, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # 从Config读取合成样本权重，如果未提供
        if synthetic_weight is None:
            self.synthetic_weight = Config.SYNTHETIC_SAMPLE_WEIGHT
        else:
            self.synthetic_weight = synthetic_weight
            
        # 从Config读取类别权重，如果未提供
        if class_weights is None:
            # 将Config中的类别权重字典转换为张量
            self.class_weights = torch.FloatTensor([
                Config.CLASS_WEIGHTS['epidural'],
                Config.CLASS_WEIGHTS['intraparenchymal'],
                Config.CLASS_WEIGHTS['intraventricular'],
                Config.CLASS_WEIGHTS['subarachnoid'],
                Config.CLASS_WEIGHTS['subdural']
            ])
        else:
            self.class_weights = class_weights
            
        # 确保权重是tensor
        if not isinstance(self.class_weights, torch.Tensor):
            self.class_weights = torch.FloatTensor(self.class_weights)
        
    def to(self, device):
        self.class_weights = self.class_weights.to(device)
        return super().to(device)
        
    def forward(self, inputs, targets, is_synthetic=None):
        """
        多标签分类的增强型Focal Loss，支持类别加权和合成样本加权
        
        参数:
            inputs: 可以是预测字典或直接预测值
            targets: 真实标签，形状为 [batch_size, num_classes]
            is_synthetic: 标识样本是否为合成样本的布尔张量，形状为 [batch_size]
        """
        # 处理预测字典的情况（多模态模型的输出）
        if isinstance(inputs, dict):
            # 计算每个预测头的损失，并加权求和
            loss_2d = self._compute_loss(inputs['logits_2d'], targets, is_synthetic)
            loss_sequence = self._compute_loss(inputs['logits_sequence'], targets, is_synthetic)
            loss_ensemble = self._compute_loss(inputs['logits_ensemble'], targets, is_synthetic)
            
            # 加权组合，将集成模型的权重设置得更高
            return 0.25 * loss_2d + 0.25 * loss_sequence + 0.5 * loss_ensemble
        else:
            # 单一预测的情况
            return self._compute_loss(inputs, targets, is_synthetic)
    
    def _compute_loss(self, inputs, targets, is_synthetic=None):
        """计算单个预测头的损失"""
        # 确保输入和目标形状相同
        if inputs.shape != targets.shape:
            raise ValueError(f"输入形状 {inputs.shape} 与目标形状 {targets.shape} 不匹配")
            
        # 确保类别权重的长度与类别数量匹配
        if len(self.class_weights) != inputs.shape[1]:
            raise ValueError(f"类别权重长度 {len(self.class_weights)} 与类别数 {inputs.shape[1]} 不匹配")
            
        # 将类别权重扩展到与损失相同的形状
        expanded_weights = self.class_weights.expand(inputs.shape[0], -1).to(inputs.device)
            
        # 分别计算正样本和负样本的损失
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        
        # Focal Loss公式: -alpha * (1-pt)^gamma * log(pt)
        pt = torch.exp(-bce_loss)  # pt = 预测概率
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        # 应用类别权重
        focal_loss = focal_loss * expanded_weights
        
        # 如果提供了合成样本标识，则对合成样本应用权重
        if is_synthetic is not None:
            # 创建权重掩码：合成样本权重较低，真实样本权重为1.0
            sample_weights = torch.ones_like(is_synthetic, dtype=torch.float32)
            sample_weights[is_synthetic] = self.synthetic_weight
            
            # 扩展权重维度以匹配focal_loss的形状
            sample_weights = sample_weights.view(-1, 1).expand_as(focal_loss)
            
            # 应用样本类型权重
            focal_loss = focal_loss * sample_weights
        
        # 对所有样本和所有类别取平均
        return focal_loss.mean()

def get_model(use_sequence_model=False):
    """
    创建模型实例
    参数:
        use_sequence_model: 是否使用序列模型（需要连续切片数据）
    """
    model = MultiModalBrainHemorrhageModel(use_sequence_model=use_sequence_model)
    return model

def get_optimizer(model, stage='initial'):
    """
    根据训练阶段返回不同的优化器配置
    
    stage: 'initial' - 只训练分类头
           'sequence' - 训练序列模型
           'ensemble' - 训练集成模型
           'fine_tune' - 微调部分层
           'full_tune' - 微调所有层
    """
    if stage == 'initial':
        # 初始阶段只训练2D分类头
        optimizer = torch.optim.AdamW([
            {'params': model.classifier_2d.parameters(), 'lr': Config.LEARNING_RATE, 'weight_decay': Config.WEIGHT_DECAY}
        ])
    elif stage == 'sequence' and hasattr(model, 'sequence_model'):
        # 训练序列模型
        optimizer = torch.optim.AdamW([
            {'params': model.sequence_model.parameters(), 'lr': Config.LEARNING_RATE, 'weight_decay': Config.WEIGHT_DECAY},
            {'params': model.classifier_sequence.parameters(), 'lr': Config.LEARNING_RATE, 'weight_decay': Config.WEIGHT_DECAY},
            {'params': model.attention.parameters(), 'lr': Config.LEARNING_RATE, 'weight_decay': Config.WEIGHT_DECAY}
        ])
    elif stage == 'ensemble' and hasattr(model, 'classifier_ensemble'):
        # 训练集成模型
        optimizer = torch.optim.AdamW([
            {'params': model.classifier_ensemble.parameters(), 'lr': Config.LEARNING_RATE, 'weight_decay': Config.WEIGHT_DECAY}
        ])
    elif stage == 'fine_tune':
        # 微调部分层
        optimizer = torch.optim.AdamW([
            {'params': model.vit_model.parameters(), 'lr': Config.LEARNING_RATE * 0.1, 'weight_decay': Config.WEIGHT_DECAY},
            {'params': model.classifier_2d.parameters(), 'lr': Config.LEARNING_RATE, 'weight_decay': Config.WEIGHT_DECAY}
        ])
        if hasattr(model, 'sequence_model'):
            optimizer.add_param_group({
                'params': model.sequence_model.parameters(), 
                'lr': Config.LEARNING_RATE * 0.5, 
                'weight_decay': Config.WEIGHT_DECAY
            })
            optimizer.add_param_group({
                'params': model.classifier_sequence.parameters(), 
                'lr': Config.LEARNING_RATE, 
                'weight_decay': Config.WEIGHT_DECAY
            })
            optimizer.add_param_group({
                'params': model.attention.parameters(), 
                'lr': Config.LEARNING_RATE, 
                'weight_decay': Config.WEIGHT_DECAY
            })
            optimizer.add_param_group({
                'params': model.classifier_ensemble.parameters(), 
                'lr': Config.LEARNING_RATE, 
                'weight_decay': Config.WEIGHT_DECAY
            })
    else:
        # 全模型微调
        all_params = model.parameters()
        optimizer = torch.optim.AdamW(all_params, lr=Config.LEARNING_RATE * 0.1, weight_decay=Config.WEIGHT_DECAY)
    
    return optimizer 