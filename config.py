import torch

class Config:
    # 数据路径
    TRAIN_CSV = "E:/shujuwajue/subset/train_label.csv"
    TEST_CSV = "E:/shujuwajue/subset/test_label.csv"
    TRAIN_IMAGES_DIR = "E:/shujuwajue/subset/train_images/"
    TEST_IMAGES_DIR = "E:/shujuwajue/subset/test_images/"
    
    # 模型参数
    MODEL_NAME = "vit_base_patch16_224"
    NUM_CLASSES = 5  # 5个出血类别: epidural, intraparenchymal, intraventricular, subarachnoid, subdural
    IMAGE_SIZE = 224  # 保持224以匹配预训练模型
    BATCH_SIZE = 64
    EPOCHS = 100  # 增加总训练轮数到100
    LEARNING_RATE = 1e-4  # 提高学习率以加速收敛
    WEIGHT_DECAY = 1e-3  # 增强正则化，从1e-4增至1e-3
    
    # 多阶段训练配置
    MULTI_STAGE_TRAINING = True  # 是否使用多阶段训练
    USE_SEQUENCE_MODEL = False  # 是否使用序列模型（需要连续切片数据）
    SEQUENCE_LENGTH = 24  # 序列长度，每个样本包含的连续切片数
    STAGE1_EPOCHS = 50  # 阶段1：ViT分类头训练（增加到50轮）
    STAGE2_EPOCHS = 30  # 阶段2：序列模型训练（增加到30轮）
    STAGE3_EPOCHS = 30  # 阶段3：集成模型训练（增加到30轮）
    STAGE4_EPOCHS = 25  # 阶段4：部分层微调（增加到25轮）
    STAGE5_EPOCHS = 20  # 阶段5：全模型微调（增加到20轮）
    
    # 类别权重配置 - 按照稀有程度设置权重，但减小差异
    CLASS_WEIGHTS = {
        'epidural': 8.0,          # 硬膜外出血（最稀有），从15.0降至8.0
        'intraparenchymal': 4.0,  # 脑实质出血，从6.0降至4.0
        'intraventricular': 5.0,  # 脑室出血，从8.0降至5.0
        'subarachnoid': 4.0,      # 蛛网膜下腔出血，从6.0降至4.0
        'subdural': 3.0           # 硬膜下出血，从5.0降至3.0
    }
    
    # 预测阈值 - 针对不同类别设置不同阈值
    PREDICTION_THRESHOLDS = {
        'epidural': 0.6,         # 较低阈值提高召回率
        'intraparenchymal': 0.35,
        'intraventricular': 0.3,
        'subarachnoid': 0.35,
        'subdural': 0.35
    }
    
    # 数据加载参数
    NUM_WORKERS = 8  # 数据加载线程数
    PIN_MEMORY = True  # 使用固定内存
    PREFETCH_FACTOR = 2  # 预加载因子
    
    # 数据增强参数
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    HUE = 0.1
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 窗宽窗位组合
    WINDOW_SETTINGS = [
        (40, 80),      # Channel 0: Brain window (WC=40, WW=80)
        (80, 200),     # Channel 1: Blood/Subdural window (WC=80, WW=200)
        (600, 2800)    # Channel 2: Bone window (WC=600, WW=2800)
    ]
    
    # 图像归一化参数
    NORMALIZE_MEAN = [0.1738, 0.1433, 0.1970]  # CT图像特定的均值
    NORMALIZE_STD = [0.3161, 0.2850, 0.3111]   # CT图像特定的标准差
    
    # 训练配置
    EARLY_STOPPING_PATIENCE = 15  # 早停耐心增加到15轮
    SCHEDULER_PATIENCE = 8  # 学习率调度器耐心增加到8轮
    SCHEDULER_FACTOR = 0.5  # 学习率衰减因子调整为0.5
    
    # 合成样本权重配置
    SYNTHETIC_SAMPLE_WEIGHT = 0.3  # 合成样本的权重，从0.5降至0.3，减少合成样本的影响
    
    # 数据平衡配置
    USE_DATA_BALANCE = True  # 是否使用数据平衡
    BALANCE_METHOD = "kmeans_smote"  # 可选: "kmeans_smote", "borderline_smote", "adasyn"
    TARGET_RATIO = 0.1  # 每个类别的目标比例（占原始数据集大小的比例）
    MAX_RATIO = 0.2  # 每个类别的最大比例
    BALANCED_DATASET_PATH = "balanced_dataset/enhanced_train_label.csv"  # 平衡后数据集保存路径
    
    # CUDA优化配置
    CUDNN_BENCHMARK = True  # 启用cuDNN基准测试
    CUDA_LAUNCH_BLOCKING = 0  # 禁用CUDA启动阻塞