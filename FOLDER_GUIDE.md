# CoCoSeg 文件夹说明文档

本文件用于帮助整理 `CoCoSeg/` 代码库。包含完整结构、每个文件用途，以及“建议保留/可清理”标注。

标注说明：
- **[核心]** 训练/推理/模型核心代码，建议保留
- **[文档]** 说明文档，建议保留
- **[配置]** 配置文件，建议保留
- **[数据]** 数据集或样例数据，可按需要移动
- **[权重]** 预训练或训练权重，按需要保留
- **[产出]** 运行产生的日志/结果，可清理
- **[缓存]** 缓存/编译产物/中间文件，可清理
- **[兼容]** 旧版本或对比代码，按需保留

---

## 1. 代码库完整结构（总览）

```
CoCoSeg/
├── README.md
├── main.py
├── test_model.py
├── run_train.sh
├── requirements.txt
├── generate_split_txt.py
├── split_dataset.py
├── SPLIT_TXT_USAGE.md
├── TRAINING_GUIDE.md
├── TRAIN_COMMAND.md
├── TRAINING_IMPROVEMENTS.md
├── MODEL_EVALUATION.md
├── ARCHITECTURE_DIAGRAM.md
├── ARCHITECTURE_IMPROVEMENTS.md
├── DIFFERENCES_WITH_CIPA.md
├── configs/
│   ├── README.md
│   ├── default_config.json
│   ├── large_batch_config.json
│   └── focal_loss_config.json
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── segmentation_loss.py
│   ├── model.py
│   ├── train_tasks.py
│   ├── measure_model.py
│   ├── P_loss.py
│   └── vmamba/
│       ├── __init__.py
│       ├── builder.py
│       ├── dual_vmamba.py
│       ├── vmamba.py
│       ├── MambaDecoder.py
│       ├── mamba_net_utils.py
│       ├── local_vmamba/
│       │   ├── local_scan.py
│       │   └── region_mamba.py
│       └── selective_scan/
│           ├── setup.py
│           ├── readme.md
│           ├── test_selective_scan.py
│           ├── selective_scan/
│           │   ├── __init__.py
│           │   └── selective_scan_interface.py
│           ├── csrc/...
│           ├── build/...
│           └── selective_scan.egg-info/...
├── models_archived/
│   ├── __init__.py
│   ├── segmentation_loss.py
│   ├── model.py
│   ├── train_tasks.py
│   ├── measure_model.py
│   └── P_loss.py
├── utils/
│   ├── __init__.py
│   ├── attention.py
│   ├── checkpoint.py
│   ├── early_stopping.py
│   ├── ema.py
│   ├── init_func.py
│   ├── logger.py
│   ├── pyt_utils.py
│   ├── save_image.py
│   ├── utils.py
│   └── visualizer.py
├── pytorch_ssim/
│   └── __init__.py
├── pretrained/
│   └── vmamba/...
├── logs/...
├── test_results/...
├── PCLT20K/...
├── pkdata/...
├── pk_split/
│   ├── split_info.json
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── train/...
│   ├── val/...
│   └── test/...
├── dataset/...
├── .git/
├── .gitattributes
└── .DS_Store
```

> 说明：`...` 表示该目录包含大量数据/编译产物，内容略。

---

## 2. 逐项说明（文件/目录用途与整理建议）

### 顶层文件
- `README.md` **[文档]**：项目总览、架构说明、使用方法与训练配置说明。
- `main.py` **[核心]**：训练/验证/测试入口，包含数据加载、损失、优化器、调度器、EMA、阈值扫描、TensorBoard。
- `test_model.py` **[核心]**：模型评估脚本，支持保存预测并统计多项指标。
- `run_train.sh` **[核心]**：一键训练脚本（封装常用参数）。
- `requirements.txt` **[配置]**：Python依赖。
- `generate_split_txt.py` **[核心]**：按病人ID生成 `train/val/test.txt` 的分割文件。
- `split_dataset.py` **[核心]**：数据拆分与整理工具（用于生成分割或复制数据）。
- `SPLIT_TXT_USAGE.md` **[文档]**：TXT分割文件的使用说明。
- `TRAINING_GUIDE.md` **[文档]**：训练完整指南与注意事项。
- `TRAIN_COMMAND.md` **[文档]**：训练命令示例与参数说明。
- `TRAINING_IMPROVEMENTS.md` **[文档]**：训练策略改进记录。
- `MODEL_EVALUATION.md` **[文档]**：评估指标与实验记录。
- `ARCHITECTURE_DIAGRAM.md` **[文档]**：架构详解与绘图说明。
- `ARCHITECTURE_IMPROVEMENTS.md` **[文档]**：架构改进建议与对比。
- `DIFFERENCES_WITH_CIPA.md` **[文档]**：与 CIPA 的差异说明。
- `.gitattributes` **[配置]**：Git 属性配置。
- `.DS_Store` **[缓存]**：macOS 生成的缓存文件，可删除。

### 配置目录
- `configs/` **[配置]**：训练配置文件集合。
  - `default_config.json`：默认训练参数。
  - `large_batch_config.json`：大批次训练配置。
  - `focal_loss_config.json`：Focal Loss 训练配置。
  - `README.md`：配置说明与使用方式。

### 数据加载
- `data/dataset.py` **[核心]**：数据集类、预处理、增强与PCLT20K TXT加载逻辑。
- `data/__init__.py` **[核心]**：数据模块初始化。

### 模型目录
- `models/segmentation_loss.py` **[核心]**：Dice/Tversky/IoU/Focal/组合损失。
- `models/model.py` **[兼容]**：旧UNet基线模型（对比使用）。
- `models/train_tasks.py` **[兼容]**：旧版训练流程（保留参考）。
- `models/measure_model.py` **[辅助]**：参数量/FLOPs统计工具。
- `models/P_loss.py` **[兼容]**：感知损失（分割任务一般不使用）。

#### VMamba 主体
- `models/vmamba/builder.py` **[核心]**：模型构建入口（EncoderDecoder）。
- `models/vmamba/dual_vmamba.py` **[核心]**：双独立VMamba编码器与多层融合。
- `models/vmamba/vmamba.py` **[核心]**：VMamba核心实现（SS2D等）。
- `models/vmamba/MambaDecoder.py` **[核心]**：解码器与上采样模块。
- `models/vmamba/mamba_net_utils.py` **[核心]**：CRM/SS1D等工具模块。
- `models/vmamba/local_vmamba/` **[核心]**：
  - `region_mamba.py`：区域交互模块（DCIM）。
  - `local_scan.py`：局部扫描实现（Triton加速）。
- `models/vmamba/selective_scan/` **[核心/编译]**：
  - `setup.py`：CUDA扩展编译脚本。
  - `csrc/`：CUDA源码。
  - `selective_scan/`：Python接口。
  - `build/`、`selective_scan.egg-info/`：编译产物 **[缓存]** 可删。
  - `test_selective_scan.py`：CUDA扩展测试。

#### 归档模型
- `models_archived/` **[兼容]**：旧版本模型实现的归档副本（可按需删除）。

### 工具目录
`utils/` **[核心/辅助]**：
- `ema.py`：EMA权重维护。
- `early_stopping.py`：早停机制。
- `logger.py`：日志输出封装。
- `checkpoint.py`：检查点保存/加载。
- `save_image.py`、`visualizer.py`：可视化/保存输出。
- `init_func.py`、`pyt_utils.py`、`utils.py`、`attention.py`：通用工具。

### 其他模块
- `pytorch_ssim/__init__.py` **[辅助]**：SSIM相关工具（可能用于实验）。

### 数据与权重目录
以下目录属于数据或权重，**可按需要独立迁移**：
- `pretrained/vmamba/` **[权重]**：VMamba预训练权重。
- `PCLT20K/` **[数据]**：原始PCLT20K数据集。
- `pkdata/` **[数据]**：训练所用数据集（具体内容视你的整理）。
- `pk_split/` **[数据]**：含 `train/val/test.txt` 与拆分后的样本或结构化数据。
- `dataset/` **[数据]**：自定义/整理后的数据目录（需核对具体内容）。

### 运行产出目录（可清理）
- `logs/` **[产出]**：训练日志、checkpoint、TensorBoard。
- `test_results/` **[产出]**：推理/评估结果与预测输出。

### 缓存目录
- `__pycache__/` **[缓存]**：Python缓存目录，可删除。
- `models/vmamba/selective_scan/build/` **[缓存]**：CUDA扩展编译产物，可删除后重新编译。

---

## 3. 整理建议（按“保留/可移/可删”）

**建议保留（核心代码 + 文档）**  
`main.py`, `test_model.py`, `models/`, `data/`, `utils/`, `configs/`, `README.md` 及各类说明文档。

**建议单独归档（数据/权重）**  
`PCLT20K/`, `pkdata/`, `pk_split/`, `dataset/`, `pretrained/`  
可移动到统一数据盘，保留目录结构与 `train/val/test.txt`。

**可清理（运行产物/缓存）**  
`logs/`, `test_results/`, `__pycache__/`, `selective_scan/build/`, `.DS_Store`

---

## 4. 快速索引（整理时常用）
- 训练入口：`main.py`
- 评估入口：`test_model.py`
- 数据加载：`data/dataset.py`
- 损失函数：`models/segmentation_loss.py`
- 编码器/融合：`models/vmamba/dual_vmamba.py`
- 解码器：`models/vmamba/MambaDecoder.py`
- CUDA扩展：`models/vmamba/selective_scan/`
- 配置样例：`configs/*.json`
- 训练日志：`logs/`
- 预测结果：`test_results/`

