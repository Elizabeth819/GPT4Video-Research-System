# LLaVA-NeXT 鬼探头视频打标系统

基于LLaVA-Video-7B-Qwen2模型的鬼探头检测与视频打标系统，使用与GPT-4.1相同的平衡提示词确保评估一致性。

## 🎯 项目概述

本项目利用LLaVA-NeXT的多模态视频理解能力，对DADA-100视频数据集进行鬼探头（Ghost Probing）检测和打标。系统设计遵循以下原则：

- **完全开源**：基于LLaVA-NeXT开源模型，可本地部署
- **评估一致性**：使用与GPT-4.1相同的平衡提示词
- **生产就绪**：包含完整的批处理、评估和部署系统
- **Azure ML集成**：支持在Azure ML A100集群上运行

## 📁 项目结构

```
LlaVA/
├── llava_ghost_probing_detector.py      # 核心检测器
├── llava_ghost_probing_batch.py         # 批处理脚本
├── llava_ghost_probing_evaluation.py    # 评估脚本
├── azure_ml_llava_ghost_probing.yml     # Azure ML配置
├── submit_azure_llava_job.py            # Azure ML作业提交
├── test_single_video.py                 # 单视频测试
├── requirements.txt                     # 依赖包列表
├── README.md                            # 项目文档
└── LLaVA-NeXT/                          # LLaVA-NeXT源码
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 安装PyTorch (CUDA 11.7版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 安装视频处理库
pip install decord
```

### 2. 测试单个视频

```bash
# 测试模型加载
python test_single_video.py --model-test-only

# 测试单个视频
python test_single_video.py --video /path/to/video.avi --output result.json

# 运行综合测试
python test_single_video.py --comprehensive
```

### 3. 批量处理

```bash
# 本地批处理
python llava_ghost_probing_batch.py \
    --video-folder /path/to/DADA-100-videos \
    --output-folder ./results \
    --limit 10 \
    --save-interval 5

# Azure ML批处理
python submit_azure_llava_job.py --action submit --limit 100
```

### 4. 结果评估

```bash
# 评估与ground truth对比
python llava_ghost_probing_evaluation.py \
    --llava-results ./results/llava_ghost_probing_final_TIMESTAMP.json \
    --groundtruth-file /path/to/groundtruth_labels.csv \
    --output-folder ./evaluation_results
```

## 📊 系统特性

### 核心检测功能

- **视频理解**：基于LLaVA-Video-7B-Qwen2的多帧视频分析
- **三级分类**：HIGH-CONFIDENCE、POTENTIAL、NORMAL鬼探头检测
- **环境感知**：区分高速公路、交叉口、停车场等不同场景
- **距离阈值**：<3米高信度，3-5米潜在鬼探头

### 技术架构

- **模型**：LLaVA-Video-7B-Qwen2
- **框架**：LLaVA-NeXT + PyTorch
- **视频处理**：Decord + OpenCV
- **批处理**：支持断点续传和错误恢复
- **评估**：准确率、精确率、召回率、F1分数

### Azure ML集成

- **计算集群**：llava-a100-low-priority (A100 GPU)
- **环境**：AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10
- **存储**：Azure Blob Storage数据集
- **监控**：Azure ML Studio实时监控

## 🎭 提示词设计

使用与GPT-4.1完全相同的平衡提示词：

```python
# 三级分类系统
1. HIGH-CONFIDENCE Ghost Probing (use "ghost probing")
   - <3米极近距离突然出现
   - 来自盲区(停车位、建筑物、角落)
   - 高风险环境(高速公路、乡村道路)

2. POTENTIAL Ghost Probing (use "potential ghost probing")  
   - 3-5米中等距离突然出现
   - 在存在一定不可预测性的环境中

3. NORMAL Traffic (descriptive terms)
   - 预期的交通行为
   - 交叉口、人行横道的正常通行
```

## 📈 性能目标

基于GPT-4.1 Balanced的基准性能：

| 指标 | GPT-4.1 Balanced | LLaVA目标 |
|------|------------------|-----------|
| F1 Score | 0.712 | ≥0.650 |
| Recall | 96.3% | ≥90% |
| Precision | 56.5% | ≥50% |
| Accuracy | 57.6% | ≥55% |

## 🔧 配置说明

### 环境变量

```bash
# CUDA设备
export CUDA_VISIBLE_DEVICES=0

# Hugging Face缓存
export HF_HOME=/tmp/huggingface
export TRANSFORMERS_CACHE=/tmp/transformers

# Python路径
export PYTHONPATH=/path/to/LlaVA:/path/to/LlaVA-NeXT
```

### Azure ML配置

```yaml
# azure_ml_llava_ghost_probing.yml
compute: azureml:llava-a100-low-priority
environment: azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10
inputs:
  video_data:
    path: azureml://datastores/workspaceblobstore/paths/DADA-100-videos/
outputs:
  results:
    path: azureml://datastores/workspaceblobstore/paths/llava-ghost-probing-results/
```

## 📝 使用示例

### 单视频检测

```python
from llava_ghost_probing_detector import LLaVAGhostProbingDetector

# 初始化检测器
detector = LLaVAGhostProbingDetector()

# 分析视频
result = detector.analyze_video("video.avi", "video_001")

# 提取标签
label, confidence = detector.extract_ghost_probing_label(result)
print(f"检测结果: {label} (置信度: {confidence})")
```

### 批量处理

```python
from llava_ghost_probing_batch import LLaVAGhostProbingBatchProcessor

# 创建批处理器
processor = LLaVAGhostProbingBatchProcessor(
    video_folder="/path/to/videos",
    output_folder="/path/to/results"
)

# 开始批处理
stats = processor.process_batch(limit=100, save_interval=10)
```

### 结果评估

```python
from llava_ghost_probing_evaluation import LLaVAGhostProbingEvaluator

# 创建评估器
evaluator = LLaVAGhostProbingEvaluator()

# 评估结果
metrics = evaluator.evaluate_results("results.json")
print(f"F1 Score: {metrics['performance_metrics']['f1_score']}")
```

## 📊 输出格式

### 检测结果JSON

```json
{
  "video_id": "images_1_001",
  "ghost_probing_label": "ghost_probing",
  "confidence": 0.9,
  "llava_analysis": {
    "summary": "车辆从盲区突然出现...",
    "key_actions": "ghost probing",
    "key_objects": "1) 前方: 突然出现的车辆, 2米, 高撞击风险",
    "next_action": {
      "speed_control": "rapid deceleration",
      "direction_control": "turn left",
      "lane_control": "change left"
    }
  }
}
```

### 评估指标

```json
{
  "performance_metrics": {
    "accuracy": 0.8750,
    "precision": 0.7500,
    "recall": 0.9000,
    "f1_score": 0.8182
  },
  "confusion_matrix": {
    "true_negatives": 45,
    "false_positives": 10,
    "false_negatives": 5,
    "true_positives": 40
  }
}
```

## 🚨 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少max_frames参数
   detector = LLaVAGhostProbingDetector(max_frames=32)
   ```

2. **模型加载失败**
   ```bash
   # 检查网络连接和Hugging Face访问
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **视频格式不支持**
   ```bash
   # 转换视频格式
   ffmpeg -i input.mp4 -c:v libx264 output.avi
   ```

### Azure ML问题

1. **计算集群不可用**
   - 检查集群状态和配额
   - 尝试其他可用集群

2. **作业失败**
   ```bash
   # 检查作业日志
   python submit_azure_llava_job.py --action status --job-name JOB_NAME
   ```

## 🤝 贡献指南

1. **代码规范**：遵循PEP 8标准
2. **测试要求**：新功能需包含单元测试
3. **文档更新**：更新相关文档和README
4. **性能评估**：确保不降低现有性能指标

## 📄 许可证

本项目基于原始GPT4Video-cobra-auto项目许可证。LLaVA-NeXT组件遵循其原始Apache 2.0许可证。

## 🔗 相关链接

- [LLaVA-NeXT GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [LLaVA-Video Paper](http://arxiv.org/abs/2410.02713)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [DADA-2000 Dataset](https://github.com/JWFangit/LOTVS-DADA)

## 📧 联系信息

如有问题或建议，请通过以下方式联系：

- 项目Issue：GitHub Issues
- 技术讨论：项目Wiki
- 模型问题：LLaVA-NeXT官方repo