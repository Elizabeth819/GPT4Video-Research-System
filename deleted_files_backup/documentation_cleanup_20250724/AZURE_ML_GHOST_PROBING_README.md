# Azure ML Ghost Probing Detection with GPT-4.1 Balanced

本文档介绍如何使用Azure ML A100 GPU和平衡版GPT-4.1 prompt处理100个视频进行鬼探头检测。

## 📋 项目概述

- **目标**: 使用GPT-4.1 balanced版本的prompt处理images_1_001到images_5_XXX的100个视频
- **平台**: Azure ML A100 GPU
- **输出**: 与groundtruth.txt格式一致的结果，便于准确率、精确度、召回率等指标对比
- **基础**: 基于BALANCED_GPT41_PROMPT_FINAL.md中的配置

## 🚀 快速开始

### 1. 环境设置

```bash
# 运行环境检查和设置
python setup_azure_ghost_probing.py

# 这将检查:
# - Python依赖项
# - 必要的文件
# - 环境变量配置
# - 数据文件
# - Azure和OpenAI连接
```

### 2. 配置环境变量

复制生成的`.env.template`文件为`.env`并填写配置:

```bash
# Azure OpenAI配置 (必需)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
VISION_DEPLOYMENT_NAME=gpt-4.1

# Azure Whisper配置 (必需)
AZURE_WHISPER_KEY=your_azure_whisper_key
AZURE_WHISPER_DEPLOYMENT=your_whisper_deployment
AZURE_WHISPER_ENDPOINT=https://your-whisper-endpoint.cognitiveservices.azure.com

# Azure ML配置 (可选)
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
AZURE_WORKSPACE_NAME=your_workspace_name
AZURE_COMPUTE_NAME=gpu-cluster-a100
```

### 3. 本地测试

```bash
# 预览将处理的视频
python batch_ghost_probing_gpt41_balanced.py --dry-run

# 处理少量视频进行测试
python batch_ghost_probing_gpt41_balanced.py --max-videos 5
```

### 4. 提交到Azure ML

```bash
# 方法1: 使用生成的脚本
./submit_ghost_probing_job.sh

# 方法2: 直接使用Python
python submit_azure_ghost_probing_job.py \
    --subscription-id "$AZURE_SUBSCRIPTION_ID" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --workspace-name "$AZURE_WORKSPACE_NAME" \
    --compute-name gpu-cluster-a100
```

## 📊 输出格式

### 结果文件结构
```
result/ghost_probing_gpt41_balanced/
├── ghost_probing_results_20240118_143022.json      # 详细JSON结果
├── ghost_probing_comparison_20240118_143022.csv    # 对比CSV文件
├── ghost_probing_report_20240118_143022.md         # 处理报告
└── intermediate_results_20240118_143022.json       # 中间结果备份
```

### 输出格式说明

**CSV文件格式** (与groundtruth.txt一致):
```csv
video_id	predicted_label	ground_truth_label	processing_status
images_1_001.avi	none	none	success
images_1_002.avi	5s: ghost probing	5s: ghost probing	success
images_1_003.avi	2s: ghost probing	2s: ghost probing	success
```

**JSON文件格式** (详细信息):
```json
{
  "video_id": "images_1_002.avi",
  "predicted_label": "5s: ghost probing",
  "ground_truth_label": "5s: ghost probing",
  "detection_result": "ghost probing",
  "timestamp": "5s",
  "processing_status": "success"
}
```

## 📈 性能指标

处理完成后会自动计算以下指标:

- **准确率 (Accuracy)**: 整体预测正确率
- **精确度 (Precision)**: 预测为鬼探头的正确率
- **召回率 (Recall)**: 实际鬼探头的检出率
- **F1分数**: 精确度和召回率的调和平均
- **误报率**: 错误预测为鬼探头的比率

## 🔧 技术配置

### GPT-4.1 Balanced Prompt特点

1. **分层判断机制**:
   - 高确信度鬼探头: `"ghost probing"`
   - 潜在鬼探头: `"potential ghost probing"`
   - 正常交通: 描述性语言

2. **环境上下文理解**:
   - 高风险环境(高速路、郊区): 更敏感
   - 低风险环境(交叉口、人行横道): 更谨慎

3. **误报控制策略**:
   - 严格距离要求(<3米)
   - 瞬间出现特征
   - 排除预期行为

### Azure ML配置

- **计算资源**: Standard_NC24ads_A100_v4 (A100 GPU)
- **环境**: CUDA 11.8 + cuDNN 8 + PyTorch 2.0
- **超时设置**: 4小时
- **并发处理**: 支持多进程

## 📁 文件说明

| 文件名 | 说明 |
|--------|------|
| `batch_ghost_probing_gpt41_balanced.py` | 主要批处理脚本 |
| `submit_azure_ghost_probing_job.py` | Azure ML作业提交脚本 |
| `setup_azure_ghost_probing.py` | 环境设置和检查脚本 |
| `azure_ghost_probing_env.yml` | Conda环境配置 |
| `azure_ml_ghost_probing_gpt41_config.yml` | Azure ML作业配置 |
| `BALANCED_GPT41_PROMPT_FINAL.md` | Prompt详细说明 |

## 🎯 预期性能

基于之前99个视频的测试结果:

- **F1分数**: 0.712
- **召回率**: 96.3%
- **精确度**: 56.5%
- **准确率**: 57.6%
- **误报率**: 88.9%

## 🔍 故障排除

### 常见问题

1. **环境变量未设置**
   ```bash
   # 检查环境变量
   python setup_azure_ghost_probing.py --test-connection
   ```

2. **Azure认证失败**
   ```bash
   # 使用Azure CLI登录
   az login
   ```

3. **GPU资源不足**
   ```bash
   # 检查Azure ML计算资源
   az ml compute show -n gpu-cluster-a100 -w your-workspace -g your-resource-group
   ```

4. **API配额限制**
   - 检查Azure OpenAI配额
   - 调整批处理大小
   - 设置重试策略

### 日志文件

- `ghost_probing_batch.log`: 批处理日志
- Azure ML作业日志: 在Azure ML Studio中查看

## 📞 支持

如遇问题，请检查:
1. 环境变量配置是否正确
2. Azure ML资源是否可用
3. API配额是否充足
4. 网络连接是否正常

## 🔄 监控作业

```bash
# 监控特定作业
python submit_azure_ghost_probing_job.py --monitor-only job-name

# 下载作业结果
python submit_azure_ghost_probing_job.py --download-only job-name
```

## 📋 检查清单

使用前请确认:

- [ ] 环境变量已正确配置
- [ ] DADA-2000-videos文件夹包含目标视频
- [ ] result/groundtruth_labels.csv文件存在
- [ ] Azure ML workspace和compute已创建
- [ ] Azure OpenAI和Whisper API可用
- [ ] 网络连接正常
- [ ] 运行权限充足

---

**注意**: 本项目严格遵循CLAUDE.md中的规则，不会创建虚假数据或模拟结果。所有输出都基于实际的视频处理和API调用结果。