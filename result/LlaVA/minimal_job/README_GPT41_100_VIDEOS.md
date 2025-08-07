# GPT-4.1平衡版100视频鬼探头检测

## 📋 项目说明

使用GPT-4.1平衡版prompt在Azure ML A100 GPU上处理100个DADA视频(images_1_001到images_5_xxx)，进行鬼探头检测，输出格式与GPT-4.1保持一致以便性能对比。

## 🎯 核心功能

- **平衡版Prompt**: 使用经过优化的GPT-4.1 prompt，F1分数0.712，召回率96.3%
- **100视频处理**: 处理完整的DADA数据集images_1_001到images_5_xxx
- **GPU加速**: 在Azure ML A100 GPU上运行，提供高性能处理
- **标准化输出**: 输出格式与GPT-4.1保持一致，支持直接性能对比
- **Ground Truth评估**: 自动对比ground truth计算准确率、精确度、召回率等指标

## 📁 文件结构

```
minimal_job/
├── gpt41_balanced_100_videos.py    # 主处理脚本
├── gpt41_100_videos_job.yml        # Azure ML作业配置
├── evaluate_gpt41_100_videos.py    # 评估脚本
├── README_GPT41_100_VIDEOS.md      # 本说明文件
└── outputs/
    └── results/                     # 输出结果目录
```

## ⚙️ 配置要求

### Azure OpenAI配置
在提交作业前，请在`gpt41_100_videos_job.yml`中设置正确的Azure OpenAI配置：

```yaml
environment_variables:
  AZURE_VISION_KEY: "your-actual-azure-openai-key"
  VISION_ENDPOINT: "https://your-resource.openai.azure.com/"
  VISION_DEPLOYMENT_NAME: "your-gpt4-vision-deployment"
```

### 模型参数 (已优化)
- **Frame Interval**: 10秒
- **Frames per Interval**: 10帧  
- **Temperature**: 0.3
- **Max Tokens**: 2000
- **API Version**: 2024-02-15-preview

## 🚀 运行步骤

### 1. 提交Azure ML作业

```bash
# 在当前目录下运行
az ml job create --file gpt41_100_videos_job.yml \
  --workspace-name llava-workspace \
  --resource-group llava-resourcegroup
```

### 2. 监控作业进度

```bash
# 查看作业状态
az ml job show --name <job-name> \
  --workspace-name llava-workspace \
  --resource-group llava-resourcegroup

# 查看实时日志
az ml job stream --name <job-name> \
  --workspace-name llava-workspace \
  --resource-group llava-resourcegroup
```

### 3. 下载结果

作业完成后，下载结果：
```bash
az ml job download --name <job-name> \
  --workspace-name llava-workspace \
  --resource-group llava-resourcegroup \
  --download-path ./gpt41_results
```

### 4. 评估结果

```bash
# 运行评估脚本
python evaluate_gpt41_100_videos.py
```

## 📊 预期输出

### 主要结果文件
- `gpt41_balanced_100_videos_<timestamp>.json`: 完整检测结果
- `gpt41_evaluation_detailed_<timestamp>.json`: 详细评估结果  
- `gpt41_evaluation_summary_<timestamp>.json`: GPT-4.1兼容格式摘要

### 中间结果文件
- `gpt41_balanced_intermediate_<count>_<timestamp>.json`: 每10个视频的中间保存

### 结果格式示例

```json
{
  "metadata": {
    "model": "GPT-4.1-Balanced",
    "total_videos": 100,
    "successful_videos": 98,
    "ghost_probing_detected": 15,
    "potential_ghost_probing_detected": 8
  },
  "results": [
    {
      "video_id": "images_1_001",
      "segment_id": "segment_1", 
      "Start_Timestamp": "0.0s",
      "End_Timestamp": "10.0s",
      "key_actions": "emergency braking due to ghost probing",
      "summary": "Vehicle suddenly appears from blind spot...",
      "processing_time": 12.5,
      "model": "GPT-4.1-Balanced"
    }
  ]
}
```

## 📈 性能评估

### 评估指标
- **准确率 (Accuracy)**: 总体预测正确率
- **精确度 (Precision)**: 检测出的鬼探头中真实的比例
- **召回率 (Recall)**: 真实鬼探头中被检测出的比例
- **F1分数**: 精确度和召回率的调和平均
- **特异性 (Specificity)**: 正常视频被正确识别的比例

### GPT-4.1基线对比
| 指标 | GPT-4.1基线 | 预期性能 |
|------|-------------|----------|
| F1分数 | 0.712 | ≥ 0.70 |
| 召回率 | 96.3% | ≥ 95% |
| 精确度 | 56.5% | ≥ 55% |
| 准确率 | 57.6% | ≥ 55% |

## 🔧 故障排除

### 常见问题

1. **API密钥错误**
   ```
   错误: Authentication failed
   解决: 检查AZURE_VISION_KEY是否正确设置
   ```

2. **配额限制**
   ```
   错误: Rate limit exceeded
   解决: 调整重试间隔或联系Azure支持增加配额
   ```

3. **视频文件未找到**
   ```
   错误: No video files found
   解决: 确认数据集已正确上传到Azure ML
   ```

4. **GPU内存不足**
   ```
   错误: CUDA out of memory
   解决: 减少frames_per_interval或使用更大的GPU
   ```

### 日志检查

主要日志位置：
- `user_logs/std_log.txt`: 标准输出日志
- `system_logs/`: 系统级日志
- `outputs/results/`: 结果文件

关键日志信息：
- `🎬 开始处理视频`: 视频处理开始
- `✅ 处理完成`: 单个视频处理完成
- `🚨 高置信度鬼探头检测`: 检测到高置信度鬼探头
- `⚠️ 潜在鬼探头检测`: 检测到潜在鬼探头

## 💡 优化建议

### 性能优化
1. **并行处理**: 可考虑分批处理减少单次作业时间
2. **缓存优化**: 使用Azure缓存服务减少重复API调用
3. **GPU利用**: 确保充分利用A100 GPU性能

### 准确性优化  
1. **Prompt调优**: 根据结果进一步微调prompt
2. **阈值调整**: 基于评估结果调整置信度阈值
3. **上下文增强**: 增加更多环境上下文信息

## 📞 技术支持

如遇到问题，请检查：
1. Azure ML工作区配置
2. GPU资源可用性
3. API配额和限制
4. 网络连接状态

相关文档：
- [Azure OpenAI文档](https://docs.microsoft.com/azure/cognitive-services/openai/)
- [Azure ML文档](https://docs.microsoft.com/azure/machine-learning/)
- [GPT-4.1平衡版Prompt配置](../BALANCED_GPT41_PROMPT_FINAL.md)

---

**注意**: 本项目使用Azure OpenAI服务，会产生API调用费用。请合理使用并监控费用。