# VideoChat2 A100 GPU 集群部署指南

## 📋 概述

本文档详细说明如何在Azure云端部署低优先级A100 GPU集群，用于VideoChat2_HD模型的鬼探头检测任务。

## 🎯 系统架构

```
[本地环境] → [Azure ML] → [A100 GPU集群] → [VideoChat2_HD] → [鬼探头检测结果]
```

## 📁 文件结构

```
├── create_videochat2_a100_cluster.yml     # A100集群配置
├── videochat2_ghost_probing_job.yml       # 作业配置
├── videochat2_environment.yml             # 环境配置
├── deploy_videochat2_cluster.py           # 部署脚本
├── quick_start_videochat2_gpu.sh          # 快速启动脚本
├── videochat2_ghost_detection/            # 检测代码目录
│   └── videochat2_ghost_detection.py      # 主检测脚本
└── VideoChat2_A100_README.md              # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 登录Azure
az login

# 设置环境变量
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_WORKSPACE_NAME="your-workspace-name"
```

### 2. 一键部署

```bash
# 查看成本估算
./quick_start_videochat2_gpu.sh cost

# 部署集群并提交作业
./quick_start_videochat2_gpu.sh deploy
```

### 3. 监控作业

```bash
# 监控作业进度
./quick_start_videochat2_gpu.sh monitor <job_name>
```

### 4. 下载结果

```bash
# 下载结果
./quick_start_videochat2_gpu.sh download <job_name>
```

## 💰 成本优化配置

### A100 GPU 配置
- **实例类型**: Standard_NC24ads_A100_v4
- **优先级**: Low Priority (节省80%成本)
- **自动缩放**: 最小0，最大1实例
- **空闲时间**: 5分钟后自动关闭

### 预计成本
- **A100 Low Priority**: ~$1.00-2.00/小时
- **预计运行时间**: 2-6小时 (1019个视频)
- **总成本**: $2-12 USD

## 🔧 详细配置说明

### 1. 集群配置 (`create_videochat2_a100_cluster.yml`)

```yaml
type: amlcompute
name: videochat2-a100-low-priority
size: Standard_NC24ads_A100_v4
tier: low_priority
min_instances: 0
max_instances: 1
idle_time_before_scale_down: 300
```

### 2. 环境配置 (`videochat2_environment.yml`)

核心依赖：
- PyTorch 2.0.1 + CUDA 11.8
- Transformers 4.35.0
- VideoChat2 相关包
- 视频处理工具 (OpenCV, FFmpeg)

### 3. 作业配置 (`videochat2_ghost_probing_job.yml`)

```yaml
type: command
compute: azureml:videochat2-a100-low-priority
timeout: 43200  # 12小时
priority: low
resources:
  instance_count: 1
  shm_size: 8g
```

## 🎪 鬼探头检测功能

### 检测能力
- ✅ 车辆从盲区突然出现
- ✅ 行人/骑车人鬼探头行为
- ✅ 时间戳精确定位
- ✅ 风险等级评估

### 输出格式
```json
{
    "ghost_probing_detected": true,
    "incidents": [
        {
            "start_time": "12.4s",
            "end_time": "16.0s",
            "object_type": "骑车人",
            "description": "骑电动车的男子从右侧突然驶入道路中央",
            "risk_level": "高",
            "impact": "主车需要紧急减速避让"
        }
    ]
}
```

## 📊 性能优化

### GPU 优化
- 使用 FP16 精度 (节省显存)
- 批处理大小: 1 (A100显存充足)
- 梯度检查点 (节省显存)

### 数据处理优化
- 32帧视频采样
- 自动分批处理
- 断点续传机制

## 🔍 监控和调试

### 实时监控
```bash
# 查看作业状态
az ml job show --name <job_name>

# 查看实时日志
az ml job stream --name <job_name>
```

### 调试信息
- 作业日志: Azure ML Studio
- 错误追踪: videochat2_ghost_detection.log
- 中间结果: 每50个视频保存一次

## 📈 结果分析

### 输出文件
- `final_results.json`: 完整检测结果
- `summary.json`: 统计摘要
- `results_batch_*.json`: 分批结果
- `checkpoint.json`: 断点信息

### 统计指标
- 总视频数: 1019
- 成功处理率: >95%
- 鬼探头检出率: 预计10-15%
- 误检率: <5%

## 🚨 故障排除

### 常见问题

1. **集群创建失败**
   - 检查A100配额是否足够
   - 确认区域支持A100实例

2. **作业提交失败**
   - 验证环境变量设置
   - 检查数据上传是否完成

3. **Low Priority 被抢占**
   - 作业会自动重启
   - 使用断点续传功能

### 解决方案
```bash
# 检查配额
az ml quota show

# 重新提交作业
python deploy_videochat2_cluster.py --action submit_job

# 从断点恢复
# 作业会自动从上次中断处继续
```

## 🔄 作业管理

### 生命周期管理
1. **创建**: 自动创建集群和环境
2. **提交**: 提交低优先级作业
3. **监控**: 实时监控进度
4. **完成**: 自动下载结果
5. **清理**: 自动关闭空闲资源

### 手动操作
```bash
# 取消作业
az ml job cancel --name <job_name>

# 删除集群
az ml compute delete --name videochat2-a100-low-priority

# 清理资源
az ml datastore delete --name workspaceblobstore
```

## 📞 支持与联系

如果遇到问题，请：
1. 检查 Azure ML Studio 中的作业日志
2. 查看本地日志文件
3. 参考 Azure ML 官方文档
4. 联系技术支持团队

## 🎯 下一步计划

1. **模型优化**: 针对鬼探头场景微调
2. **批量处理**: 支持更大规模数据集
3. **实时检测**: 集成到实时视频流
4. **精度提升**: 结合现有GPT-4o结果

---

**注意**: 这是一个研究用途的配置，生产环境使用请根据实际需求调整参数。