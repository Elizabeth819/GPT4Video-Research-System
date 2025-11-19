# Azure ML DriveMM推理指南

这个项目用于在Azure ML GPU集群上运行DriveMM视频推理任务。

## 🚀 快速开始

### 1. 配置Azure环境

首先需要配置Azure订阅信息。复制 `config.json.example` 为 `config.json` 并填入你的信息：

```json
{
  "subscription_id": "你的订阅ID",
  "resource_group": "你的资源组",
  "workspace_name": "你的工作区名称",
  "compute_target": "drivemm-a100-cluster",
  "experiment_name": "drivemm-inference"
}
```

### 2. 设置Azure Storage连接

设置环境变量：

```bash
export AZURE_STORAGE_CONNECTION_STRING="你的存储连接字符串"
```

### 3. 运行推理

#### 方法1: 使用Azure CLI（推荐）

```bash
# 登录Azure
az login

# 设置订阅
az account set --subscription <你的订阅ID>

# 提交作业
az ml job create --file azure_ml_drivemm_real_job.yml \
  --workspace-name <工作区名称> \
  --resource-group <资源组名称>
```

#### 方法2: 使用Python脚本

```bash
# 运行设置向导
python run_drivemm_azure.py

# 或直接使用SDK提交
python setup_drivemm_azure.py
```

## 📊 GPU要求

### DriveMM模型规格
- **参数量**: 8.45B (84.5亿参数)
- **模型大小**: ~17GB
- **GPU内存需求**: ~22-25GB VRAM

### 推荐配置
- ✅ **Standard_NC24ads_A100_v4** (1x A100 40GB) - 推荐
- ✅ **Standard_NC48ads_A100_v4** (2x A100 40GB) - 更快
- ✅ **Standard_NC96ads_A100_v4** (4x A100 40GB) - 最佳性能

### 成本估算
- Standard_NC24ads_A100_v4: ~$3.67/小时
- 预计推理时间: 2-4小时
- 预计总成本: $7-15

## 📁 项目结构

```
.
├── azure_drivemm_real_inference.py    # DriveMM推理脚本
├── azure_ml_drivemm_real_job.yml      # Azure ML作业配置
├── azure_drivemm_environment.yml       # Conda环境配置
├── run_drivemm_azure.py               # 设置向导脚本
├── setup_drivemm_azure.py             # SDK提交脚本
└── config.json                         # Azure配置（不要提交）
```

## 🔧 主要文件说明

### `azure_drivemm_real_inference.py`
真实DriveMM模型推理脚本：
- 从Azure Storage读取视频（dada-videos容器）
- 使用DriveMM模型进行推理
- 输出结果到JSON文件

### `azure_ml_drivemm_real_job.yml`
Azure ML作业配置文件：
- 指定GPU集群
- 配置环境和依赖
- 设置输入输出路径

### `azure_drivemm_environment.yml`
Conda环境配置：
- PyTorch with CUDA
- Transformers
- Azure SDK

## 📊 监控作业

### 查看作业状态
```bash
az ml job show --name <作业名称> \
  --workspace-name <工作区> \
  --resource-group <资源组>
```

### 查看日志
```bash
az ml job logs --name <作业名称> \
  --workspace-name <工作区> \
  --resource-group <资源组>
```

### Azure ML Studio
访问 https://ml.azure.com 在图形界面中监控

## 🛠️ 故障排除

### GPU配额不足
联系Azure支持申请A100配额

### 模型下载失败
- 检查网络连接
- 确认HuggingFace权限
- 可能需要设置HF_TOKEN

### 内存不足
- 增加作业配置中的shm_size
- 使用更大的VM规格

### 存储连接问题
- 确认存储连接字符串正确
- 检查容器名称（dada-videos）
- 验证网络访问权限

## 📝 输出结果

推理完成后会生成：
- `azure_drivemm_real_inference_results.json` - 完整推理结果
- 每个视频的详细分析
- Ghost probing检测结果
- 统计汇总信息

## ⚠️ 注意事项

1. **不要提交敏感信息**: `config.json` 和包含密钥的文件已在 `.gitignore` 中
2. **成本控制**: 记得及时停止不需要的集群
3. **数据安全**: 确保存储账户访问权限设置正确
4. **版本兼容**: 使用指定的PyTorch和CUDA版本

## 🤝 贡献

如有问题或改进建议，请提交Issue或Pull Request。

## 📄 License

内部使用项目
