# 🚀 快速开始指南 - DriveMM Azure ML推理

这是一个简化的快速开始指南，帮助你快速在Azure ML上运行DriveMM推理。

## ⚡ 5分钟快速开始

### 1️⃣ 克隆仓库
```bash
git clone https://github.com/Elizabeth819/GPT4Video-Research-System.git
cd GPT4Video-Research-System
```

### 2️⃣ 配置Azure信息
```bash
# 复制配置示例文件
cp config.json.example config.json

# 编辑config.json，填入你的Azure信息
# - subscription_id: 你的Azure订阅ID
# - resource_group: 你的资源组名称
# - workspace_name: 你的Azure ML工作区名称
```

### 3️⃣ 运行设置向导
```bash
# 查看GPU要求和配置说明
python run_drivemm_azure.py
```

### 4️⃣ 提交Azure ML作业

#### 选项A: 使用Azure CLI (推荐)
```bash
# 1. 登录Azure
az login

# 2. 设置订阅
az account set --subscription <你的订阅ID>

# 3. 设置存储连接字符串 (从Azure Portal获取)
export AZURE_STORAGE_CONNECTION_STRING='DefaultEndpointsProtocol=https;AccountName=...'

# 4. 提交作业
az ml job create --file azure_ml_drivemm_real_job.yml \
  --workspace-name <工作区名称> \
  --resource-group <资源组名称>
```

#### 选项B: 使用Python SDK
```bash
# 1. 安装依赖
pip install azure-ai-ml azure-identity

# 2. 设置存储连接字符串
export AZURE_STORAGE_CONNECTION_STRING='DefaultEndpointsProtocol=https;AccountName=...'

# 3. 运行提交脚本
python setup_drivemm_azure.py
```

### 5️⃣ 监控作业
访问 https://ml.azure.com 查看作业进度

## 📋 关键信息

### GPU要求
- **推荐**: Standard_NC24ads_A100_v4 (1x A100 40GB)
- **模型大小**: ~17GB
- **内存需求**: ~22-25GB VRAM
- **成本**: ~$3.67/小时

### 视频来源
- **容器**: dada-videos (Azure Storage)
- **自动读取**: 脚本会自动从storage account读取所有视频

### 输出结果
- 推理结果将保存为JSON文件
- 包含每个视频的详细分析
- 可下载后进行对比分析

## ⚠️ 注意事项

1. **配额检查**: 确保你有足够的A100 GPU配额
2. **成本控制**: 推理完成后记得停止集群
3. **敏感信息**: 不要提交`config.json`到Git
4. **存储连接**: 确保设置了正确的存储连接字符串

## 🆘 需要帮助？

- 查看详细文档: [README_AZURE_ML.md](README_AZURE_ML.md)
- GPU要求分析: 运行 `python run_drivemm_azure.py`
- 问题排查: 查看Azure ML Studio中的日志

## 📞 联系方式

如有问题请联系项目维护者。

---

💡 **提示**: 第一次运行建议使用Azure CLI方式，更容易调试和监控。
