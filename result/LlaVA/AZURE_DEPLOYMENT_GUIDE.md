# 🚀 Azure ML LLaVA鬼探头检测部署指南

本指南将帮助您在Azure ML上成功运行100个DADA视频的LLaVA鬼探头检测批处理。

## 📋 前置检查清单

### 1. 运行预检查脚本

```bash
cd /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA
python azure_ml_precheck.py
```

**确保所有检查通过**，特别是：
- ✅ Azure CLI已安装并登录
- ✅ Azure ML SDK已安装  
- ✅ 工作区连接正常
- ✅ 计算集群可用
- ✅ 所有代码文件存在

### 2. 验证Azure配置

```bash
# 检查当前登录账户
az account show

# 确认订阅ID
az account list --query "[].{Name:name, SubscriptionId:id, IsDefault:isDefault}"
```

## 🎯 执行步骤

### 步骤1: 测试运行（推荐）

首先进行测试运行，验证配置：

```bash
# Dry run模式，验证配置但不提交真实作业
python submit_azure_llava_job.py --action submit --limit 5 --dry-run
```

### 步骤2: 小规模测试

如果预检查通过，先用5个视频测试：

```bash
# 提交5个视频的测试作业
python submit_azure_llava_job.py --action submit --limit 5 --no-dry-run
```

**监控测试作业**：
```bash
# 获取作业名称后检查状态
python submit_azure_llava_job.py --action status --job-name YOUR_JOB_NAME
```

### 步骤3: 完整批处理

测试成功后，运行完整的100个视频：

```bash
# 🚀 提交100个视频的完整作业
python submit_azure_llava_job.py --action submit --limit 100 --no-dry-run
```

## 📊 预期结果

### 时间估算
- **总时间**: 2-3小时
- **每视频**: ~1-2分钟
- **中间保存**: 每5个视频保存一次

### 成本估算
- **计算资源**: Standard_NC24ads_A100_v4 (A100 GPU)
- **每小时费用**: ~$3.67
- **总成本**: $7-11 USD

### 输出文件
作业完成后将生成：
- `llava_ghost_probing_final_TIMESTAMP.json` - 完整结果
- `llava_ghost_probing_results_TIMESTAMP.csv` - CSV格式
- `llava_ghost_probing_simplified_TIMESTAMP.json` - 简化格式

## 🔍 监控和管理

### 检查作业状态

```bash
# 检查特定作业状态
python submit_azure_llava_job.py --action status --job-name JOB_NAME

# 列出最近的作业
python submit_azure_llava_job.py --action list
```

### Azure ML Studio监控

1. 访问 [Azure ML Studio](https://ml.azure.com)
2. 选择您的工作区：`llava-workspace`
3. 导航到 "作业" → "实验" → `llava-ghost-probing-experiment`
4. 实时查看日志和指标

### 下载结果

```bash
# 下载作业输出
python submit_azure_llava_job.py --action download --job-name JOB_NAME --download-path ./results
```

## 🚨 故障排除

### 常见问题

**1. 作业提交失败**
```bash
# 检查Azure登录状态
az account show

# 重新登录
az login
```

**2. 计算集群不可用**
```bash
# 检查集群状态
az ml compute list --workspace-name llava-workspace --resource-group llava-resourcegroup
```

**3. 内存不足错误**
- 检查作业日志
- 可能需要调整max_frames参数

**4. 模型下载失败**
- 检查网络连接
- Azure ML通常有很好的模型缓存

### 作业重启

如果作业失败，可以从中断点重新开始：

```bash
# 使用--start-at参数从特定视频开始
python llava_ghost_probing_batch.py --start-at 50 --limit 50
```

## 📈 性能优化建议

### 1. 分批处理
对于大规模处理，建议分批：

```bash
# 第一批：1-50
python submit_azure_llava_job.py --action submit --limit 50 --job-name llava-batch-1

# 第二批：51-100  
python submit_azure_llava_job.py --action submit --limit 50 --job-name llava-batch-2 --start-at 50
```

### 2. 并行作业
可以同时运行多个较小的作业，但注意配额限制。

### 3. 资源监控
- 监控GPU利用率
- 关注内存使用
- 检查网络I/O

## 🎯 作业命令总结

```bash
# 完整的推荐执行流程

# 1. 预检查
python azure_ml_precheck.py

# 2. 测试运行
python submit_azure_llava_job.py --action submit --limit 5 --no-dry-run

# 3. 检查测试状态
python submit_azure_llava_job.py --action status --job-name TEST_JOB_NAME

# 4. 完整批处理 (您要运行的命令)
python submit_azure_llava_job.py --action submit --limit 100 --no-dry-run

# 5. 监控作业
python submit_azure_llava_job.py --action status --job-name FULL_JOB_NAME

# 6. 下载结果
python submit_azure_llava_job.py --action download --job-name FULL_JOB_NAME
```

## 🏁 完成后的下一步

1. **下载结果文件**
2. **运行评估脚本**：
   ```bash
   python llava_ghost_probing_evaluation.py --llava-results RESULT_FILE.json
   ```
3. **生成对比报告**
4. **分析性能指标**

## 📞 支持联系

- **Azure ML问题**: Azure支持门户
- **代码问题**: 项目GitHub Issues  
- **紧急问题**: 检查Azure ML Studio日志

---

**🎉 准备就绪！现在可以运行您的命令：**

```bash
python submit_azure_llava_job.py --action submit --limit 100 --no-dry-run
```