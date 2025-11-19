#!/bin/bash

# Azure ML DriveLM作业提交脚本
# 使用Azure CLI提交到768核NC 96A100

echo "🌐 Azure ML DriveLM作业提交"
echo "=================================="

# 设置变量
SUBSCRIPTION_ID="0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
RESOURCE_GROUP="your-ml-resource-group"  # 请修改为实际资源组
WORKSPACE_NAME="your-ml-workspace"       # 请修改为实际工作区
COMPUTE_NAME="drivelm-a100-cluster"
EXPERIMENT_NAME="drivelm-ghost-probing"

# 登录Azure (如果需要)
echo "🔑 检查Azure登录状态..."
az account show --subscription $SUBSCRIPTION_ID || az login

# 设置默认订阅
az account set --subscription $SUBSCRIPTION_ID

# 创建计算集群 (如果不存在)
echo "🖥️ 创建/检查GPU计算集群..."
az ml compute create \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE_NAME \
    --name $COMPUTE_NAME \
    --type amlcompute \
    --size Standard_NC96ads_A100_v4 \
    --min-instances 0 \
    --max-instances 2 \
    --idle-time-before-scale-down 1800

# 上传代码和数据
echo "📤 准备代码和数据..."
# 这里需要将DADA-2000视频和脚本上传到Azure ML

# 提交作业
echo "🚀 提交DriveLM处理作业..."
az ml job create \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE_NAME \
    --file drivelm_job.yml

echo "✅ 作业提交完成！"
echo "🔗 请在Azure ML Studio中监控作业进度"
