#!/usr/bin/env python3
"""
VideoChat2 A100 GPU成本估算脚本
"""

def show_cost_estimation():
    """显示成本估算"""
    print("🎯 VideoChat2 A100 GPU 成本估算 (100个视频)")
    print("=" * 60)
    
    # 基础参数
    video_count = 100
    gpu_type = "Standard_NC24ads_A100_v4"
    priority = "Low Priority"
    
    # 成本估算
    hourly_cost_low = 1.0  # USD per hour
    hourly_cost_high = 2.0  # USD per hour
    
    # 时间估算 (秒)
    seconds_per_video_low = 2  # 2 seconds per video
    seconds_per_video_high = 6  # 6 seconds per video
    
    total_minutes_low = video_count * seconds_per_video_low / 60
    total_minutes_high = video_count * seconds_per_video_high / 60
    
    # 计算成本 (分钟转小时)
    total_cost_low = (video_count * seconds_per_video_low / 3600) * hourly_cost_low
    total_cost_high = (video_count * seconds_per_video_high / 3600) * hourly_cost_high
    
    print(f"🖥️  GPU配置: {gpu_type}")
    print(f"⚡ 优先级: {priority}")
    print(f"🎬 视频数量: {video_count} 个")
    print(f"📁 视频范围: images_1_001 ~ images_5_XXX (前100个)")
    print("")
    
    print("⏱️  预计处理时间:")
    print(f"   - 最快: {total_minutes_low:.1f} 分钟")
    print(f"   - 最慢: {total_minutes_high:.1f} 分钟")
    print("")
    
    print("💰 预计成本:")
    print(f"   - 最低: ${total_cost_low:.2f} USD")
    print(f"   - 最高: ${total_cost_high:.2f} USD")
    print("")
    
    # 与全量对比
    full_video_count = 1019
    full_cost_low = full_video_count * seconds_per_video_low / 3600 * hourly_cost_low
    full_cost_high = full_video_count * seconds_per_video_high / 3600 * hourly_cost_high
    
    time_savings = (1 - video_count / full_video_count) * 100
    cost_savings_low = (1 - total_cost_low / full_cost_low) * 100
    cost_savings_high = (1 - total_cost_high / full_cost_high) * 100
    
    print(f"📊 相比全量({full_video_count}个视频)的节省:")
    print(f"   - 时间节省: {time_savings:.1f}%")
    print(f"   - 成本节省: {cost_savings_low:.1f}% ~ {cost_savings_high:.1f}%")
    print(f"   - 全量成本: ${full_cost_low:.2f} ~ ${full_cost_high:.2f} USD")
    print("")
    
    print("📋 视频分布:")
    video_distribution = {
        "images_1_*": 27,
        "images_2_*": 4, 
        "images_3_*": 7,
        "images_4_*": 8,
        "images_5_*": 79
    }
    
    selected_count = 0
    for pattern, count in video_distribution.items():
        if selected_count + count <= 100:
            actual_count = count
            selected_count += count
        else:
            actual_count = 100 - selected_count
            selected_count = 100
            
        print(f"   - {pattern}: {actual_count:2d} 个视频")
        if selected_count >= 100:
            break
    
    print("")
    print("⚠️  注意事项:")
    print("   - Low Priority实例可能被抢占，导致作业重启")
    print("   - 系统会自动从断点恢复，不影响最终结果")
    print("   - 相比Regular实例，Low Priority节省约80%成本")
    print("   - 建议在非高峰时段提交作业")
    
    return {
        'video_count': video_count,
        'estimated_cost_range': (total_cost_low, total_cost_high),
        'estimated_time_range': (total_minutes_low, total_minutes_high),
        'savings_percentage': time_savings
    }

if __name__ == "__main__":
    estimation = show_cost_estimation()
    
    print("\n" + "=" * 60)
    print("🚀 准备就绪！运行以下命令开始部署:")
    print("   export AZURE_SUBSCRIPTION_ID='your-subscription-id'")
    print("   export AZURE_RESOURCE_GROUP='your-resource-group'")
    print("   export AZURE_WORKSPACE_NAME='your-workspace-name'")
    print("   ./quick_start_videochat2_gpu.sh deploy")
    print("=" * 60)