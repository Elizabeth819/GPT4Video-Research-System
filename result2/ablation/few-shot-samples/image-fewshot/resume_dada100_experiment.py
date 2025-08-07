#!/usr/bin/env python3
"""
Resume DADA-100 Few-shot Ablation Experiment
恢复DADA-100消融实验从失败的视频继续处理
"""

import os
import sys
import glob
from dada100_ablation_experiment import DADA100AblationExperiment

def find_latest_experiment():
    """找到最新的实验目录"""
    base_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
    experiment_dirs = glob.glob(os.path.join(base_dir, "run_dada100_image_fewshot_*"))
    
    if not experiment_dirs:
        print("❌ 未找到任何实验目录")
        return None
    
    # 按修改时间排序，获取最新的
    latest_dir = max(experiment_dirs, key=os.path.getmtime)
    experiment_name = os.path.basename(latest_dir)
    
    print(f"📁 找到最新实验目录: {experiment_name}")
    return experiment_name

def find_resume_point(experiment_name):
    """找到实验的恢复点"""
    base_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
    results_dir = os.path.join(base_dir, experiment_name, "results")
    
    if not os.path.exists(results_dir):
        print(f"❌ 结果目录不存在: {results_dir}")
        return 0
    
    # 统计已处理的视频数量(排除.开头的隐藏文件)
    result_files = glob.glob(os.path.join(results_dir, "actionSummary_*.json"))
    processed_count = len(result_files)
    
    print(f"📊 已处理视频: {processed_count} 个")
    
    if processed_count > 0:
        # 显示已处理的视频列表
        processed_videos = []
        for result_file in result_files:
            filename = os.path.basename(result_file)
            video_id = filename.replace("actionSummary_", "").replace(".json", "")
            processed_videos.append(video_id)
        
        processed_videos.sort()
        print("✅ 已处理的视频:")
        for video_id in processed_videos[:5]:  # 只显示前5个
            print(f"   {video_id}")
        if len(processed_videos) > 5:
            print(f"   ... 以及其他{len(processed_videos)-5}个")
        print(f"   最后处理: {processed_videos[-1]}")
    
    return processed_count

def resume_experiment():
    """恢复实验"""
    print("🔄 恢复DADA-100 Few-shot消融实验")
    
    # 找到最新的实验
    experiment_name = find_latest_experiment()
    if not experiment_name:
        return
    
    # 找到恢复点
    processed_count = find_resume_point(experiment_name)
    
    print(f"\n🚀 从第{processed_count + 1}个视频开始恢复实验")
    print(f"📁 实验目录: {experiment_name}")
    
    # 创建实验实例（重用现有目录）
    experiment = DADA100AblationExperiment(experiment_name=experiment_name)
    
    # 验证API配置
    if not experiment.openai_api_key or not experiment.vision_endpoint:
        print("⚠️  API配置未完成")
        print("需要设置: AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME")
        return
    
    total_videos = len(experiment.video_files)
    remaining_videos = total_videos - processed_count
    
    print(f"📊 恢复统计:")
    print(f"   总视频数: {total_videos}")
    print(f"   已处理: {processed_count}")
    print(f"   待处理: {remaining_videos}")
    print(f"⏱️  预计剩余时间: ~{remaining_videos * 0.42:.1f}分钟")
    
    if remaining_videos == 0:
        print("🎉 实验已完成，无需恢复!")
        return
    
    try:
        # 从中断点继续
        print(f"\n✅ 开始恢复实验，从第{processed_count + 1}个视频继续...")
        results = experiment.run_dada100_ablation_experiment(start_from=processed_count)
        
        print("\n🎉 DADA-100消融实验恢复完成！")
        print(f"📊 最终统计:")
        print(f"   成功: {results['successful_analyses']}/{total_videos}")
        print(f"   失败: {results['failed_analyses']}/{total_videos}")
        print(f"📁 结果位置: {experiment.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")
        print("💾 已处理的结果已保存在实验目录中")
    except Exception as e:
        print(f"\n❌ 恢复过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("💾 已处理的结果已保存在实验目录中")

if __name__ == "__main__":
    resume_experiment()