#!/usr/bin/env python3
"""
批处理脚本：使用平衡版GPT-4.1 prompt处理所有100个Ground Truth视频
"""

import os
import subprocess
import pandas as pd
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_ground_truth_videos():
    """加载Ground Truth视频列表"""
    labels_file = "result/groundtruth_labels.csv"
    if os.path.exists(labels_file):
        df = pd.read_csv(labels_file, sep='\t')
        # 提取视频ID，去掉.avi后缀
        video_ids = [vid.replace('.avi', '') for vid in df['video_id'].tolist()]
        return video_ids
    else:
        print("❌ 找不到Ground Truth标签文件")
        return []

def process_single_video(video_id, output_dir="result/gpt41-balanced-full"):
    """处理单个视频"""
    video_file = f"{video_id}.avi"
    
    # 检查是否已处理
    result_file = f"{output_dir}/actionSummary_{video_id}.json"
    if os.path.exists(result_file):
        return {"video_id": video_id, "status": "already_processed", "message": "跳过已处理"}
    
    cmd = [
        "python", "ActionSummary-gpt41-balanced-prompt.py",
        "--single", f"DADA-2000-videos/{video_file}",
        "--output-dir", output_dir,
        "--interval", "10",
        "--frames", "10"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return {"video_id": video_id, "status": "success", "message": "处理成功"}
        else:
            return {"video_id": video_id, "status": "failed", "message": f"处理失败: {result.stderr[:200]}"}
    except subprocess.TimeoutExpired:
        return {"video_id": video_id, "status": "timeout", "message": "处理超时"}
    except Exception as e:
        return {"video_id": video_id, "status": "error", "message": f"处理异常: {str(e)}"}

def main():
    print("🔧 开始批处理平衡版GPT-4.1 prompt")
    print("=" * 60)
    
    # 加载Ground Truth视频列表
    video_ids = load_ground_truth_videos()
    print(f"📊 总共需要处理 {len(video_ids)} 个视频")
    
    if not video_ids:
        print("❌ 没有找到要处理的视频")
        return
    
    # 确保输出目录存在
    output_dir = "result/gpt41-balanced-full"
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查已处理的视频
    processed_count = 0
    if os.path.exists(output_dir):
        processed_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        processed_count = len(processed_files)
    
    print(f"📋 已处理视频数量: {processed_count}")
    remaining_videos = [vid for vid in video_ids 
                       if not os.path.exists(f"{output_dir}/actionSummary_{vid}.json")]
    print(f"📋 剩余待处理: {len(remaining_videos)} 个视频")
    
    if not remaining_videos:
        print("✅ 所有视频已处理完成！")
        return
    
    # 统计变量
    results = {
        "success": 0,
        "failed": 0,
        "timeout": 0,
        "error": 0,
        "already_processed": processed_count
    }
    
    failed_videos = []
    
    # 使用ThreadPoolExecutor进行并发处理
    max_workers = 2  # 降低并发数以避免API限制
    
    print(f"\n🚀 开始处理 {len(remaining_videos)} 个剩余视频...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_video = {executor.submit(process_single_video, video_id, output_dir): video_id 
                          for video_id in remaining_videos}
        
        # 处理完成的任务
        with tqdm(total=len(remaining_videos), desc="处理视频") as pbar:
            for future in as_completed(future_to_video):
                video_id = future_to_video[future]
                try:
                    result = future.result()
                    status = result["status"]
                    message = result["message"]
                    
                    results[status] += 1
                    
                    if status == "success":
                        tqdm.write(f"✅ {video_id}: {message}")
                    elif status == "already_processed":
                        tqdm.write(f"⏭️  {video_id}: {message}")
                    else:
                        tqdm.write(f"❌ {video_id}: {message}")
                        failed_videos.append(video_id)
                        
                except Exception as exc:
                    tqdm.write(f"❌ {video_id}: 处理异常 - {str(exc)}")
                    results["error"] += 1
                    failed_videos.append(video_id)
                
                pbar.update(1)
                
                # 添加短暂延迟避免API限制
                time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("📊 批处理完成统计")
    print("=" * 60)
    print(f"✅ 新增成功处理: {results['success']} 个视频")
    print(f"⏭️  之前已处理: {results['already_processed']} 个视频")
    print(f"❌ 处理失败: {results['failed']} 个视频")
    print(f"⏰ 处理超时: {results['timeout']} 个视频")
    print(f"🔥 处理异常: {results['error']} 个视频")
    
    total_processed = results['success'] + results['already_processed']
    print(f"\n📈 总处理率: {total_processed}/{len(video_ids)} ({total_processed/len(video_ids)*100:.1f}%)")
    
    if failed_videos:
        print(f"\n🔄 失败的视频 ({len(failed_videos)}个):")
        for video in failed_videos[:10]:  # 只显示前10个
            print(f"   • {video}")
        if len(failed_videos) > 10:
            print(f"   ... 还有 {len(failed_videos) - 10} 个")
    
    print(f"\n📂 结果保存在: {output_dir}")
    
    # 如果全部处理完成，提示可以进行评估
    if total_processed >= len(video_ids) * 0.9:  # 90%以上完成
        print(f"\n🎉 处理完成度达到 {total_processed/len(video_ids)*100:.1f}%")
        print("📊 可以开始进行平衡版性能评估了!")

if __name__ == "__main__":
    main()