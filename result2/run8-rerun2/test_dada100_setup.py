#!/usr/bin/env python3
"""
测试DADA-100消融实验设置
"""

import os
import json
import datetime
from dotenv import load_dotenv
import pandas as pd
import glob

# 加载环境变量
load_dotenv()

def test_dada100_setup():
    """测试DADA-100消融实验配置"""
    print("🧪 测试DADA-100 Few-shot消融实验配置")
    
    # 检查few-shot图像
    fewshot_image_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
    expected_images = [
        "ghost_probing_sample1_before.jpg",
        "ghost_probing_sample1_during.jpg", 
        "ghost_probing_sample1_after.jpg",
        "ghost_probing_sample2_before.jpg",
        "ghost_probing_sample2_during.jpg",
        "ghost_probing_sample2_after.jpg",
        "ghost_probing_sample3_before.jpg",
        "ghost_probing_sample3_during.jpg",
        "ghost_probing_sample3_after.jpg"
    ]
    
    print(f"\n📷 检查Few-shot图像目录: {fewshot_image_dir}")
    fewshot_images = []
    
    for i, img_name in enumerate(expected_images):
        img_path = os.path.join(fewshot_image_dir, img_name)
        if os.path.exists(img_path):
            file_size = os.path.getsize(img_path)
            print(f"✅ {i+1}/9: {img_name} ({file_size:,} bytes)")
            fewshot_images.append(img_path)
        else:
            print(f"❌ {i+1}/9: {img_name} - 文件不存在")
    
    print(f"\n📊 Few-shot图像状态: {len(fewshot_images)}/9 张图像可用")
    
    # 检查DADA-100视频目录
    video_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos"
    print(f"\n🎬 检查DADA-100视频目录: {video_dir}")
    
    if os.path.exists(video_dir):
        video_files = glob.glob(os.path.join(video_dir, "*.avi"))
        video_files = sorted(video_files)
        print(f"✅ 找到 {len(video_files)} 个.avi视频文件")
        
        # 显示前5个视频和最后5个视频
        print("   前5个视频:")
        for i, video_file in enumerate(video_files[:5]):
            video_name = os.path.basename(video_file)
            file_size = os.path.getsize(video_file)
            print(f"   📹 {i+1}: {video_name} ({file_size:,} bytes)")
        
        if len(video_files) > 10:
            print("   ...")
            print("   后5个视频:")
            for i, video_file in enumerate(video_files[-5:]):
                video_name = os.path.basename(video_file)
                file_size = os.path.getsize(video_file)
                print(f"   📹 {len(video_files)-4+i}: {video_name} ({file_size:,} bytes)")
    else:
        print(f"❌ DADA-100视频目录不存在: {video_dir}")
        return False
    
    # 检查DADA-100 ground truth文件
    gt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
    print(f"\n📋 检查DADA-100 Ground Truth标签: {gt_path}")
    
    if os.path.exists(gt_path):
        try:
            gt_data = pd.read_csv(gt_path, sep=',')
            print(f"✅ 成功加载 {len(gt_data)} 条标签记录")
            print(f"   列名: {list(gt_data.columns)}")
            
            # 分析标签分布
            if 'ground_truth_label' in gt_data.columns:
                ghost_probing_count = 0
                none_count = 0
                
                for label in gt_data['ground_truth_label']:
                    if 'ghost probing' in str(label):
                        ghost_probing_count += 1
                    elif str(label) == 'none':
                        none_count += 1
                        
                print(f"   标签分布:")
                print(f"     Ghost Probing: {ghost_probing_count} 个")
                print(f"     None: {none_count} 个")
                print(f"     其他: {len(gt_data) - ghost_probing_count - none_count} 个")
                
                # 显示几个ghost probing的例子
                ghost_examples = gt_data[gt_data['ground_truth_label'].str.contains('ghost probing', na=False)].head(3)
                if not ghost_examples.empty:
                    print(f"   Ghost Probing示例:")
                    for _, row in ghost_examples.iterrows():
                        print(f"     {row['video_id']} -> {row['ground_truth_label']}")
                
        except Exception as e:
            print(f"❌ 标签文件读取失败: {str(e)}")
            return False
    else:
        print(f"❌ 标签文件不存在")
        return False
    
    # 检查环境变量
    print(f"\n🔑 检查API配置:")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    vision_endpoint = os.environ.get("VISION_ENDPOINT", "")
    vision_deployment = os.environ.get("VISION_DEPLOYMENT_NAME", "")
    
    if api_key:
        print(f"✅ OPENAI_API_KEY: 已设置 (长度: {len(api_key)})")
    else:
        print(f"❌ OPENAI_API_KEY: 未设置")
    
    if vision_endpoint:
        print(f"✅ VISION_ENDPOINT: {vision_endpoint}")
    else:
        print(f"❌ VISION_ENDPOINT: 未设置")
    
    if vision_deployment:
        print(f"✅ VISION_DEPLOYMENT_NAME: {vision_deployment}")
    else:
        print(f"❌ VISION_DEPLOYMENT_NAME: 未设置")
    
    # 生成实验配置预览
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_preview = {
        "experiment_type": "DADA-100 Few-shot Ablation",
        "timestamp": timestamp,
        "comparison_baseline": "Run8-Rerun (text few-shot only)",
        "enhanced_version": "Run8-Rerun + 9 Image Few-shot",
        "dataset": {
            "name": "DADA-100-videos",
            "total_videos": len(video_files) if os.path.exists(video_dir) else 0,
            "ghost_probing_videos": ghost_probing_count if 'gt_data' in locals() else "unknown",
            "none_videos": none_count if 'gt_data' in locals() else "unknown"
        },
        "few_shot_config": {
            "available_images": len(fewshot_images),
            "expected_images": 9,
            "sequences": 3,
            "pattern": "before-during-after",
            "source_videos": ["images_1_003.avi", "images_1_006.avi", "images_1_008.avi"]
        },
        "ground_truth": {
            "file": gt_path,
            "status": "available" if os.path.exists(gt_path) else "missing"
        },
        "api_config": {
            "api_key_set": bool(api_key),
            "endpoint_set": bool(vision_endpoint),
            "deployment_set": bool(vision_deployment)
        }
    }
    
    print(f"\n📄 DADA-100消融实验配置预览:")
    print(json.dumps(config_preview, indent=2, ensure_ascii=False))
    
    # 评估就绪状态
    ready_checks = [
        len(fewshot_images) == 9,
        os.path.exists(video_dir) and len(video_files) > 0,
        os.path.exists(gt_path),
        bool(api_key),
        bool(vision_endpoint),
        bool(vision_deployment)
    ]
    
    ready_status = all(ready_checks)
    
    print(f"\n🚀 DADA-100消融实验就绪状态: {'✅ 就绪' if ready_status else '❌ 未就绪'}")
    
    if ready_status:
        print("\n🎯 实验设计:")
        print("   对照组: Run8-Rerun (仅文本few-shot)")
        print("   实验组: Run8-Rerun + 9张图像few-shot") 
        print("   数据集: DADA-100-videos (101个标注视频)")
        print("   评估指标: Ghost Probing检测的Precision, Recall, F1-score")
        print("\n✅ 可以开始运行消融实验!")
    else:
        print("\n需要解决的问题:")
        if len(fewshot_images) != 9:
            print(f"   - Few-shot图像: 需要9张，当前{len(fewshot_images)}张")
        if not os.path.exists(video_dir) or len(video_files) == 0:
            print(f"   - DADA-100视频: 目录不存在或无视频文件")
        if not os.path.exists(gt_path):
            print(f"   - Ground truth: 标签文件不存在")
        if not api_key:
            print(f"   - API配置: OPENAI_API_KEY未设置")
        if not vision_endpoint:
            print(f"   - API配置: VISION_ENDPOINT未设置")
        if not vision_deployment:
            print(f"   - API配置: VISION_DEPLOYMENT_NAME未设置")
    
    return ready_status

if __name__ == "__main__":
    test_dada100_setup()