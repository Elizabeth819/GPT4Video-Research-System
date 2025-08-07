#!/usr/bin/env python3
"""
LLaVA Ghost Probing Single Video Test Script
测试LLaVA鬼探头检测系统的单个视频处理功能
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/test_single_video.py
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.append('/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA')
from llava_ghost_probing_detector import LLaVAGhostProbingDetector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_video(video_path: str, output_file: str = None):
    """
    测试单个视频的鬼探头检测
    
    Args:
        video_path: 视频文件路径
        output_file: 输出文件路径（可选）
    """
    try:
        logger.info("🚀 开始LLaVA鬼探头检测单视频测试")
        logger.info(f"📹 测试视频: {video_path}")
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"❌ 视频文件不存在: {video_path}")
            return False
        
        # 初始化检测器
        logger.info("🔧 正在初始化LLaVA检测器...")
        detector = LLaVAGhostProbingDetector()
        
        # 分析视频
        video_id = Path(video_path).stem
        logger.info(f"🎬 开始分析视频: {video_id}")
        
        start_time = datetime.now()
        result = detector.analyze_video(video_path, video_id)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if result:
            # 提取鬼探头标签
            ghost_label, confidence = detector.extract_ghost_probing_label(result)
            
            # 构建完整结果
            test_result = {
                'test_info': {
                    'video_path': video_path,
                    'video_id': video_id,
                    'test_timestamp': start_time.isoformat(),
                    'processing_time_seconds': round(processing_time, 2)
                },
                'detection_result': {
                    'ghost_probing_label': ghost_label,
                    'confidence': confidence
                },
                'llava_analysis': result,
                'system_info': {
                    'model': 'LLaVA-Video-7B-Qwen2',
                    'framework': 'LLaVA-NeXT',
                    'prompt_type': 'balanced_gpt41_compatible'
                }
            }
            
            # 打印结果
            print("\n" + "="*60)
            print("🎯 LLaVA鬼探头检测测试结果")
            print("="*60)
            print(f"📹 视频: {video_id}")
            print(f"⏱️  处理时间: {processing_time:.2f}秒")
            print(f"🏷️  检测结果: {ghost_label}")
            print(f"📊 置信度: {confidence}")
            print("-"*40)
            print("📝 详细分析:")
            print(f"  场景描述: {result.get('summary', 'N/A')}")
            print(f"  关键动作: {result.get('key_actions', 'N/A')}")
            print(f"  关键对象: {result.get('key_objects', 'N/A')}")
            print(f"  情感倾向: {result.get('sentiment', 'N/A')}")
            print(f"  场景主题: {result.get('scene_theme', 'N/A')}")
            
            if 'next_action' in result:
                next_action = result['next_action']
                print(f"  下一步动作:")
                print(f"    速度控制: {next_action.get('speed_control', 'N/A')}")
                print(f"    方向控制: {next_action.get('direction_control', 'N/A')}")
                print(f"    车道控制: {next_action.get('lane_control', 'N/A')}")
            print("="*60)
            
            # 保存结果到文件
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(test_result, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 测试结果已保存到: {output_file}")
            
            logger.info("✅ 单视频测试成功完成")
            return True
            
        else:
            logger.error("❌ 视频分析失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载功能"""
    try:
        logger.info("🔍 测试LLaVA模型加载...")
        detector = LLaVAGhostProbingDetector()
        logger.info("✅ 模型加载测试成功")
        return True
    except Exception as e:
        logger.error(f"❌ 模型加载测试失败: {e}")
        return False

def run_comprehensive_test():
    """运行综合测试"""
    logger.info("🧪 开始LLaVA鬼探头检测综合测试")
    
    # 1. 测试模型加载
    if not test_model_loading():
        return False
    
    # 2. 寻找测试视频
    test_video_paths = [
        "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_001.avi",
        "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_002.avi",
        "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/images_1_003.avi"
    ]
    
    success_count = 0
    total_count = 0
    
    for video_path in test_video_paths:
        if os.path.exists(video_path):
            total_count += 1
            logger.info(f"\n🎬 测试视频 {total_count}: {Path(video_path).name}")
            
            output_file = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/test_result_{Path(video_path).stem}.json"
            
            if test_single_video(video_path, output_file):
                success_count += 1
            
            logger.info("-" * 40)
    
    # 测试总结
    print("\n" + "="*60)
    print("🏁 LLaVA鬼探头检测综合测试完成")
    print("="*60)
    print(f"📊 总计测试: {total_count} 个视频")
    print(f"✅ 成功: {success_count} 个")
    print(f"❌ 失败: {total_count - success_count} 个")
    print(f"📈 成功率: {success_count/total_count*100:.1f}%" if total_count > 0 else "无测试视频")
    print("="*60)
    
    return success_count == total_count

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLaVA鬼探头检测单视频测试')
    parser.add_argument('--video', type=str,
                       help='指定测试视频路径')
    parser.add_argument('--output', type=str,
                       help='结果输出文件路径')
    parser.add_argument('--comprehensive', action='store_true',
                       help='运行综合测试')
    parser.add_argument('--model-test-only', action='store_true',
                       help='仅测试模型加载')
    
    args = parser.parse_args()
    
    if args.model_test_only:
        # 仅测试模型加载
        success = test_model_loading()
        sys.exit(0 if success else 1)
    
    elif args.comprehensive:
        # 运行综合测试
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    
    elif args.video:
        # 测试指定视频
        success = test_single_video(args.video, args.output)
        sys.exit(0 if success else 1)
    
    else:
        # 默认运行综合测试
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()