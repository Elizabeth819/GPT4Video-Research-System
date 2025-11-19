#!/usr/bin/env python3
"""
Quick Launch Script for Video-LLaMA2 Ghost Probing Detection
快速启动Video-LLaMA2鬼探头检测的便捷脚本
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置Azure ML环境变量
os.environ["AZURE_SUBSCRIPTION_ID"] = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
os.environ["AZURE_RESOURCE_GROUP"] = "video-llama2-ghost-probing-rg"
os.environ["AZURE_WORKSPACE_NAME"] = "video-llama2-ghost-probing-ws"

def show_menu():
    """显示操作菜单"""
    print("=" * 60)
    print("🎬 Video-LLaMA2 Ghost Probing Detection")
    print("=" * 60)
    print(f"Azure 订阅: {os.environ['AZURE_SUBSCRIPTION_ID']}")
    print(f"资源组: {os.environ['AZURE_RESOURCE_GROUP']}")
    print(f"工作区: {os.environ['AZURE_WORKSPACE_NAME']}")
    print("=" * 60)
    print("选择操作:")
    print("1. 🔍 检查环境 (本地测试)")
    print("2. 🚀 提交Azure ML作业")
    print("3. 👁️ 监控现有作业")
    print("4. 📥 下载作业结果")
    print("5. 🎯 本地单视频测试")
    print("6. 📋 查看作业历史")
    print("0. 退出")
    print("=" * 60)

def check_local_environment():
    """检查本地环境"""
    logger.info("🔍 检查本地环境...")
    
    # 检查必要文件
    required_files = [
        "video_llama2_ghost_probing_detector.py",
        "video_llama2_environment.yml",
        "submit_videollama2_ghost_probing_job.py",
        "eval_configs/video_llama_eval_withaudio.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✅ {file_path}")
        else:
            logger.error(f"❌ {file_path} 不存在")
            missing_files.append(file_path)
    
    # 检查视频数据
    video_folder = Path("../../DADA-2000-videos")
    if video_folder.exists():
        target_videos = []
        for i in range(1, 6):
            pattern = f"images_{i}_*.avi"
            videos = sorted(video_folder.glob(pattern))
            target_videos.extend(videos)
            if len(target_videos) >= 100:
                break
        
        target_videos = target_videos[:100]
        logger.info(f"✅ 找到 {len(target_videos)} 个目标视频")
        
        if len(target_videos) < 100:
            logger.warning(f"⚠️ 视频数量不足100个")
    else:
        logger.error("❌ 视频文件夹不存在")
        missing_files.append("DADA-2000-videos")
    
    # 检查ground truth
    gt_file = Path("../../result/groundtruth_labels.csv")
    if gt_file.exists():
        logger.info("✅ Ground truth文件存在")
    else:
        logger.error("❌ Ground truth文件不存在")
        missing_files.append("groundtruth_labels.csv")
    
    if missing_files:
        logger.error(f"❌ 缺少必要文件: {missing_files}")
        return False
    else:
        logger.info("🎉 本地环境检查通过!")
        return True

def submit_azure_ml_job():
    """提交Azure ML作业"""
    logger.info("🚀 提交Azure ML作业...")
    
    try:
        result = subprocess.run([
            sys.executable, "submit_videollama2_ghost_probing_job.py"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            logger.info("✅ 作业提交成功")
        else:
            logger.error("❌ 作业提交失败")
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"❌ 作业提交异常: {e}")
        return False

def monitor_job():
    """监控作业"""
    job_name = input("请输入要监控的作业名称: ").strip()
    if not job_name:
        logger.error("❌ 作业名称不能为空")
        return
    
    logger.info(f"👁️ 监控作业: {job_name}")
    
    try:
        subprocess.run([
            sys.executable, "submit_videollama2_ghost_probing_job.py",
            "--monitor-only", job_name
        ])
    except Exception as e:
        logger.error(f"❌ 监控失败: {e}")

def download_results():
    """下载作业结果"""
    job_name = input("请输入要下载结果的作业名称: ").strip()
    if not job_name:
        logger.error("❌ 作业名称不能为空")
        return
    
    logger.info(f"📥 下载作业结果: {job_name}")
    
    try:
        subprocess.run([
            sys.executable, "submit_videollama2_ghost_probing_job.py",
            "--download-only", job_name
        ])
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")

def test_single_video():
    """本地单视频测试"""
    logger.info("🎯 本地单视频测试...")
    
    # 查找示例视频
    video_folder = Path("../../DADA-2000-videos")
    if not video_folder.exists():
        logger.error("❌ 视频文件夹不存在")
        return
    
    # 找到第一个视频进行测试
    test_video = None
    for i in range(1, 6):
        pattern = f"images_{i}_*.avi"
        videos = sorted(video_folder.glob(pattern))
        if videos:
            test_video = videos[0]
            break
    
    if not test_video:
        logger.error("❌ 没有找到可测试的视频")
        return
    
    logger.info(f"🎬 测试视频: {test_video.name}")
    
    try:
        # 运行单视频测试
        result = subprocess.run([
            sys.executable, "video_llama2_ghost_probing_detector.py",
            "--single-video", str(test_video),
            "--config", "eval_configs/video_llama_eval_withaudio.yaml"
        ], capture_output=True, text=True)
        
        print("测试结果:")
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            logger.info("✅ 单视频测试成功")
        else:
            logger.error("❌ 单视频测试失败")
            
    except Exception as e:
        logger.error(f"❌ 单视频测试异常: {e}")

def show_job_history():
    """显示作业历史"""
    logger.info("📋 查看作业历史...")
    
    # 查找作业信息文件
    job_files = list(Path(".").glob("video_llama2_job_info_*.json"))
    
    if not job_files:
        logger.info("📄 没有找到作业历史记录")
        return
    
    print(f"找到 {len(job_files)} 个作业记录:")
    print("-" * 60)
    
    for i, job_file in enumerate(sorted(job_files, reverse=True)):
        try:
            import json
            with open(job_file, 'r') as f:
                job_info = json.load(f)
            
            print(f"{i+1}. {job_info.get('job_name', 'Unknown')}")
            print(f"   状态: {job_info.get('status', 'Unknown')}")
            print(f"   提交时间: {job_info.get('submission_time', 'Unknown')}")
            print(f"   Studio链接: {job_info.get('studio_url', 'Unknown')}")
            print("-" * 60)
            
        except Exception as e:
            logger.error(f"❌ 读取作业文件失败 {job_file}: {e}")

def main():
    """主函数"""
    while True:
        show_menu()
        
        try:
            choice = input("请选择操作 (0-6): ").strip()
            
            if choice == "0":
                logger.info("👋 退出程序")
                break
            elif choice == "1":
                check_local_environment()
            elif choice == "2":
                submit_azure_ml_job()
            elif choice == "3":
                monitor_job()
            elif choice == "4":
                download_results()
            elif choice == "5":
                test_single_video()
            elif choice == "6":
                show_job_history()
            else:
                logger.warning("⚠️ 无效选择，请重新输入")
            
            input("\n按回车键继续...")
            
        except KeyboardInterrupt:
            logger.info("\n👋 程序被用户中断")
            break
        except Exception as e:
            logger.error(f"❌ 程序异常: {e}")
            input("\n按回车键继续...")

if __name__ == "__main__":
    main()