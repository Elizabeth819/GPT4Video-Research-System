#!/usr/bin/env python3
"""
并行运行Few-shot消融实验
同时启动1, 2, 5个样本的实验以节省时间
"""

import subprocess
import os
import sys
import threading
import time
import logging
import datetime

def setup_logging():
    """设置日志"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/parallel_experiments_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_single_experiment(script_path, experiment_name, limit=100):
    """运行单个实验的线程函数"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 [{experiment_name}] 开始运行")
    
    try:
        cmd = [sys.executable, script_path, "--limit", str(limit)]
        start_time = time.time()
        
        # 运行实验并等待完成
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ [{experiment_name}] 完成 (耗时: {duration/60:.1f}分钟)")
            print(f"✅ {experiment_name} 实验完成！耗时: {duration/60:.1f}分钟")
        else:
            logger.error(f"❌ [{experiment_name}] 失败")
            logger.error(f"错误输出: {result.stderr}")
            print(f"❌ {experiment_name} 实验失败")
            
    except Exception as e:
        logger.error(f"💥 [{experiment_name}] 异常: {str(e)}")
        print(f"💥 {experiment_name} 异常: {str(e)}")

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("🎯 开始并行运行Few-shot消融实验")
    print("🎯 开始并行运行Few-shot消融实验")
    
    # 定义实验
    experiments = [
        {
            "name": "1-sample",
            "script": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/1-sample/run8_ablation_1sample.py"
        },
        {
            "name": "5-samples", 
            "script": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/5-samples/run8_ablation_5samples.py"
        }
    ]
    
    # 创建线程
    threads = []
    for exp in experiments:
        thread = threading.Thread(
            target=run_single_experiment,
            args=(exp["script"], exp["name"], 100)
        )
        threads.append(thread)
        thread.start()
        logger.info(f"🧵 {exp['name']} 线程已启动")
        print(f"🧵 {exp['name']} 实验已启动")
        
        # 线程间间隔5秒避免同时启动的资源冲突
        time.sleep(5)
    
    print(f"📊 所有实验已启动，等待完成...")
    print(f"📊 2-samples实验已在另一个进程中运行")
    logger.info("所有实验线程已启动，等待完成")
    
    # 等待所有线程完成
    for i, thread in enumerate(threads):
        thread.join()
        logger.info(f"线程 {experiments[i]['name']} 已完成")
    
    logger.info("🎉 所有并行实验完成")
    print("🎉 所有并行实验完成！")

if __name__ == "__main__":
    main()