#!/usr/bin/env python3
"""
完成剩余的消融实验
确保每个实验都处理满100个视频
"""

import subprocess
import sys
import time
import json
import os
import threading
import logging
import datetime

def setup_logging():
    """设置日志"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/complete_remaining_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_completion_status():
    """检查各实验的完成状态"""
    experiments = {
        '1-sample': '/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/1-sample/ablation_1sample_results_20250731_134158.json',
        '2-samples': '/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples/ablation_2samples_results_20250731_133056.json',
        '5-samples': '/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/5-samples/ablation_5samples_results_20250731_134159.json'
    }
    
    status = {}
    for name, path in experiments.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            completed = len(data.get('detailed_results', []))
            status[name] = {
                'completed': completed,
                'remaining': 100 - completed,
                'path': path
            }
        else:
            status[name] = {
                'completed': 0,
                'remaining': 100,
                'path': path
            }
    
    return status

def run_experiment_from_checkpoint(exp_name, script_path, completed_count):
    """从检查点继续运行实验"""
    logger = logging.getLogger(__name__)
    remaining = 100 - completed_count
    
    if remaining <= 0:
        logger.info(f"✅ {exp_name} 已完成100个视频")
        return True
    
    logger.info(f"🔄 {exp_name} 需要完成剩余 {remaining} 个视频")
    
    try:
        # 修改脚本以支持从检查点继续
        cmd = [sys.executable, script_path, "--limit", "100"]
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {exp_name} 完成 (耗时: {duration/60:.1f}分钟)")
            return True
        else:
            logger.error(f"❌ {exp_name} 失败")
            logger.error(f"错误输出: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {exp_name} 超时")
        return False
    except Exception as e:
        logger.error(f"💥 {exp_name} 异常: {str(e)}")
        return False

def run_single_experiment_thread(exp_name, script_path, completed_count):
    """在线程中运行单个实验"""
    logger = logging.getLogger(__name__)
    success = run_experiment_from_checkpoint(exp_name, script_path, completed_count)
    if success:
        logger.info(f"🎉 {exp_name} 线程完成")
    else:
        logger.error(f"❌ {exp_name} 线程失败")
    return success

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("🎯 开始完成剩余消融实验")
    
    # 检查当前状态
    status = check_completion_status()
    
    print("📊 当前完成状态:")
    for name, info in status.items():
        print(f"  {name}: {info['completed']}/100 (剩余: {info['remaining']})")
    
    # 定义需要完成的实验
    experiments_to_run = []
    
    scripts = {
        '1-sample': '/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/1-sample/run8_ablation_1sample.py',
        '2-samples': '/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples/run8_ablation_2samples.py',
        '5-samples': '/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/5-samples/run8_ablation_5samples.py'
    }
    
    for name, info in status.items():
        if info['remaining'] > 0:
            experiments_to_run.append((name, scripts[name], info['completed']))
    
    if not experiments_to_run:
        print("🎉 所有实验已完成!")
        return
    
    print(f"🚀 将并行运行 {len(experiments_to_run)} 个实验:")
    for name, _, completed in experiments_to_run:
        remaining = 100 - completed
        print(f"  - {name}: 剩余 {remaining} 个视频")
    
    # 并行运行所有需要完成的实验
    threads = []
    for name, script_path, completed_count in experiments_to_run:
        thread = threading.Thread(
            target=run_single_experiment_thread,
            args=(name, script_path, completed_count)
        )
        threads.append(thread)
        thread.start()
        logger.info(f"🧵 {name} 线程已启动")
        
        # 间隔5秒启动下一个线程
        if len(threads) < len(experiments_to_run):
            time.sleep(5)
    
    # 等待所有线程完成
    logger.info("⏳ 等待所有实验完成...")
    for i, thread in enumerate(threads):
        thread.join()
        exp_name = experiments_to_run[i][0]
        logger.info(f"✅ {exp_name} 线程已结束")
    
    # 最终检查
    final_status = check_completion_status()
    print("\n📊 最终完成状态:")
    all_complete = True
    for name, info in final_status.items():
        status_icon = "✅" if info['completed'] >= 100 else "⚠️"
        print(f"  {status_icon} {name}: {info['completed']}/100")
        if info['completed'] < 100:
            all_complete = False
    
    if all_complete:
        print("\n🎉 所有消融实验已完成!")
    else:
        print("\n⚠️ 部分实验未完成，请检查日志")
    
    logger.info("消融实验补完任务结束")

if __name__ == "__main__":
    main()