#!/usr/bin/env python3
"""
Azure ML Job Monitor Script
持续监控LLaVA鬼探头检测作业进展
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/monitor_job.py
"""

import time
import sys
import logging
from datetime import datetime
import subprocess

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobMonitor:
    """作业监控器"""
    
    def __init__(self, job_name: str, check_interval: int = 300):
        """
        初始化监控器
        
        Args:
            job_name: 作业名称
            check_interval: 检查间隔（秒）
        """
        self.job_name = job_name
        self.check_interval = check_interval
        self.start_time = datetime.now()
        
    def check_job_status(self):
        """检查作业状态"""
        try:
            cmd = [
                'python', 'submit_azure_llava_job.py', 
                '--action', 'status', 
                '--job-name', self.job_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout
                # 从输出中提取状态
                for line in output.split('\n'):
                    if '状态:' in line:
                        status = line.split('状态:')[1].strip()
                        return status
                return "Unknown"
            else:
                logger.error(f"状态检查失败: {result.stderr}")
                return "Error"
                
        except Exception as e:
            logger.error(f"状态检查异常: {e}")
            return "Error"
    
    def monitor(self):
        """开始监控"""
        logger.info(f"🔍 开始监控作业: {self.job_name}")
        logger.info(f"⏱️ 检查间隔: {self.check_interval}秒")
        
        previous_status = None
        
        while True:
            try:
                current_status = self.check_job_status()
                current_time = datetime.now()
                elapsed = current_time - self.start_time
                
                # 状态变化时记录
                if current_status != previous_status:
                    logger.info(f"📊 状态变化: {previous_status} → {current_status}")
                    logger.info(f"⏰ 运行时间: {elapsed}")
                    previous_status = current_status
                
                # 检查是否完成
                if current_status in ["Completed", "Failed", "Canceled"]:
                    logger.info(f"🏁 作业已结束: {current_status}")
                    
                    if current_status == "Completed":
                        logger.info("✅ 作业成功完成！")
                        self.handle_completion()
                    else:
                        logger.error(f"❌ 作业失败: {current_status}")
                    
                    break
                
                # 定期状态报告
                elif current_status == "Running":
                    logger.info(f"🔄 作业运行中... (已运行 {elapsed})")
                
                # 等待下次检查
                logger.info(f"⏳ 等待 {self.check_interval} 秒后再次检查...")
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("⚠️ 用户中断监控")
                break
            except Exception as e:
                logger.error(f"❌ 监控过程中出错: {e}")
                time.sleep(60)  # 出错时等待1分钟再重试
    
    def handle_completion(self):
        """处理作业完成"""
        logger.info("🎉 作业完成，开始后续处理...")
        
        # 1. 下载结果
        logger.info("📥 下载作业结果...")
        try:
            cmd = [
                'python', 'submit_azure_llava_job.py',
                '--action', 'download',
                '--job-name', self.job_name,
                '--download-path', './llava_results'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ 结果下载成功")
            else:
                logger.error(f"❌ 结果下载失败: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ 下载过程中出错: {e}")
        
        # 2. 显示完成总结
        total_time = datetime.now() - self.start_time
        logger.info("=" * 60)
        logger.info("🎯 LLaVA鬼探头检测作业完成总结")
        logger.info("=" * 60)
        logger.info(f"📋 作业ID: {self.job_name}")
        logger.info(f"⏱️ 总运行时间: {total_time}")
        logger.info(f"📁 结果位置: ./llava_results/")
        logger.info("=" * 60)
        
        # 3. 提供下一步建议
        logger.info("💡 后续操作建议:")
        logger.info("1. 查看结果文件: ls -la ./llava_results/")
        logger.info("2. 运行评估脚本:")
        logger.info("   python llava_ghost_probing_evaluation.py --llava-results ./llava_results/llava_ghost_probing_final_*.json")
        logger.info("3. 生成对比报告")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python monitor_job.py <job_name>")
        print("示例: python monitor_job.py cool_bucket_d45w5vfx73")
        sys.exit(1)
    
    job_name = sys.argv[1]
    
    print("🚀 LLaVA作业监控器启动")
    print(f"📋 监控作业: {job_name}")
    print("💡 使用 Ctrl+C 停止监控")
    print("-" * 50)
    
    monitor = JobMonitor(job_name, check_interval=300)  # 5分钟检查一次
    monitor.monitor()

if __name__ == "__main__":
    main()