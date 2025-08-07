#!/usr/bin/env python3
"""
Azure ML Job Monitor
实时监控Azure ML作业状态
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/monitor_job_simple.py
"""

import os
import sys
import time
import logging
from datetime import datetime

try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
except ImportError:
    print("❌ Azure ML SDK未安装，请运行: pip install azure-ai-ml")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureMLJobMonitor:
    """Azure ML作业监控器"""
    
    def __init__(self):
        """初始化监控器"""
        try:
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id="0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
                resource_group_name="llava-resourcegroup",
                workspace_name="llava-workspace"
            )
            logger.info("✅ Azure ML客户端初始化成功")
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> dict:
        """获取作业状态"""
        try:
            job = self.ml_client.jobs.get(job_name)
            return {
                "name": job.name,
                "status": job.status,
                "created_at": job.creation_context.created_at,
                "start_time": getattr(job, 'start_time', None),
                "end_time": getattr(job, 'end_time', None),
                "studio_url": job.studio_url,
                "compute": getattr(job, 'compute', None)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def monitor_job(self, job_name: str, check_interval: int = 30, max_duration: int = 14400):
        """
        监控作业直到完成
        
        Args:
            job_name: 作业名称
            check_interval: 检查间隔（秒）
            max_duration: 最大监控时间（秒，默认4小时）
        """
        start_time = time.time()
        last_status = None
        
        print(f"🔄 开始监控作业: {job_name}")
        print(f"⏱️ 检查间隔: {check_interval}秒")
        print(f"⏰ 最大监控时间: {max_duration/3600:.1f}小时")
        print("="*80)
        
        try:
            while time.time() - start_time < max_duration:
                # 获取作业状态
                status_info = self.get_job_status(job_name)
                
                if "error" in status_info:
                    print(f"❌ 获取状态失败: {status_info['error']}")
                    time.sleep(check_interval)
                    continue
                
                current_status = status_info["status"]
                current_time = datetime.now().strftime("%H:%M:%S")
                
                # 如果状态发生变化，显示详细信息
                if current_status != last_status:
                    print(f"\n[{current_time}] 📊 状态变化: {last_status} → {current_status}")
                    
                    if current_status == "Running":
                        print(f"🚀 作业开始运行!")
                        print(f"🔗 监控链接: {status_info['studio_url']}")
                        
                    elif current_status == "Completed":
                        print(f"✅ 作业完成!")
                        elapsed = time.time() - start_time
                        print(f"⏱️ 总耗时: {elapsed/60:.1f}分钟")
                        print(f"🔗 结果链接: {status_info['studio_url']}")
                        print("\n🎉 监控完成！")
                        return "Completed"
                        
                    elif current_status == "Failed":
                        print(f"❌ 作业失败!")
                        print(f"🔗 错误详情: {status_info['studio_url']}")
                        print("\n💡 建议运行故障分析:")
                        print(f"python investigate_failed_job.py --job-name {job_name}")
                        return "Failed"
                        
                    elif current_status == "Canceled":
                        print(f"⏹️ 作业已取消")
                        return "Canceled"
                    
                    last_status = current_status
                else:
                    # 状态未变化，显示简化信息
                    elapsed = time.time() - start_time
                    print(f"[{current_time}] {current_status} - 已运行 {elapsed/60:.1f}分钟")
                
                # 等待下次检查
                time.sleep(check_interval)
            
            # 超时
            print(f"\n⏰ 监控超时 ({max_duration/3600:.1f}小时)")
            final_status = self.get_job_status(job_name)
            print(f"📊 最终状态: {final_status.get('status', 'Unknown')}")
            return "Timeout"
            
        except KeyboardInterrupt:
            print(f"\n⚡ 监控被用户中断")
            final_status = self.get_job_status(job_name)
            print(f"📊 当前状态: {final_status.get('status', 'Unknown')}")
            return "Interrupted"
        except Exception as e:
            print(f"\n❌ 监控过程中出错: {e}")
            return "Error"
    
    def show_job_summary(self, job_name: str):
        """显示作业摘要信息"""
        status_info = self.get_job_status(job_name)
        
        if "error" in status_info:
            print(f"❌ 无法获取作业信息: {status_info['error']}")
            return
        
        print(f"\n📋 作业摘要: {job_name}")
        print("="*60)
        print(f"状态: {status_info['status']}")
        print(f"创建时间: {status_info['created_at']}")
        print(f"计算集群: {status_info.get('compute', 'Unknown')}")
        print(f"Studio链接: {status_info['studio_url']}")
        print("="*60)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure ML作业实时监控器')
    parser.add_argument('--job-name', type=str, required=True,
                       help='要监控的作业名称')
    parser.add_argument('--interval', type=int, default=30,
                       help='检查间隔（秒）')
    parser.add_argument('--max-hours', type=float, default=4.0,
                       help='最大监控时间（小时）')
    parser.add_argument('--summary-only', action='store_true',
                       help='只显示作业摘要，不进行持续监控')
    
    args = parser.parse_args()
    
    try:
        monitor = AzureMLJobMonitor()
        
        if args.summary_only:
            # 只显示摘要
            monitor.show_job_summary(args.job_name)
        else:
            # 持续监控
            result = monitor.monitor_job(
                args.job_name, 
                args.interval, 
                int(args.max_hours * 3600)
            )
            
            print(f"\n🏁 监控结束，最终结果: {result}")
            
    except Exception as e:
        logger.error(f"❌ 监控失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()