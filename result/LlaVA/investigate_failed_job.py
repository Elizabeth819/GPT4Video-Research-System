#!/usr/bin/env python3
"""
Azure ML Failed Job Investigation Script
调查失败的Azure ML作业 'crimson_boniato_k1kg8q62fr'
文件路径: /Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/investigate_failed_job.py
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
except ImportError:
    print("❌ Azure ML SDK未安装，请运行: pip install azure-ai-ml")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureMLJobInvestigator:
    """Azure ML作业故障调查器"""
    
    def __init__(self, 
                 subscription_id: str = "0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
                 resource_group: str = "llava-resourcegroup", 
                 workspace_name: str = "llava-workspace"):
        """
        初始化Azure ML客户端
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        
        try:
            # 初始化ML客户端
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            
            logger.info(f"✅ Azure ML客户端初始化成功")
            logger.info(f"📋 订阅: {subscription_id}")
            logger.info(f"📋 资源组: {resource_group}")
            logger.info(f"📋 工作区: {workspace_name}")
            
        except Exception as e:
            logger.error(f"❌ Azure ML客户端初始化失败: {e}")
            raise
    
    def investigate_job(self, job_name: str) -> dict:
        """
        深度调查失败的作业
        
        Args:
            job_name: 作业名称
            
        Returns:
            调查结果字典
        """
        investigation_report = {
            "job_name": job_name,
            "investigation_time": datetime.now().isoformat(),
            "status": None,
            "error_details": [],
            "logs": [],
            "recommendations": []
        }
        
        try:
            logger.info(f"🔍 开始调查作业: {job_name}")
            logger.info("=" * 60)
            
            # 1. 获取作业基本信息
            job_info = self._get_job_basic_info(job_name)
            investigation_report.update(job_info)
            
            # 2. 获取详细错误信息
            error_details = self._get_job_error_details(job_name)
            investigation_report["error_details"] = error_details
            
            # 3. 获取作业日志
            logs = self._get_job_logs(job_name)
            investigation_report["logs"] = logs
            
            # 4. 分析失败原因
            failure_analysis = self._analyze_failure_causes(investigation_report)
            investigation_report["failure_analysis"] = failure_analysis
            
            # 5. 生成修复建议
            recommendations = self._generate_recommendations(investigation_report)
            investigation_report["recommendations"] = recommendations
            
            # 6. 保存调查报告
            self._save_investigation_report(investigation_report)
            
            return investigation_report
            
        except Exception as e:
            logger.error(f"❌ 调查作业失败: {e}")
            investigation_report["investigation_error"] = str(e)
            return investigation_report
    
    def _get_job_basic_info(self, job_name: str) -> dict:
        """获取作业基本信息"""
        logger.info("📊 获取作业基本信息...")
        
        try:
            job = self.ml_client.jobs.get(job_name)
            
            basic_info = {
                "status": job.status,
                "creation_time": str(job.creation_context.created_at) if job.creation_context else None,
                "start_time": str(job.start_time) if hasattr(job, 'start_time') and job.start_time else None,
                "end_time": str(job.end_time) if hasattr(job, 'end_time') and job.end_time else None,
                "compute": job.compute if hasattr(job, 'compute') else None,
                "environment": job.environment if hasattr(job, 'environment') else None,
                "command": job.command if hasattr(job, 'command') else None,
                "studio_url": job.studio_url if hasattr(job, 'studio_url') else None,
                "experiment_name": job.experiment_name if hasattr(job, 'experiment_name') else None
            }
            
            logger.info(f"📋 作业状态: {basic_info['status']}")
            logger.info(f"🕒 创建时间: {basic_info['creation_time']}")
            logger.info(f"🖥️  计算集群: {basic_info['compute']}")
            logger.info(f"🌍 环境: {basic_info['environment']}")
            logger.info(f"🔗 Studio链接: {basic_info['studio_url']}")
            
            return basic_info
            
        except ResourceNotFoundError:
            logger.error(f"❌ 作业不存在: {job_name}")
            return {"error": "Job not found"}
        except Exception as e:
            logger.error(f"❌ 获取作业基本信息失败: {e}")
            return {"error": str(e)}
    
    def _get_job_error_details(self, job_name: str) -> list:
        """获取作业详细错误信息"""
        logger.info("🔍 获取详细错误信息...")
        error_details = []
        
        try:
            job = self.ml_client.jobs.get(job_name)
            
            # 检查作业级别错误
            if hasattr(job, 'error') and job.error:
                error_details.append({
                    "type": "job_error",
                    "message": str(job.error.message) if hasattr(job.error, 'message') else str(job.error),
                    "code": str(job.error.code) if hasattr(job.error, 'code') else None,
                    "details": str(job.error.details) if hasattr(job.error, 'details') else None
                })
                logger.error(f"❌ 作业错误: {job.error}")
            
            # 检查状态相关错误
            if job.status in ["Failed", "Canceled"]:
                error_details.append({
                    "type": "status_error",
                    "status": job.status,
                    "message": f"Job ended with status: {job.status}"
                })
            
            # 检查计算相关错误
            if hasattr(job, 'compute') and job.compute:
                try:
                    compute = self.ml_client.compute.get(job.compute)
                    if compute.provisioning_state != "Succeeded":
                        error_details.append({
                            "type": "compute_error",
                            "compute_name": job.compute,
                            "provisioning_state": compute.provisioning_state,
                            "message": f"Compute cluster is in {compute.provisioning_state} state"
                        })
                        logger.warning(f"⚠️  计算集群状态异常: {compute.provisioning_state}")
                except Exception as compute_error:
                    error_details.append({
                        "type": "compute_check_error",
                        "message": f"Failed to check compute cluster: {compute_error}"
                    })
            
            if error_details:
                logger.info(f"🔍 发现 {len(error_details)} 个错误")
            else:
                logger.info("ℹ️  未发现明显错误信息")
            
            return error_details
            
        except Exception as e:
            logger.error(f"❌ 获取错误详情失败: {e}")
            return [{"type": "investigation_error", "message": str(e)}]
    
    def _get_job_logs(self, job_name: str) -> list:
        """获取作业日志"""
        logger.info("📝 尝试获取作业日志...")
        logs = []
        
        try:
            # 尝试下载作业日志
            download_path = f"./job_logs_{job_name}"
            Path(download_path).mkdir(parents=True, exist_ok=True)
            
            try:
                # 尝试下载日志文件
                self.ml_client.jobs.download(
                    name=job_name,
                    download_path=download_path
                )
                
                # 读取下载的日志文件
                log_files = []
                for root, dirs, files in os.walk(download_path):
                    for file in files:
                        if file.endswith(('.log', '.txt', '.out', '.err')):
                            log_files.append(os.path.join(root, file))
                
                for log_file in log_files:
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            logs.append({
                                "file": log_file,
                                "content": content[:5000],  # 限制前5000字符
                                "size": len(content)
                            })
                        logger.info(f"📄 读取日志文件: {log_file}")
                    except Exception as read_error:
                        logs.append({
                            "file": log_file,
                            "error": f"Failed to read: {read_error}"
                        })
                
            except Exception as download_error:
                logger.warning(f"⚠️  无法下载日志: {download_error}")
                logs.append({
                    "type": "download_error",
                    "message": str(download_error)
                })
            
            if logs:
                logger.info(f"📝 获取到 {len(logs)} 个日志文件")
            else:
                logger.info("ℹ️  未找到可用日志")
            
            return logs
            
        except Exception as e:
            logger.error(f"❌ 获取日志失败: {e}")
            return [{"type": "log_error", "message": str(e)}]
    
    def _analyze_failure_causes(self, investigation_report: dict) -> dict:
        """分析失败原因"""
        logger.info("🔬 分析失败原因...")
        
        analysis = {
            "primary_cause": "Unknown",
            "secondary_causes": [],
            "evidence": [],
            "confidence": "Low"
        }
        
        try:
            # 分析状态
            if investigation_report.get("status") == "Failed":
                analysis["evidence"].append("Job status is Failed")
            
            # 分析错误信息
            error_details = investigation_report.get("error_details", [])
            for error in error_details:
                if error.get("type") == "job_error":
                    message = error.get("message", "").lower()
                    
                    # 常见错误模式匹配
                    if "command" in message or "parsing" in message:
                        analysis["primary_cause"] = "Command Parsing Error"
                        analysis["confidence"] = "High"
                        analysis["evidence"].append(f"Command parsing error detected: {error.get('message')}")
                    
                    elif "environment" in message or "package" in message or "import" in message:
                        analysis["primary_cause"] = "Environment/Package Error"
                        analysis["confidence"] = "High"
                        analysis["evidence"].append(f"Environment error detected: {error.get('message')}")
                    
                    elif "compute" in message or "resource" in message:
                        analysis["primary_cause"] = "Compute Resource Error"
                        analysis["confidence"] = "Medium"
                        analysis["evidence"].append(f"Compute error detected: {error.get('message')}")
                    
                    elif "data" in message or "input" in message or "file" in message:
                        analysis["primary_cause"] = "Data Access Error"
                        analysis["confidence"] = "Medium"
                        analysis["evidence"].append(f"Data access error detected: {error.get('message')}")
                
                elif error.get("type") == "compute_error":
                    analysis["secondary_causes"].append("Compute cluster not ready")
                    analysis["evidence"].append(f"Compute state: {error.get('provisioning_state')}")
            
            # 分析日志内容
            logs = investigation_report.get("logs", [])
            for log in logs:
                content = log.get("content", "").lower()
                if "error" in content or "failed" in content:
                    if "modulenotfounderror" in content or "importerror" in content:
                        if analysis["primary_cause"] == "Unknown":
                            analysis["primary_cause"] = "Missing Python Package"
                            analysis["confidence"] = "High"
                    elif "cuda" in content or "gpu" in content:
                        analysis["secondary_causes"].append("GPU/CUDA related issue")
                    elif "permission" in content or "access" in content:
                        analysis["secondary_causes"].append("File/Permission access issue")
            
            # 分析命令
            command = investigation_report.get("command", "")
            if command and isinstance(command, str):
                if len(command.split('\n')) > 1:
                    analysis["secondary_causes"].append("Multi-line command may have parsing issues")
                if "&&" in command:
                    analysis["secondary_causes"].append("Chained commands may fail at any step")
            
            logger.info(f"🎯 主要原因: {analysis['primary_cause']}")
            logger.info(f"📊 置信度: {analysis['confidence']}")
            if analysis["secondary_causes"]:
                logger.info(f"🔍 次要原因: {', '.join(analysis['secondary_causes'])}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 分析失败原因出错: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, investigation_report: dict) -> list:
        """生成修复建议"""
        logger.info("💡 生成修复建议...")
        
        recommendations = []
        
        try:
            failure_analysis = investigation_report.get("failure_analysis", {})
            primary_cause = failure_analysis.get("primary_cause", "Unknown")
            secondary_causes = failure_analysis.get("secondary_causes", [])
            
            # 基于主要原因的建议
            if primary_cause == "Command Parsing Error":
                recommendations.extend([
                    {
                        "priority": "HIGH",
                        "category": "Command Fix",
                        "action": "使用单行命令，避免YAML多行格式",
                        "details": "将所有命令连接成单行，使用 && 分隔"
                    },
                    {
                        "priority": "HIGH", 
                        "category": "Command Fix",
                        "action": "移除YAML中的 > 符号",
                        "details": "直接使用 command: 而不是 command: >"
                    },
                    {
                        "priority": "MEDIUM",
                        "category": "Testing",
                        "action": "在本地测试命令语法",
                        "details": "先在本地shell中验证命令是否正确"
                    }
                ])
            
            elif primary_cause == "Environment/Package Error":
                recommendations.extend([
                    {
                        "priority": "HIGH",
                        "category": "Environment",
                        "action": "使用预构建的PyTorch环境",
                        "details": "使用 AzureML-pytorch-1.13-ubuntu20.04-py38-cuda11.7-gpu"
                    },
                    {
                        "priority": "HIGH",
                        "category": "Dependencies", 
                        "action": "简化依赖安装",
                        "details": "只安装必需的包，避免版本冲突"
                    },
                    {
                        "priority": "MEDIUM",
                        "category": "Testing",
                        "action": "创建自定义环境",
                        "details": "预先构建包含所有依赖的Docker环境"
                    }
                ])
            
            elif primary_cause == "Compute Resource Error":
                recommendations.extend([
                    {
                        "priority": "HIGH",
                        "category": "Compute",
                        "action": "检查计算集群状态",
                        "details": "确保llava-a100-low-priority集群正在运行"
                    },
                    {
                        "priority": "MEDIUM",
                        "category": "Compute",
                        "action": "使用备用计算集群",
                        "details": "创建新的A100计算集群作为备用"
                    }
                ])
            
            elif primary_cause == "Data Access Error":
                recommendations.extend([
                    {
                        "priority": "HIGH",
                        "category": "Data",
                        "action": "验证数据路径",
                        "details": "确认DADA-100-videos数据已正确上传"
                    },
                    {
                        "priority": "MEDIUM", 
                        "category": "Data",
                        "action": "检查数据权限",
                        "details": "确认工作区对数据存储有读取权限"
                    }
                ])
            
            # 基于次要原因的建议
            for secondary_cause in secondary_causes:
                if "Multi-line command" in secondary_cause:
                    recommendations.append({
                        "priority": "HIGH",
                        "category": "Command Structure",
                        "action": "重构为单行命令",
                        "details": "将多行命令合并为单行，使用 && 连接"
                    })
                
                elif "Chained commands" in secondary_cause:
                    recommendations.append({
                        "priority": "MEDIUM",
                        "category": "Error Handling", 
                        "action": "添加错误处理",
                        "details": "在关键步骤添加错误检查和日志输出"
                    })
                
                elif "GPU/CUDA" in secondary_cause:
                    recommendations.append({
                        "priority": "MEDIUM",
                        "category": "GPU Setup",
                        "action": "验证CUDA环境",
                        "details": "添加GPU可用性检查命令"
                    })
            
            # 通用建议
            recommendations.extend([
                {
                    "priority": "MEDIUM",
                    "category": "Debugging",
                    "action": "启用详细日志",
                    "details": "在命令中添加 set -x 以启用bash调试"
                },
                {
                    "priority": "LOW",
                    "category": "Monitoring",
                    "action": "设置作业监控",
                    "details": "定期检查作业状态和日志"
                }
            ])
            
            # 按优先级排序
            recommendations.sort(key=lambda x: {"HIGH": 1, "MEDIUM": 2, "LOW": 3}[x["priority"]])
            
            logger.info(f"💡 生成了 {len(recommendations)} 个建议")
            for i, rec in enumerate(recommendations[:5], 1):  # 显示前5个
                logger.info(f"{i}. [{rec['priority']}] {rec['action']}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ 生成建议失败: {e}")
            return [{"error": str(e)}]
    
    def _save_investigation_report(self, report: dict):
        """保存调查报告"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"job_investigation_{report['job_name']}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 调查报告已保存: {filename}")
            
            # 生成简化的文本报告
            text_filename = f"job_investigation_{report['job_name']}_{timestamp}.txt"
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(f"Azure ML作业调查报告\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"作业名称: {report['job_name']}\n")
                f.write(f"调查时间: {report['investigation_time']}\n")
                f.write(f"作业状态: {report.get('status', 'Unknown')}\n\n")
                
                f.write("错误详情:\n")
                f.write("-" * 30 + "\n")
                for error in report.get('error_details', []):
                    f.write(f"类型: {error.get('type', 'Unknown')}\n")
                    f.write(f"消息: {error.get('message', 'No message')}\n\n")
                
                f.write("失败原因分析:\n")
                f.write("-" * 30 + "\n")
                analysis = report.get('failure_analysis', {})
                f.write(f"主要原因: {analysis.get('primary_cause', 'Unknown')}\n")
                f.write(f"置信度: {analysis.get('confidence', 'Unknown')}\n")
                f.write(f"次要原因: {', '.join(analysis.get('secondary_causes', []))}\n\n")
                
                f.write("修复建议:\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(report.get('recommendations', []), 1):
                    f.write(f"{i}. [{rec.get('priority', 'UNKNOWN')}] {rec.get('action', 'No action')}\n")
                    f.write(f"   详情: {rec.get('details', 'No details')}\n\n")
            
            logger.info(f"📄 文本报告已保存: {text_filename}")
            
        except Exception as e:
            logger.error(f"❌ 保存报告失败: {e}")
    
    def print_summary(self, report: dict):
        """打印调查摘要"""
        print("\n" + "=" * 80)
        print("🔍 AZURE ML作业调查摘要")
        print("=" * 80)
        print(f"📋 作业名称: {report['job_name']}")
        print(f"📊 作业状态: {report.get('status', 'Unknown')}")
        
        # 错误摘要
        error_count = len(report.get('error_details', []))
        print(f"❌ 发现错误: {error_count} 个")
        
        # 失败原因
        analysis = report.get('failure_analysis', {})
        print(f"🎯 主要原因: {analysis.get('primary_cause', 'Unknown')}")
        print(f"📈 置信度: {analysis.get('confidence', 'Unknown')}")
        
        # 关键建议
        recommendations = report.get('recommendations', [])
        high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
        print(f"💡 高优先级建议: {len(high_priority)} 个")
        
        print("\n🚀 立即行动建议:")
        print("-" * 40)
        for i, rec in enumerate(high_priority[:3], 1):
            print(f"{i}. {rec.get('action', 'No action')}")
        
        print("\n" + "=" * 80)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Azure ML作业故障调查工具')
    parser.add_argument('--job-name', type=str, default='crimson_boniato_k1kg8q62fr',
                       help='要调查的作业名称')
    parser.add_argument('--detailed', action='store_true',
                       help='显示详细调查结果')
    
    args = parser.parse_args()
    
    try:
        # 创建调查器
        investigator = AzureMLJobInvestigator()
        
        # 开始调查
        report = investigator.investigate_job(args.job_name)
        
        # 显示摘要
        investigator.print_summary(report)
        
        if args.detailed:
            print("\n" + "=" * 80)
            print("📋 详细调查结果:")
            print("=" * 80)
            print(json.dumps(report, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"❌ 调查失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()