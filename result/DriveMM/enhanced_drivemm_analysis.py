#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版DriveMM分析脚本
尝试获取更完整的Azure ML作业结果并进行深度分析
"""

import json
import csv
import os
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_drivemm_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedDriveMAnalyzer:
    """增强版DriveMM分析器"""
    
    def __init__(self):
        self.azure_config = {
            "subscription_id": "0d3f39ba-7349-4bd7-8122-649ff18f0a4a",
            "resource_group": "drivelm-rg",
            "workspace_name": "drivelm-ml-workspace",
            "storage_account": "drivelmmstorage2e932dad7"
        }
        self.job_id = "neat_tail_fndr1mjp80"
        
    def check_azure_ml_job_status(self) -> Dict:
        """检查Azure ML作业状态"""
        try:
            cmd = [
                "az", "ml", "job", "show",
                "--name", self.job_id,
                "--resource-group", self.azure_config["resource_group"],
                "--workspace-name", self.azure_config["workspace_name"],
                "--query", "properties",
                "-o", "json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                job_info = json.loads(result.stdout)
                logger.info(f"✅ 作业状态: {job_info.get('status', 'Unknown')}")
                return job_info
            else:
                logger.error(f"❌ 获取作业状态失败: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ 检查作业状态失败: {str(e)}")
            return {}
            
    def search_azure_storage_results(self) -> List[str]:
        """搜索Azure存储中的结果文件"""
        possible_files = []
        
        # 搜索可能的容器和路径
        containers = [
            "azureml",
            f"azureml-blobstore-{self.azure_config['subscription_id'].replace('-', '')}",
            "dada-videos",
            "wisead-videos"
        ]
        
        for container in containers:
            try:
                cmd = [
                    "az", "storage", "blob", "list",
                    "--account-name", self.azure_config["storage_account"],
                    "--container-name", container,
                    "--query", f"[?contains(name,'{self.job_id}') || contains(name,'drivemm') || contains(name,'inference')].{{name:name,size:properties.contentLength}}",
                    "-o", "json"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    files = json.loads(result.stdout)
                    for file_info in files:
                        possible_files.append({
                            "container": container,
                            "name": file_info["name"],
                            "size": file_info["size"]
                        })
                        
            except Exception as e:
                logger.warning(f"⚠️ 搜索容器 {container} 失败: {str(e)}")
                
        return possible_files
        
    def download_potential_results(self, files: List[Dict]) -> List[str]:
        """下载潜在的结果文件"""
        downloaded_files = []
        
        for file_info in files:
            if file_info["name"].endswith('.json') and file_info["size"] > 100:
                try:
                    local_path = f"downloaded_{file_info['name'].replace('/', '_')}"
                    
                    cmd = [
                        "az", "storage", "blob", "download",
                        "--account-name", self.azure_config["storage_account"],
                        "--container-name", file_info["container"],
                        "--name", file_info["name"],
                        "--file", local_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        downloaded_files.append(local_path)
                        logger.info(f"✅ 下载成功: {local_path}")
                    else:
                        logger.warning(f"⚠️ 下载失败: {file_info['name']}")
                        
                except Exception as e:
                    logger.error(f"❌ 下载文件失败: {str(e)}")
                    
        return downloaded_files
        
    def analyze_log_for_insights(self, log_path: str) -> Dict:
        """分析日志文件以获取更多见解"""
        insights = {
            "total_videos_attempted": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "error_patterns": [],
            "model_responses": []
        }
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 分析视频处理数量
            video_patterns = [
                "🤖 DriveMM优化推理:",
                "处理视频:",
                "images_"
            ]
            
            for pattern in video_patterns:
                count = content.count(pattern)
                if count > 0:
                    insights["total_videos_attempted"] = max(
                        insights["total_videos_attempted"], count
                    )
                    
            # 分析错误模式
            error_patterns = [
                "未找到JSON开始标记",
                "使用模板响应",
                "JSON解析失败",
                "生成失败"
            ]
            
            for pattern in error_patterns:
                if pattern in content:
                    insights["error_patterns"].append(pattern)
                    
            # 提取模型响应
            import re
            response_pattern = r"🔍 最终响应前\d+字符: (.*?)(?=\n|$)"
            matches = re.findall(response_pattern, content)
            insights["model_responses"] = matches[:5]  # 前5个响应
            
        except Exception as e:
            logger.error(f"❌ 日志分析失败: {str(e)}")
            
        return insights
        
    def generate_comprehensive_report(self, 
                                    job_info: Dict,
                                    storage_files: List[Dict],
                                    log_insights: Dict) -> str:
        """生成综合分析报告"""
        
        report = []
        report.append("=" * 100)
        report.append("🔍 DriveMM Azure ML作业综合分析报告")
        report.append(f"📅 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🆔 作业ID: {self.job_id}")
        report.append("=" * 100)
        report.append("")
        
        # 作业信息
        report.append("🎯 作业信息")
        if job_info:
            report.append(f"   状态: {job_info.get('status', 'Unknown')}")
            report.append(f"   创建时间: {job_info.get('createdDateTime', 'Unknown')}")
            report.append(f"   结束时间: {job_info.get('endDateTime', 'Unknown')}")
            report.append(f"   计算目标: {job_info.get('computeId', 'Unknown')}")
        else:
            report.append("   ❌ 无法获取作业信息")
        report.append("")
        
        # 存储文件
        report.append("📁 存储文件分析")
        if storage_files:
            report.append(f"   发现 {len(storage_files)} 个相关文件:")
            for file_info in storage_files:
                report.append(f"   - {file_info['name']} ({file_info['size']} bytes)")
        else:
            report.append("   ❌ 未找到相关结果文件")
        report.append("")
        
        # 日志见解
        report.append("📊 日志分析见解")
        if log_insights:
            report.append(f"   尝试处理视频数: {log_insights.get('total_videos_attempted', 0)}")
            report.append(f"   成功预测数: {log_insights.get('successful_predictions', 0)}")
            report.append(f"   失败预测数: {log_insights.get('failed_predictions', 0)}")
            
            if log_insights.get('error_patterns'):
                report.append("   错误模式:")
                for pattern in log_insights['error_patterns']:
                    report.append(f"     - {pattern}")
                    
            if log_insights.get('model_responses'):
                report.append("   模型响应示例:")
                for i, response in enumerate(log_insights['model_responses'][:3]):
                    report.append(f"     {i+1}. {response[:80]}...")
        report.append("")
        
        # 问题诊断
        report.append("🔧 问题诊断")
        problems = []
        
        if not storage_files:
            problems.append("结果文件未找到 - 可能保存失败")
            
        if "未找到JSON开始标记" in log_insights.get('error_patterns', []):
            problems.append("JSON解析失败 - 模型输出格式问题")
            
        if "使用模板响应" in log_insights.get('error_patterns', []):
            problems.append("使用模板响应 - 可能导致结果不准确")
            
        if log_insights.get('total_videos_attempted', 0) < 10:
            problems.append("处理视频数过少 - 可能作业提前终止")
            
        if problems:
            for problem in problems:
                report.append(f"   ❌ {problem}")
        else:
            report.append("   ✅ 未发现明显问题")
        report.append("")
        
        # 建议
        report.append("💡 建议")
        suggestions = [
            "检查Azure ML作业配置，确保输出路径正确",
            "改进模型输出格式，确保JSON结构化输出",
            "增加错误处理机制，避免依赖模板响应",
            "扩大测试数据集，至少包含50-100个视频",
            "考虑使用更稳定的推理框架"
        ]
        
        for suggestion in suggestions:
            report.append(f"   • {suggestion}")
        report.append("")
        
        report.append("=" * 100)
        
        return "\n".join(report)
        
    def run_comprehensive_analysis(self) -> None:
        """运行综合分析"""
        logger.info("🚀 开始DriveMM综合分析...")
        
        # 1. 检查作业状态
        job_info = self.check_azure_ml_job_status()
        
        # 2. 搜索存储文件
        storage_files = self.search_azure_storage_results()
        logger.info(f"📁 发现 {len(storage_files)} 个相关文件")
        
        # 3. 下载潜在结果
        downloaded_files = self.download_potential_results(storage_files)
        logger.info(f"⬇️ 下载了 {len(downloaded_files)} 个文件")
        
        # 4. 分析日志
        log_insights = {}
        log_path = "azure_ml_outputs/artifacts/user_logs/std_log.txt"
        if os.path.exists(log_path):
            log_insights = self.analyze_log_for_insights(log_path)
            logger.info("📊 日志分析完成")
        
        # 5. 生成综合报告
        report = self.generate_comprehensive_report(
            job_info, storage_files, log_insights
        )
        
        # 6. 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"comprehensive_drivemm_analysis_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(report)
        logger.info(f"✅ 综合分析完成，报告保存至: {report_file}")
        
        return downloaded_files

def main():
    """主函数"""
    analyzer = EnhancedDriveMAnalyzer()
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()