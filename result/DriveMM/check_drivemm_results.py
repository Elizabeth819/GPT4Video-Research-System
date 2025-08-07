#!/usr/bin/env python3
"""
检查DriveMM推理结果并生成报告
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

def run_command(command):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def get_job_status(job_name):
    """获取作业状态"""
    command = f"az ml job show --name {job_name} --resource-group drivelm-rg --workspace-name drivelm-ml-workspace --query '{{Name:name,Status:status,StartTime:creation_context.created_at,EndTime:end_time}}' --output json"
    
    stdout, stderr, returncode = run_command(command)
    
    if returncode == 0:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return None
    else:
        print(f"❌ 获取作业状态失败: {stderr}")
        return None

def download_job_outputs(job_name):
    """下载作业输出"""
    print(f"📥 尝试下载作业输出: {job_name}")
    
    # 创建输出目录
    output_dir = f"job_outputs_{job_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载输出
    command = f"az ml job download --name {job_name} --resource-group drivelm-rg --workspace-name drivelm-ml-workspace --output-name outputs --path {output_dir}"
    stdout, stderr, returncode = run_command(command)
    
    if returncode == 0:
        print(f"✅ 作业输出已下载到: {output_dir}")
        return output_dir
    else:
        print(f"❌ 下载作业输出失败: {stderr}")
        return None

def analyze_drivemm_results(output_dir):
    """分析DriveMM结果"""
    print(f"📊 分析DriveMM结果: {output_dir}")
    
    results_file = os.path.join(output_dir, "azure_drivemm_real_inference_results.json")
    
    if os.path.exists(results_file):
        print(f"✅ 找到结果文件: {results_file}")
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            summary = results.get('real_drivemm_analysis_summary', {})
            detailed_results = results.get('detailed_results', [])
            
            print("=" * 60)
            print("🎉 DriveMM分析结果摘要")
            print("=" * 60)
            print(f"📋 模型: {summary.get('model', 'Unknown')}")
            print(f"🔗 模型来源: {summary.get('model_source', 'Unknown')}")
            print(f"💾 总视频数: {summary.get('total_videos_analyzed', 0)}")
            
            detection_results = summary.get('detection_results', {})
            print(f"🔍 检测结果:")
            print(f"   - 高确信度鬼探头: {detection_results.get('high_confidence_ghost_probing', 0)}")
            print(f"   - 潜在鬼探头: {detection_results.get('potential_ghost_probing', 0)}")
            print(f"   - 正常交通: {detection_results.get('normal_traffic', 0)}")
            
            detection_rates = summary.get('detection_rates', {})
            print(f"📊 检测率:")
            print(f"   - 鬼探头检测率: {detection_rates.get('ghost_probing_rate', 0):.2%}")
            print(f"   - 潜在鬼探头率: {detection_rates.get('potential_ghost_probing_rate', 0):.2%}")
            print(f"   - 正常交通率: {detection_rates.get('normal_traffic_rate', 0):.2%}")
            
            return results
            
        except Exception as e:
            print(f"❌ 分析结果失败: {e}")
            return None
    else:
        print(f"❌ 未找到结果文件: {results_file}")
        return None

def generate_comparison_report(drivemm_results):
    """生成与GPT-4.1的对比报告"""
    print("📝 生成对比报告...")
    
    # 读取GPT-4.1结果（如果存在）
    gpt41_results_file = "result/gpt4o-100-3rd/evaluation_results.json"
    
    report = {
        "comparison_timestamp": datetime.now().isoformat(),
        "drivemm_results": drivemm_results,
        "gpt41_baseline": None,
        "comparison_summary": {}
    }
    
    if os.path.exists(gpt41_results_file):
        try:
            with open(gpt41_results_file, 'r', encoding='utf-8') as f:
                gpt41_data = json.load(f)
                report["gpt41_baseline"] = gpt41_data
        except Exception as e:
            print(f"⚠️ 无法读取GPT-4.1基准结果: {e}")
    
    # 保存对比报告
    report_file = "drivemm_vs_gpt41_comparison_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 对比报告已保存: {report_file}")
    return report_file

def main():
    """主函数"""
    job_name = "red_diamond_xfbmkt8klp"
    
    print("🔍 DriveMM结果检查器")
    print("=" * 60)
    print(f"📋 作业名称: {job_name}")
    print("=" * 60)
    
    # 检查作业状态
    job_info = get_job_status(job_name)
    if job_info:
        status = job_info.get('Status', 'Unknown')
        print(f"📊 作业状态: {status}")
        
        if status == "Completed":
            print("✅ 作业已完成，准备下载结果...")
            
            # 下载作业输出
            output_dir = download_job_outputs(job_name)
            if output_dir:
                # 分析结果
                drivemm_results = analyze_drivemm_results(output_dir)
                if drivemm_results:
                    # 生成对比报告
                    report_file = generate_comparison_report(drivemm_results)
                    print(f"🎉 分析完成！对比报告: {report_file}")
                else:
                    print("❌ 结果分析失败")
            else:
                print("❌ 下载作业输出失败")
        else:
            print(f"⏳ 作业状态: {status}，请等待完成后再运行此脚本")
    else:
        print("❌ 无法获取作业状态")

if __name__ == "__main__":
    main()