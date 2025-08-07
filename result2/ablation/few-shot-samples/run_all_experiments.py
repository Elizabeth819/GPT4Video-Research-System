#!/usr/bin/env python3
"""
Few-shot样本数量消融实验总控脚本
按顺序运行1、2、5个样本的消融实验，并生成综合分析报告
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

def run_experiment(experiment_name, script_path, limit=100):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"🚀 开始运行: {experiment_name}")
    print(f"📁 脚本路径: {script_path}")
    print(f"🎯 视频数量: {limit}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行实验脚本
        result = subprocess.run([
            sys.executable, script_path, 
            "--limit", str(limit)
        ], capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} 完成成功！")
            print(f"⏱️  用时: {duration/60:.1f} 分钟")
            print(f"📊 输出: {result.stdout.strip()}")
            return True, duration, result.stdout
        else:
            print(f"❌ {experiment_name} 失败！")
            print(f"错误信息: {result.stderr}")
            return False, duration, result.stderr
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {experiment_name} 运行异常: {str(e)}")
        return False, duration, str(e)

def load_experiment_results(experiment_dir):
    """加载实验结果"""
    results_files = [f for f in os.listdir(experiment_dir) if f.startswith('ablation_') and f.endswith('_results_') and f.endswith('.json')]
    
    if not results_files:
        return None
    
    # 取最新的结果文件
    latest_file = sorted(results_files)[-1]
    results_path = os.path.join(experiment_dir, latest_file)
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载结果失败 {results_path}: {str(e)}")
        return None

def calculate_metrics_from_results(results_data):
    """从结果数据计算性能指标"""
    if not results_data or 'detailed_results' not in results_data:
        return None
    
    detailed_results = results_data['detailed_results']
    
    # 统计混淆矩阵
    tp = sum(1 for r in detailed_results if r['evaluation'] == 'TP')
    tn = sum(1 for r in detailed_results if r['evaluation'] == 'TN')
    fp = sum(1 for r in detailed_results if r['evaluation'] == 'FP')
    fn = sum(1 for r in detailed_results if r['evaluation'] == 'FN')
    
    # 计算性能指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    return {
        "samples": results_data['experiment_info']['ablation_parameters']['few_shot_samples'],
        "processed_videos": len(detailed_results),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy
    }

def generate_comprehensive_report(all_metrics, experiment_log):
    """生成综合分析报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/comprehensive_analysis_{timestamp}.md"
    
    # 基线数据 (Run 8: 3 samples)
    baseline_f1 = 70.0
    baseline_recall = 84.8
    baseline_precision = 59.6
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Few-shot样本数量消融实验综合分析报告

## 实验总览
- **实验时间**: {timestamp}
- **实验目的**: 系统性评估few-shot样本数量对GPT-4o Ghost Probing检测性能的影响
- **基线对比**: Run 8 (3 few-shot samples, F1=70.0%, Recall=84.8%, Precision=59.6%)
- **测试配置**: 相同模型(GPT-4o) + 相同Temperature(0) + 相同基础prompt(Paper_Batch Complex)

## 实验配置对比

| 实验 | Few-shot样本数 | 样本组成 | 目的 |
|------|---------------|----------|------|
| 1样本实验 | 1 | Ghost Probing Detection | 测试最小few-shot学习效果 |
| 2样本实验 | 2 | Ghost Probing + Normal Driving | 测试平衡学习效果 |
| **基线(Run 8)** | **3** | **Ghost + Normal + Vehicle** | **当前最佳配置** |
| 5样本实验 | 5 | 基础3个 + Cyclist + Highway | 测试边际效应 |

## 性能结果对比

### 核心指标汇总

| 样本数 | F1分数 | 召回率 | 精确度 | 特异性 | 平衡准确率 | 处理视频数 |
|--------|--------|--------|--------|--------|----------|-----------|""")
        
        # 添加基线数据
        f.write(f"\n| 3 (基线) | {baseline_f1:.1f}% | {baseline_recall:.1f}% | {baseline_precision:.1f}% | 28.3% | 56.6% | 119 |")
        
        # 添加实验数据
        for metrics in sorted(all_metrics, key=lambda x: x['samples']):
            f.write(f"\n| {metrics['samples']} | {metrics['f1_score']*100:.1f}% | {metrics['recall']*100:.1f}% | {metrics['precision']*100:.1f}% | {metrics['specificity']*100:.1f}% | {metrics['balanced_accuracy']*100:.1f}% | {metrics['processed_videos']} |")
        
        f.write(f"""

### 性能变化趋势分析

#### 📊 F1分数变化趋势
""")
        
        for metrics in sorted(all_metrics, key=lambda x: x['samples']):
            f1_diff = metrics['f1_score']*100 - baseline_f1
            f.write(f"- **{metrics['samples']}样本**: {metrics['f1_score']*100:.1f}% ({f1_diff:+.1f}% vs 基线)\n")
        
        f.write(f"""
#### 🎯 召回率变化趋势 (安全系统关键指标)
""")
        
        for metrics in sorted(all_metrics, key=lambda x: x['samples']):
            recall_diff = metrics['recall']*100 - baseline_recall
            f.write(f"- **{metrics['samples']}样本**: {metrics['recall']*100:.1f}% ({recall_diff:+.1f}% vs 基线)\n")
        
        f.write(f"""
#### 🔍 精确度变化趋势
""")
        
        for metrics in sorted(all_metrics, key=lambda x: x['samples']):
            precision_diff = metrics['precision']*100 - baseline_precision
            f.write(f"- **{metrics['samples']}样本**: {metrics['precision']*100:.1f}% ({precision_diff:+.1f}% vs 基线)\n")
        
        f.write(f"""

## 关键发现

### 🔬 Few-shot学习效果分析
""")
        
        # 分析发现
        if len(all_metrics) >= 2:
            metrics_1 = next((m for m in all_metrics if m['samples'] == 1), None)
            metrics_2 = next((m for m in all_metrics if m['samples'] == 2), None)
            metrics_5 = next((m for m in all_metrics if m['samples'] == 5), None)
            
            if metrics_1:
                f.write(f"""
1. **最小学习能力验证** (1样本 vs 3样本基线):
   - F1分数: {metrics_1['f1_score']*100:.1f}% vs 70.0% = {(metrics_1['f1_score']*100 - 70.0):+.1f}%
   - 单个高质量样本{'能够' if metrics_1['f1_score']*100 > 50 else '不足以'}提供基础的ghost probing检测能力
   - 召回率: {metrics_1['recall']*100:.1f}% (安全系统可接受阈值分析)
""")
            
            if metrics_2:
                f.write(f"""
2. **平衡学习效果** (2样本 vs 3样本基线):
   - F1分数: {metrics_2['f1_score']*100:.1f}% vs 70.0% = {(metrics_2['f1_score']*100 - 70.0):+.1f}%
   - Positive+Negative样本组合的平衡学习效果
   - 相比1样本的改进: {(metrics_2['f1_score'] - metrics_1['f1_score'])*100:+.1f}%
""")
            
            if metrics_5:
                f.write(f"""
3. **边际效应分析** (5样本 vs 3样本基线):
   - F1分数: {metrics_5['f1_score']*100:.1f}% vs 70.0% = {(metrics_5['f1_score']*100 - 70.0):+.1f}%
   - 边际收益: {(metrics_5['f1_score'] - 0.70)*100:+.1f}% (是否值得额外计算成本)
   - 样本多样性对性能的影响分析
""")
        
        f.write(f"""
### 🎯 最优样本数量推荐

基于实验结果分析：
""")
        
        # 找出最佳配置
        best_metrics = max(all_metrics, key=lambda x: x['f1_score'])
        f.write(f"""
- **最佳F1性能**: {best_metrics['samples']}个样本 (F1={best_metrics['f1_score']*100:.1f}%)
- **计算效率权衡**: 考虑性能提升幅度和计算成本
- **安全系统要求**: 优先考虑召回率 ≥ 80% 的配置

### 📈 学术价值

1. **Few-shot学习曲线**: 揭示了样本数量与性能的关系
2. **边际效应量化**: 为few-shot样本数量选择提供数据支持
3. **安全系统优化**: 为自动驾驶安全检测系统的few-shot配置提供指导

## 实验运行日志

""")
        
        # 添加实验日志
        for log_entry in experiment_log:
            f.write(f"- **{log_entry['experiment']}**: {log_entry['status']} (用时: {log_entry['duration']:.1f}分钟)\n")
        
        f.write(f"""
## 结论与建议

1. **最优配置确认**: 基于实验结果，{'验证了当前3样本配置的最优性' if baseline_f1 >= max(m['f1_score']*100 for m in all_metrics) else '发现了更优的样本数量配置'}
2. **实用价值**: 为AAAI26论文的few-shot学习消融实验提供了完整的数据支持
3. **工程应用**: 为实际部署时的few-shot样本数量选择提供了科学依据

## 文件路径
- 综合分析报告: `comprehensive_analysis_{timestamp}.md`
- 1样本实验: `1-sample/`
- 2样本实验: `2-samples/`  
- 5样本实验: `5-samples/`
""")
    
    print(f"📊 综合分析报告已生成: {report_path}")
    return report_path

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Few-shot样本数量消融实验总控')
    parser.add_argument('--limit', type=int, default=20, help='每个实验的视频数量限制 (建议先用20测试)')
    parser.add_argument('--experiments', nargs='+', default=['1', '2', '5'], 
                      help='要运行的实验 (1, 2, 5)')
    
    args = parser.parse_args()
    
    print(f"🎯 Few-shot样本数量消融实验开始")
    print(f"📊 每个实验处理 {args.limit} 个视频")
    print(f"🧪 实验列表: {args.experiments}")
    
    # 实验配置
    base_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples"
    experiments = {
        '1': {
            'name': '1样本消融实验',
            'script': os.path.join(base_dir, '1-sample', 'run8_ablation_1sample.py'),
            'dir': os.path.join(base_dir, '1-sample')
        },
        '2': {
            'name': '2样本消融实验', 
            'script': os.path.join(base_dir, '2-samples', 'run8_ablation_2samples.py'),
            'dir': os.path.join(base_dir, '2-samples')
        },
        '5': {
            'name': '5样本消融实验',
            'script': os.path.join(base_dir, '5-samples', 'run8_ablation_5samples.py'),
            'dir': os.path.join(base_dir, '5-samples')
        }
    }
    
    # 运行实验
    experiment_log = []
    all_metrics = []
    
    for exp_id in args.experiments:
        if exp_id not in experiments:
            print(f"⚠️  未知实验ID: {exp_id}")
            continue
        
        exp_config = experiments[exp_id]
        
        # 运行实验
        success, duration, output = run_experiment(
            exp_config['name'], 
            exp_config['script'], 
            args.limit
        )
        
        # 记录实验日志
        experiment_log.append({
            'experiment': exp_config['name'],
            'status': '成功' if success else '失败',
            'duration': duration / 60,  # 转换为分钟
            'output': output
        })
        
        # 如果成功，加载结果
        if success:
            time.sleep(2)  # 等待文件写入完成
            results_data = load_experiment_results(exp_config['dir'])
            if results_data:
                metrics = calculate_metrics_from_results(results_data)
                if metrics:
                    all_metrics.append(metrics)
                    print(f"📊 {exp_config['name']} 结果: F1={metrics['f1_score']*100:.1f}%")
        
        print(f"\n⏸️  等待5秒后继续下一个实验...")
        time.sleep(5)
    
    # 生成综合分析报告
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"📊 生成综合分析报告")
        print(f"{'='*60}")
        
        report_path = generate_comprehensive_report(all_metrics, experiment_log)
        
        print(f"\n🎉 所有消融实验完成！")
        print(f"📊 成功完成 {len(all_metrics)}/{len(args.experiments)} 个实验")
        print(f"📁 综合报告: {report_path}")
        
        # 显示简要结果
        print(f"\n📈 结果摘要:")
        for metrics in sorted(all_metrics, key=lambda x: x['samples']):
            print(f"  {metrics['samples']}样本: F1={metrics['f1_score']*100:.1f}%, Recall={metrics['recall']*100:.1f}%")
        
    else:
        print(f"\n❌ 没有成功的实验结果，无法生成综合报告")

if __name__ == "__main__":
    main()