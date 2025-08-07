#!/usr/bin/env python3
"""
Few-shot Sample Number Ablation Study Master Script
运行1, 2, 5个few-shot样本的消融实验对比
"""

import os
import sys
import subprocess
import time
import datetime
import json
import logging

def setup_logging():
    """设置主日志"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/ablation_master_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), timestamp

def run_experiment(script_path, experiment_name, limit=100):
    """运行单个消融实验"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 开始运行 {experiment_name} 实验")
    
    try:
        # 运行实验脚本
        cmd = [sys.executable, script_path, "--limit", str(limit)]
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*3)  # 3小时超时
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {experiment_name} 实验完成 (耗时: {duration/60:.1f}分钟)")
            logger.info(f"输出: {result.stdout}")
            return True, duration
        else:
            logger.error(f"❌ {experiment_name} 实验失败")
            logger.error(f"错误输出: {result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {experiment_name} 实验超时 (3小时)")
        return False, 0
    except Exception as e:
        logger.error(f"💥 {experiment_name} 实验异常: {str(e)}")
        return False, 0

def collect_results():
    """收集所有实验结果"""
    logger = logging.getLogger(__name__)
    logger.info("📊 收集实验结果")
    
    experiments = {
        "1-sample": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/1-sample",
        "2-samples": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples", 
        "5-samples": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/5-samples"
    }
    
    results_summary = {}
    
    for exp_name, exp_dir in experiments.items():
        try:
            # 查找最新的结果文件
            result_files = [f for f in os.listdir(exp_dir) if f.startswith(f"ablation_{exp_name.replace('-', '')}_results_") and f.endswith('.json')]
            if result_files:
                latest_file = sorted(result_files)[-1]
                result_path = os.path.join(exp_dir, latest_file)
                
                with open(result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 提取关键指标
                if 'detailed_results' in result_data:
                    results = result_data['detailed_results']
                    tp = sum(1 for r in results if r['evaluation'] == 'TP')
                    tn = sum(1 for r in results if r['evaluation'] == 'TN')
                    fp = sum(1 for r in results if r['evaluation'] == 'FP')
                    fn = sum(1 for r in results if r['evaluation'] == 'FN')
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                    
                    results_summary[exp_name] = {
                        "f1_score": f1_score,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                        "processed_videos": len(results),
                        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
                    }
                    
                    logger.info(f"📈 {exp_name}: F1={f1_score*100:.1f}%, Precision={precision*100:.1f}%, Recall={recall*100:.1f}%")
                else:
                    logger.warning(f"⚠️  {exp_name}: 未找到有效结果数据")
            else:
                logger.warning(f"⚠️  {exp_name}: 未找到结果文件")
                
        except Exception as e:
            logger.error(f"❌ 处理 {exp_name} 结果时出错: {str(e)}")
    
    return results_summary

def generate_comparison_report(results_summary, timestamp):
    """生成对比报告"""
    logger = logging.getLogger(__name__)
    report_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/ablation_comparison_report_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"""# Few-shot Sample Number Ablation Study Report

## 实验概述
- **实验时间**: {timestamp}
- **实验目的**: 对比不同few-shot样本数量对GPT-4o ghost probing检测的影响
- **测试配置**: 
  - 1个样本 vs 2个样本 vs 5个样本
  - 每个配置处理100个视频
  - 相同基础prompt和评估数据集
  - Temperature=0，确保结果一致性

## 实验结果对比

| 配置 | F1 Score | Precision | Recall | Accuracy | 处理视频数 |
|------|----------|-----------|---------|----------|------------|
""")
        
        # 添加结果表格
        for exp_name in ["1-sample", "2-samples", "5-samples"]:
            if exp_name in results_summary:
                r = results_summary[exp_name]
                f.write(f"| {exp_name} | {r['f1_score']*100:.1f}% | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['accuracy']*100:.1f}% | {r['processed_videos']} |\n")
            else:
                f.write(f"| {exp_name} | N/A | N/A | N/A | N/A | 0 |\n")
        
        f.write(f"""
## 详细分析

### 性能趋势
""")
        
        # 分析性能趋势
        if len(results_summary) >= 2:
            f1_scores = [(exp, results_summary[exp]['f1_score']) for exp in ["1-sample", "2-samples", "5-samples"] if exp in results_summary]
            if len(f1_scores) >= 2:
                f.write(f"- **F1 Score趋势**: {' → '.join([f'{exp}: {score*100:.1f}%' for exp, score in f1_scores])}\n")
        
        f.write(f"""
### 混淆矩阵对比
""")
        for exp_name in ["1-sample", "2-samples", "5-samples"]:
            if exp_name in results_summary:
                cm = results_summary[exp_name]['confusion_matrix']
                f.write(f"""
#### {exp_name}
- TP: {cm['TP']}, TN: {cm['TN']}, FP: {cm['FP']}, FN: {cm['FN']}
""")
        
        f.write(f"""
## 实验结论

### Few-shot学习效果分析
1. **样本数量影响**: 分析1个样本到5个样本的性能变化
2. **边际收益**: 评估增加样本数量的边际效应
3. **计算成本权衡**: 样本数量与API调用成本的平衡

### 推荐配置
基于实验结果，推荐使用以下配置以获得最佳性能与成本平衡。

## 文件路径
- 主实验日志: `ablation_master_{timestamp}.log`
- 各子实验详细结果在对应子目录中
""")
    
    logger.info(f"📋 对比报告已生成: {report_path}")
    return report_path

def main():
    """主函数"""
    logger, timestamp = setup_logging()
    logger.info("🎯 开始Few-shot Sample Number消融实验")
    
    # 定义实验脚本
    experiments = [
        {
            "name": "1 Few-shot Sample",
            "script": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/1-sample/run8_ablation_1sample.py"
        },
        {
            "name": "2 Few-shot Samples", 
            "script": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples/run8_ablation_2samples.py"
        },
        {
            "name": "5 Few-shot Samples",
            "script": "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/5-samples/run8_ablation_5samples.py"
        }
    ]
    
    # 运行所有实验
    successful_experiments = 0
    total_duration = 0
    
    for exp in experiments:
        success, duration = run_experiment(exp["script"], exp["name"], limit=100)
        total_duration += duration
        if success:
            successful_experiments += 1
        
        # 实验间暂停
        if exp != experiments[-1]:
            logger.info("⏳ 等待5秒后开始下一个实验...")
            time.sleep(5)
    
    # 收集和分析结果
    logger.info(f"📊 实验完成统计: {successful_experiments}/{len(experiments)} 成功")
    logger.info(f"⏱️  总耗时: {total_duration/60:.1f}分钟")
    
    if successful_experiments > 0:
        results_summary = collect_results()
        report_path = generate_comparison_report(results_summary, timestamp)
        
        logger.info("🎉 消融实验全部完成！")
        logger.info(f"📋 对比报告: {report_path}")
        
        # 打印简要结果
        print(f"\n{'='*60}")
        print("🎯 Few-shot Sample Number Ablation Study Results")
        print(f"{'='*60}")
        for exp_name in ["1-sample", "2-samples", "5-samples"]:
            if exp_name in results_summary:
                r = results_summary[exp_name]
                print(f"{exp_name:12} | F1: {r['f1_score']*100:5.1f}% | P: {r['precision']*100:5.1f}% | R: {r['recall']*100:5.1f}%")
        print(f"{'='*60}")
    else:
        logger.error("❌ 所有实验都失败了")

if __name__ == "__main__":
    main()