#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DriveMM鬼探头检测结果分析脚本
分析RADICAL修复版本DriveMM的推理结果与ground truth的对比
"""

import json
import csv
from typing import Dict, List, Tuple, Optional
import re
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drivemm_ghost_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DriveMGhostAnalyzer:
    """DriveMM鬼探头检测分析器"""
    
    def __init__(self, results_file: str, ground_truth_file: str):
        """
        初始化分析器
        
        Args:
            results_file: DriveMM结果文件路径
            ground_truth_file: Ground truth标签文件路径
        """
        self.results_file = results_file
        self.ground_truth_file = ground_truth_file
        self.results = None
        self.ground_truth = None
        
    def load_data(self) -> None:
        """加载数据文件"""
        try:
            # 加载DriveMM结果
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                logger.info(f"✅ 已加载DriveMM结果: {len(self.results)} 个视频")
            else:
                logger.error(f"❌ DriveMM结果文件不存在: {self.results_file}")
                return
                
            # 加载Ground Truth
            if os.path.exists(self.ground_truth_file):
                self.ground_truth = []
                with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        self.ground_truth.append(row)
                logger.info(f"✅ 已加载Ground Truth: {len(self.ground_truth)} 个视频")
            else:
                logger.error(f"❌ Ground Truth文件不存在: {self.ground_truth_file}")
                return
                
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {str(e)}")
            
    def parse_ground_truth(self, gt_label: str) -> Dict[str, Optional[str]]:
        """
        解析ground truth标签
        
        Args:
            gt_label: 原始标签字符串
            
        Returns:
            解析后的标签字典
        """
        if not gt_label or gt_label == "none":
            return {"has_ghost_probing": False, "timestamp": None, "label": "none"}
            
        if "ghost probing" in gt_label.lower():
            # 提取时间戳
            timestamp_match = re.search(r'(\d+)s?:', gt_label)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            return {
                "has_ghost_probing": True, 
                "timestamp": timestamp,
                "label": "ghost_probing"
            }
            
        if "cut-in" in gt_label.lower():
            return {"has_ghost_probing": False, "timestamp": None, "label": "cut_in"}
            
        return {"has_ghost_probing": False, "timestamp": None, "label": "other"}
        
    def parse_drivemm_prediction(self, prediction: str) -> Dict[str, any]:
        """
        解析DriveMM预测结果
        
        Args:
            prediction: 预测结果字符串
            
        Returns:
            解析后的预测字典
        """
        if prediction.lower() == "ghost_probing":
            return {"has_ghost_probing": True, "confidence": "high"}
        elif prediction.lower() == "normal":
            return {"has_ghost_probing": False, "confidence": "high"}
        else:
            return {"has_ghost_probing": False, "confidence": "low"}
            
    def analyze_results(self) -> Dict[str, any]:
        """分析结果并计算指标"""
        if self.results is None or self.ground_truth is None:
            logger.error("❌ 数据未加载，无法分析")
            return {}
            
        analysis = {
            "total_videos": len(self.results),
            "ground_truth_videos": len(self.ground_truth),
            "matched_videos": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
            "detailed_results": []
        }
        
        # 创建ground truth字典以便快速查找
        gt_dict = {}
        for row in self.ground_truth:
            video_id = row['video_id'].replace('.avi', '')
            gt_dict[video_id] = self.parse_ground_truth(row['ground_truth_label'])
            
        # 分析每个视频的结果
        for result in self.results:
            video_id = result['video_id']
            
            if video_id in gt_dict:
                analysis["matched_videos"] += 1
                
                # 获取ground truth和预测结果
                gt = gt_dict[video_id]
                pred = self.parse_drivemm_prediction(result['prediction'])
                
                # 计算混淆矩阵
                if gt["has_ghost_probing"] and pred["has_ghost_probing"]:
                    analysis["true_positives"] += 1
                    result_type = "TP"
                elif gt["has_ghost_probing"] and not pred["has_ghost_probing"]:
                    analysis["false_negatives"] += 1
                    result_type = "FN"
                elif not gt["has_ghost_probing"] and pred["has_ghost_probing"]:
                    analysis["false_positives"] += 1
                    result_type = "FP"
                else:
                    analysis["true_negatives"] += 1
                    result_type = "TN"
                    
                # 记录详细结果
                detailed_result = {
                    "video_id": video_id,
                    "ground_truth": gt,
                    "prediction": pred,
                    "result_type": result_type,
                    "raw_response": result.get('raw_response', ''),
                    "reasoning": result.get('reasoning', '')
                }
                analysis["detailed_results"].append(detailed_result)
                
        # 计算性能指标
        tp = analysis["true_positives"]
        fp = analysis["false_positives"]
        tn = analysis["true_negatives"]
        fn = analysis["false_negatives"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        analysis["metrics"] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }
        }
        
        return analysis
        
    def generate_report(self, analysis: Dict[str, any]) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 80)
        report.append("🤖 DriveMM鬼探头检测分析报告")
        report.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # 基本统计
        report.append("📊 基本统计")
        report.append(f"   DriveMM处理视频数: {analysis['total_videos']}")
        report.append(f"   Ground Truth视频数: {analysis['ground_truth_videos']}")
        report.append(f"   匹配的视频数: {analysis['matched_videos']}")
        report.append("")
        
        # 混淆矩阵
        cm = analysis["metrics"]["confusion_matrix"]
        report.append("🔍 混淆矩阵")
        report.append(f"   真正例 (TP): {cm['true_positives']}")
        report.append(f"   假正例 (FP): {cm['false_positives']}")
        report.append(f"   真负例 (TN): {cm['true_negatives']}")
        report.append(f"   假负例 (FN): {cm['false_negatives']}")
        report.append("")
        
        # 性能指标
        metrics = analysis["metrics"]
        report.append("📈 性能指标")
        report.append(f"   精确度 (Precision): {metrics['precision']:.3f}")
        report.append(f"   召回率 (Recall): {metrics['recall']:.3f}")
        report.append(f"   特异性 (Specificity): {metrics['specificity']:.3f}")
        report.append(f"   准确率 (Accuracy): {metrics['accuracy']:.3f}")
        report.append(f"   F1得分: {metrics['f1_score']:.3f}")
        report.append("")
        
        # 详细结果
        if analysis["detailed_results"]:
            report.append("📋 详细结果")
            for result in analysis["detailed_results"]:
                report.append(f"   视频: {result['video_id']}")
                report.append(f"   Ground Truth: {result['ground_truth']}")
                report.append(f"   预测结果: {result['prediction']}")
                report.append(f"   结果类型: {result['result_type']}")
                report.append(f"   原始响应: {result['raw_response'][:100]}...")
                report.append("   " + "-" * 50)
        
        # 问题分析
        report.append("⚠️ 问题分析")
        if analysis['total_videos'] == 3:
            report.append("   ⚠️ 警告: 只处理了3个视频，样本量过小")
        
        if analysis["metrics"]["confusion_matrix"]["false_positives"] > 0:
            report.append(f"   ⚠️ 存在{analysis['metrics']['confusion_matrix']['false_positives']}个假正例")
            
        if analysis["metrics"]["confusion_matrix"]["false_negatives"] > 0:
            report.append(f"   ⚠️ 存在{analysis['metrics']['confusion_matrix']['false_negatives']}个假负例")
            
        # 建议
        report.append("")
        report.append("💡 建议")
        if analysis['total_videos'] < 10:
            report.append("   • 增加测试视频数量以获得更可靠的评估")
        if metrics['precision'] < 0.8:
            report.append("   • 精确度偏低，需要减少误报")
        if metrics['recall'] < 0.8:
            report.append("   • 召回率偏低，需要提高检测能力")
            
        return "\n".join(report)
        
    def save_results(self, analysis: Dict[str, any], output_file: str) -> None:
        """保存分析结果"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 分析结果已保存: {output_file}")
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {str(e)}")
            
    def run_analysis(self) -> Dict[str, any]:
        """运行完整分析流程"""
        logger.info("🚀 开始DriveMM鬼探头检测分析...")
        
        # 加载数据
        self.load_data()
        
        # 分析结果
        analysis = self.analyze_results()
        
        # 生成报告
        report = self.generate_report(analysis)
        
        # 打印报告
        print(report)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"drivemm_ghost_analysis_{timestamp}.json"
        self.save_results(analysis, output_file)
        
        # 保存报告
        report_file = f"drivemm_ghost_analysis_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"✅ 分析报告已保存: {report_file}")
        
        return analysis

def main():
    """主函数"""
    # 文件路径
    results_file = "azure_drivemm_real_inference_results.json"
    ground_truth_file = "result/groundtruth_labels.csv"
    
    # 检查文件是否存在
    if not os.path.exists(results_file):
        print(f"❌ DriveMM结果文件不存在: {results_file}")
        return
        
    if not os.path.exists(ground_truth_file):
        print(f"❌ Ground Truth文件不存在: {ground_truth_file}")
        return
        
    # 创建分析器并运行
    analyzer = DriveMGhostAnalyzer(results_file, ground_truth_file)
    analysis = analyzer.run_analysis()
    
    print(f"\n🎉 分析完成！")
    print(f"📊 处理了 {analysis.get('total_videos', 0)} 个视频")
    print(f"📈 准确率: {analysis.get('metrics', {}).get('accuracy', 0):.3f}")
    print(f"🎯 精确度: {analysis.get('metrics', {}).get('precision', 0):.3f}")
    print(f"📍 召回率: {analysis.get('metrics', {}).get('recall', 0):.3f}")

if __name__ == "__main__":
    main()