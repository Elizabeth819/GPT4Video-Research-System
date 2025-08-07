#!/usr/bin/env python3
"""
计算GPT-4o和Gemini共同处理视频的精确度、召回率、F1等统计指标
"""

import json
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import csv

class FinalPerformanceCalculator:
    def __init__(self):
        self.ground_truth_path = "result/groundtruth_labels.csv"
        self.gpt4o_dir = "result/gpt-4o"
        self.gemini_dir = "result/gemini-testinterval"
        self.output_dir = "result/comparison"
        
        self.ground_truth = {}
        self.gpt4o_predictions = {}
        self.gemini_predictions = {}
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_ground_truth(self):
        """加载ground truth标签"""
        print("📊 加载ground truth标签...")
        
        if not os.path.exists(self.ground_truth_path):
            print(f"❌ Ground truth文件不存在: {self.ground_truth_path}")
            return False
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                video_id = row['video_id'].replace('.avi', '')
                label = row['ground_truth_label']
                
                # 解析标签：提取是否包含ghost probing
                if 'ghost probing' in label.lower():
                    self.ground_truth[video_id] = 1  # 正例：包含ghost probing
                else:
                    self.ground_truth[video_id] = 0  # 负例：不包含ghost probing
        
        print(f"✅ 加载了 {len(self.ground_truth)} 个ground truth标签")
        
        # 统计正负样本
        positive_count = sum(1 for v in self.ground_truth.values() if v == 1)
        negative_count = len(self.ground_truth) - positive_count
        print(f"   正样本(ghost probing): {positive_count}")
        print(f"   负样本(normal): {negative_count}")
        
        return True
    
    def get_processed_videos(self):
        """获取两个模型共同处理的视频列表"""
        print("🔍 查找两个模型共同处理的视频...")
        
        # 获取GPT-4o处理的视频
        gpt4o_videos = set()
        for filename in os.listdir(self.gpt4o_dir):
            if filename.startswith('actionSummary_') and filename.endswith('.json'):
                video_id = filename.replace('actionSummary_', '').replace('.json', '')
                gpt4o_videos.add(video_id)
        
        # 获取Gemini处理的视频
        gemini_videos = set()
        for filename in os.listdir(self.gemini_dir):
            if filename.startswith('actionSummary_') and filename.endswith('.json'):
                video_id = filename.replace('actionSummary_', '').replace('.json', '')
                gemini_videos.add(video_id)
        
        # 找到共同处理的视频
        common_videos = gpt4o_videos & gemini_videos
        
        # 进一步筛选：只包含有ground truth的视频
        common_with_gt = common_videos & set(self.ground_truth.keys())
        
        print(f"   GPT-4o处理的视频数: {len(gpt4o_videos)}")
        print(f"   Gemini处理的视频数: {len(gemini_videos)}")
        print(f"   共同处理的视频数: {len(common_videos)}")
        print(f"   有ground truth的共同视频数: {len(common_with_gt)}")
        
        return sorted(list(common_with_gt))
    
    def extract_ghost_probing_prediction(self, result_data, video_name):
        """从模型结果中提取ghost probing预测"""
        if not isinstance(result_data, list):
            return 0
        
        # 检查所有段落的分析结果
        for segment in result_data:
            if not isinstance(segment, dict):
                continue
                
            # 检查多个字段中是否提到ghost probing相关内容
            text_fields = []
            for field in ['summary', 'actions', 'key_actions', 'next_action', 'key_objects']:
                if field in segment and segment[field]:
                    text_fields.append(str(segment[field]).lower())
            
            combined_text = ' '.join(text_fields)
            
            # 检查是否包含ghost probing相关关键词
            ghost_keywords = [
                'ghost probing', 'ghost', 'probing', 
                'sudden appearance', 'unexpected', 'emerging',
                'appearing suddenly', 'cuts in', 'cut in',
                'overtaking', 'lane change', 'dangerous',
                'risky maneuver', 'unsafe', 'sudden',
                'abrupt', 'intrusion', 'interference',
                'cuts into', 'merging aggressively'
            ]
            
            for keyword in ghost_keywords:
                if keyword in combined_text:
                    return 1  # 预测为正例
        
        return 0  # 预测为负例
    
    def load_model_predictions(self, model_dir, model_name, video_list):
        """加载模型预测结果"""
        print(f"📊 加载{model_name}预测结果...")
        
        predictions = {}
        processed_count = 0
        
        for video_name in video_list:
            result_file = os.path.join(model_dir, f"actionSummary_{video_name}.json")
            
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        predictions[video_name] = self.extract_ghost_probing_prediction(result_data, video_name)
                        processed_count += 1
                except Exception as e:
                    print(f"❌ 加载{model_name}结果失败: {video_name} - {e}")
                    predictions[video_name] = 0  # 默认为负例
            else:
                print(f"⚠️  {model_name}结果文件不存在: {video_name}")
                predictions[video_name] = 0  # 默认为负例
        
        print(f"✅ {model_name}成功处理了 {processed_count}/{len(video_list)} 个视频")
        
        # 统计预测结果
        positive_pred = sum(1 for v in predictions.values() if v == 1)
        negative_pred = len(predictions) - positive_pred
        print(f"   预测正例: {positive_pred}")
        print(f"   预测负例: {negative_pred}")
        
        return predictions
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """计算性能指标"""
        print(f"📈 计算{model_name}性能指标...")
        
        if len(y_true) == 0:
            print(f"❌ 没有可用数据计算{model_name}的指标")
            return None
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 特异性（Specificity）
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 平衡准确率（Balanced Accuracy）
        balanced_accuracy = (recall + specificity) / 2
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_true)
        }
        
        print(f"   准确率: {accuracy:.4f}")
        print(f"   精确度: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   F1分数: {f1:.4f}")
        print(f"   特异性: {specificity:.4f}")
        print(f"   平衡准确率: {balanced_accuracy:.4f}")
        print(f"   混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        return metrics
    
    def create_detailed_analysis(self, gpt4o_metrics, gemini_metrics):
        """创建详细分析"""
        print("📋 创建详细性能分析...")
        
        # 对比两个模型
        comparison = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'balanced_accuracy']:
            gpt4o_val = gpt4o_metrics[metric]
            gemini_val = gemini_metrics[metric]
            
            comparison[metric] = {
                'gpt4o': gpt4o_val,
                'gemini': gemini_val,
                'difference': gemini_val - gpt4o_val,
                'percentage_change': ((gemini_val - gpt4o_val) / gpt4o_val * 100) if gpt4o_val > 0 else (float('inf') if gemini_val > 0 else 0)
            }
        
        # 找出表现更好的模型
        better_model_count = {
            'gpt4o': 0,
            'gemini': 0,
            'tie': 0
        }
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'balanced_accuracy']:
            gpt4o_val = gpt4o_metrics[metric]
            gemini_val = gemini_metrics[metric]
            
            if abs(gpt4o_val - gemini_val) < 0.001:  # 基本相等
                better_model_count['tie'] += 1
            elif gpt4o_val > gemini_val:
                better_model_count['gpt4o'] += 1
            else:
                better_model_count['gemini'] += 1
        
        return comparison, better_model_count
    
    def run_analysis(self):
        """运行完整分析"""
        # 加载数据
        if not self.load_ground_truth():
            return None
        
        # 获取共同处理的视频
        common_videos = self.get_processed_videos()
        
        if len(common_videos) == 0:
            print("❌ 没有找到两个模型共同处理且有ground truth的视频")
            return None
        
        # 加载预测结果
        self.gpt4o_predictions = self.load_model_predictions(self.gpt4o_dir, "GPT-4o", common_videos)
        self.gemini_predictions = self.load_model_predictions(self.gemini_dir, "Gemini", common_videos)
        
        # 准备数据
        y_true = [self.ground_truth[name] for name in common_videos]
        y_pred_gpt4o = [self.gpt4o_predictions[name] for name in common_videos]
        y_pred_gemini = [self.gemini_predictions[name] for name in common_videos]
        
        # 计算指标
        gpt4o_metrics = self.calculate_metrics(y_true, y_pred_gpt4o, "GPT-4o")
        gemini_metrics = self.calculate_metrics(y_true, y_pred_gemini, "Gemini")
        
        if gpt4o_metrics is None or gemini_metrics is None:
            return None
        
        # 详细分析
        comparison, better_model_count = self.create_detailed_analysis(gpt4o_metrics, gemini_metrics)
        
        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        performance_report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "ground_truth_file": self.ground_truth_path,
                "common_videos_count": len(common_videos),
                "positive_samples": sum(y_true),
                "negative_samples": len(y_true) - sum(y_true),
                "video_list": common_videos[:10]  # 只保存前10个作为示例
            },
            "gpt4o_metrics": gpt4o_metrics,
            "gemini_metrics": gemini_metrics,
            "comparison": comparison,
            "model_comparison_summary": better_model_count
        }
        
        # 保存报告
        report_file = os.path.join(self.output_dir, f"final_performance_metrics_{timestamp}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 最终性能分析报告已保存: {report_file}")
        
        return performance_report
    
    def print_summary(self, report):
        """打印性能总结"""
        print("\n" + "="*85)
        print("📊 GPT-4o vs Gemini 最终性能指标对比总结")
        print("="*85)
        
        gpt4o = report["gpt4o_metrics"]
        gemini = report["gemini_metrics"]
        comparison = report["comparison"]
        
        print(f"📁 数据集信息:")
        print(f"   共同处理视频数: {report['dataset_info']['common_videos_count']}")
        print(f"   正样本数: {report['dataset_info']['positive_samples']}")
        print(f"   负样本数: {report['dataset_info']['negative_samples']}")
        
        print(f"\n📈 性能指标对比:")
        print(f"{'指标':<20} {'GPT-4o':<10} {'Gemini':<10} {'差值':<12} {'提升%':<12}")
        print("-" * 74)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'balanced_accuracy']:
            gpt4o_val = gpt4o[metric]
            gemini_val = gemini[metric]
            diff = comparison[metric]['difference']
            pct_change = comparison[metric]['percentage_change']
            
            if abs(pct_change) == float('inf'):
                pct_str = "N/A"
            else:
                pct_str = f"{pct_change:+.1f}%"
            
            print(f"{metric:<20} {gpt4o_val:<10.4f} {gemini_val:<10.4f} {diff:<+12.4f} {pct_str:<12}")
        
        print(f"\n🎯 混淆矩阵对比:")
        print(f"GPT-4o: TP={gpt4o['true_positives']}, TN={gpt4o['true_negatives']}, FP={gpt4o['false_positives']}, FN={gpt4o['false_negatives']}")
        print(f"Gemini: TP={gemini['true_positives']}, TN={gemini['true_negatives']}, FP={gemini['false_positives']}, FN={gemini['false_negatives']}")
        
        summary = report["model_comparison_summary"]
        print(f"\n🏆 总体表现:")
        print(f"   GPT-4o优势指标: {summary['gpt4o']}/6")
        print(f"   Gemini优势指标: {summary['gemini']}/6")
        print(f"   平局指标: {summary['tie']}/6")
        
        if summary['gpt4o'] > summary['gemini']:
            winner = "GPT-4o"
            print(f"   🥇 整体表现更好: {winner}")
        elif summary['gemini'] > summary['gpt4o']:
            winner = "Gemini"
            print(f"   🥇 整体表现更好: {winner}")
        else:
            winner = "平局"
            print(f"   🤝 整体表现: {winner}")
        
        print(f"\n📝 关键发现:")
        if gpt4o['precision'] > gemini['precision']:
            print(f"   • GPT-4o在精确度上表现更好 ({gpt4o['precision']:.4f} vs {gemini['precision']:.4f})")
        elif gemini['precision'] > gpt4o['precision']:
            print(f"   • Gemini在精确度上表现更好 ({gemini['precision']:.4f} vs {gpt4o['precision']:.4f})")
        else:
            print(f"   • 两个模型在精确度上表现相当")
            
        if gpt4o['recall'] > gemini['recall']:
            print(f"   • GPT-4o在召回率上表现更好 ({gpt4o['recall']:.4f} vs {gemini['recall']:.4f})")
        elif gemini['recall'] > gpt4o['recall']:
            print(f"   • Gemini在召回率上表现更好 ({gemini['recall']:.4f} vs {gpt4o['recall']:.4f})")
        else:
            print(f"   • 两个模型在召回率上表现相当")
            
        if gpt4o['f1_score'] > gemini['f1_score']:
            print(f"   • GPT-4o的F1分数更高，整体性能更平衡 ({gpt4o['f1_score']:.4f} vs {gemini['f1_score']:.4f})")
        elif gemini['f1_score'] > gpt4o['f1_score']:
            print(f"   • Gemini的F1分数更高，整体性能更平衡 ({gemini['f1_score']:.4f} vs {gpt4o['f1_score']:.4f})")
        else:
            print(f"   • 两个模型的F1分数相当")
        
        print("\n" + "="*85)
        
        return winner

def main():
    calculator = FinalPerformanceCalculator()
    report = calculator.run_analysis()
    
    if report:
        winner = calculator.print_summary(report)
        return report, winner
    else:
        print("❌ 性能分析失败")
        return None, None

if __name__ == "__main__":
    main()