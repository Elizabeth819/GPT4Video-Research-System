#!/usr/bin/env python3
"""
对比GPT-4o和Gemini在100个视频上的表现
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class VideoComparator:
    def __init__(self):
        self.gpt4o_dir = "result/gpt-4o"
        self.gemini_dir = "result/gemini-testinterval"
        self.output_dir = "result/comparison"
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.gpt4o_results = {}
        self.gemini_results = {}
        
    def load_results(self):
        """加载两个模型的结果"""
        print("📊 加载GPT-4o结果...")
        
        # 获取前100个视频列表
        videos_dir = "DADA-2000-videos"
        all_videos = sorted([f for f in os.listdir(videos_dir) if f.endswith('.avi')])[:100]
        
        gpt4o_count = 0
        gemini_count = 0
        
        for video in all_videos:
            video_name = video.replace('.avi', '')
            
            # 加载GPT-4o结果
            gpt4o_file = os.path.join(self.gpt4o_dir, f"actionSummary_{video_name}.json")
            if os.path.exists(gpt4o_file):
                try:
                    with open(gpt4o_file, 'r', encoding='utf-8') as f:
                        self.gpt4o_results[video_name] = json.load(f)
                        gpt4o_count += 1
                except Exception as e:
                    print(f"❌ 加载GPT-4o结果失败: {video_name} - {e}")
            
            # 加载Gemini结果
            gemini_file = os.path.join(self.gemini_dir, f"actionSummary_{video_name}.json")
            if os.path.exists(gemini_file):
                try:
                    with open(gemini_file, 'r', encoding='utf-8') as f:
                        self.gemini_results[video_name] = json.load(f)
                        gemini_count += 1
                except Exception as e:
                    print(f"❌ 加载Gemini结果失败: {video_name} - {e}")
        
        print(f"✅ GPT-4o结果: {gpt4o_count}/100")
        print(f"✅ Gemini结果: {gemini_count}/100")
        
        return gpt4o_count, gemini_count
    
    def analyze_response_structure(self):
        """分析响应结构"""
        print("🔍 分析响应结构...")
        
        gpt4o_fields = defaultdict(int)
        gemini_fields = defaultdict(int)
        
        # 分析GPT-4o响应结构
        for video_name, result in self.gpt4o_results.items():
            if isinstance(result, list):
                for segment in result:
                    if isinstance(segment, dict):
                        for field in segment.keys():
                            gpt4o_fields[field] += 1
        
        # 分析Gemini响应结构
        for video_name, result in self.gemini_results.items():
            if isinstance(result, list):
                for segment in result:
                    if isinstance(segment, dict):
                        for field in segment.keys():
                            gemini_fields[field] += 1
        
        print(f"GPT-4o常见字段: {dict(gpt4o_fields)}")
        print(f"Gemini常见字段: {dict(gemini_fields)}")
        
        return gpt4o_fields, gemini_fields
    
    def analyze_content_length(self):
        """分析内容长度"""
        print("📏 分析内容长度...")
        
        gpt4o_lengths = []
        gemini_lengths = []
        
        # 分析GPT-4o内容长度
        for video_name, result in self.gpt4o_results.items():
            if isinstance(result, list):
                for segment in result:
                    if isinstance(segment, dict):
                        summary = segment.get('summary', '')
                        if summary:
                            gpt4o_lengths.append(len(summary))
        
        # 分析Gemini内容长度
        for video_name, result in self.gemini_results.items():
            if isinstance(result, list):
                for segment in result:
                    if isinstance(segment, dict):
                        summary = segment.get('summary', '')
                        if summary:
                            gemini_lengths.append(len(summary))
        
        gpt4o_avg = np.mean(gpt4o_lengths) if gpt4o_lengths else 0
        gemini_avg = np.mean(gemini_lengths) if gemini_lengths else 0
        
        print(f"GPT-4o平均摘要长度: {gpt4o_avg:.2f}字符")
        print(f"Gemini平均摘要长度: {gemini_avg:.2f}字符")
        
        return gpt4o_lengths, gemini_lengths
    
    def analyze_common_videos(self):
        """分析两个模型都处理的视频"""
        print("🎯 分析共同处理的视频...")
        
        common_videos = set(self.gpt4o_results.keys()) & set(self.gemini_results.keys())
        print(f"共同处理的视频数量: {len(common_videos)}")
        
        # 对共同视频进行详细分析
        comparison_data = []
        
        for video_name in common_videos:
            gpt4o_result = self.gpt4o_results[video_name]
            gemini_result = self.gemini_results[video_name]
            
            gpt4o_segments = len(gpt4o_result) if isinstance(gpt4o_result, list) else 0
            gemini_segments = len(gemini_result) if isinstance(gemini_result, list) else 0
            
            comparison_data.append({
                'video_name': video_name,
                'gpt4o_segments': gpt4o_segments,
                'gemini_segments': gemini_segments,
                'segment_diff': abs(gpt4o_segments - gemini_segments)
            })
        
        df = pd.DataFrame(comparison_data)
        
        print(f"平均段落数 - GPT-4o: {df['gpt4o_segments'].mean():.2f}")
        print(f"平均段落数 - Gemini: {df['gemini_segments'].mean():.2f}")
        print(f"段落数差异平均值: {df['segment_diff'].mean():.2f}")
        
        return df, common_videos
    
    def create_comparison_report(self):
        """创建对比报告"""
        print("📋 创建对比报告...")
        
        gpt4o_count, gemini_count = self.load_results()
        gpt4o_fields, gemini_fields = self.analyze_response_structure()
        gpt4o_lengths, gemini_lengths = self.analyze_content_length()
        comparison_df, common_videos = self.analyze_common_videos()
        
        # 生成报告
        report = {
            "comparison_timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "total_videos_tested": 100,
                "videos_source": "DADA-2000-videos (前100个)",
                "gpt4o_processed": gpt4o_count,
                "gemini_processed": gemini_count,
                "common_videos": len(common_videos)
            },
            "structure_analysis": {
                "gpt4o_fields": dict(gpt4o_fields),
                "gemini_fields": dict(gemini_fields)
            },
            "content_analysis": {
                "gpt4o_avg_summary_length": float(np.mean(gpt4o_lengths)) if gpt4o_lengths else 0,
                "gemini_avg_summary_length": float(np.mean(gemini_lengths)) if gemini_lengths else 0,
                "gpt4o_total_segments": len(gpt4o_lengths),
                "gemini_total_segments": len(gemini_lengths)
            },
            "comparison_metrics": {
                "avg_segments_gpt4o": float(comparison_df['gpt4o_segments'].mean()),
                "avg_segments_gemini": float(comparison_df['gemini_segments'].mean()),
                "avg_segment_difference": float(comparison_df['segment_diff'].mean()),
                "max_segment_difference": int(comparison_df['segment_diff'].max()),
                "videos_with_identical_segments": int(sum(comparison_df['segment_diff'] == 0))
            }
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"gpt4o_vs_gemini_100videos_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存详细对比数据
        comparison_df.to_csv(os.path.join(self.output_dir, f"detailed_comparison_{timestamp}.csv"), 
                           index=False, encoding='utf-8')
        
        print(f"✅ 报告已保存: {report_file}")
        return report
    
    def print_summary(self, report):
        """打印总结"""
        print("\n" + "="*60)
        print("📊 GPT-4o vs Gemini 100视频对比总结")
        print("="*60)
        
        dataset = report["dataset_info"]
        content = report["content_analysis"]
        comparison = report["comparison_metrics"]
        
        print(f"📁 数据集: {dataset['videos_source']}")
        print(f"🎯 测试视频数: {dataset['total_videos_tested']}")
        print(f"✅ GPT-4o处理: {dataset['gpt4o_processed']}")
        print(f"✅ Gemini处理: {dataset['gemini_processed']}")
        print(f"🔄 共同处理: {dataset['common_videos']}")
        
        print(f"\n📏 内容分析:")
        print(f"   GPT-4o平均摘要长度: {content['gpt4o_avg_summary_length']:.2f}字符")
        print(f"   Gemini平均摘要长度: {content['gemini_avg_summary_length']:.2f}字符")
        print(f"   GPT-4o总段落数: {content['gpt4o_total_segments']}")
        print(f"   Gemini总段落数: {content['gemini_total_segments']}")
        
        print(f"\n🎯 对比指标:")
        print(f"   GPT-4o平均段落数: {comparison['avg_segments_gpt4o']:.2f}")
        print(f"   Gemini平均段落数: {comparison['avg_segments_gemini']:.2f}")
        print(f"   平均段落差异: {comparison['avg_segment_difference']:.2f}")
        print(f"   最大段落差异: {comparison['max_segment_difference']}")
        print(f"   段落数相同的视频: {comparison['videos_with_identical_segments']}")
        
        print("\n" + "="*60)

def main():
    comparator = VideoComparator()
    report = comparator.create_comparison_report()
    comparator.print_summary(report)

if __name__ == "__main__":
    main()