#!/usr/bin/env python3
"""
Test Run 9: GPT-4o Ghost Probing Detection with Image Few-shot Learning (5 Videos Test)
测试脚本，验证图像few-shot集成是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run9_gpt4o_ghost_probing_image_fewshot_100videos import GPT4oRun9ImageFewshot
import logging

class GPT4oRun9Test(GPT4oRun9ImageFewshot):
    def run_test_experiment(self, num_videos=5):
        """运行测试实验（仅处理前N个视频）"""
        # 从ground truth文件中获取前N个视频
        test_videos = self.ground_truth['video_id'].tolist()[:num_videos]
        
        self.logger.info(f"开始Run 9测试实验，处理 {len(test_videos)} 个视频")
        self.logger.info(f"图像few-shot加载状态:")
        self.logger.info(f"  - Ghost Probing序列: {len(self.ghost_images)} 张")
        self.logger.info(f"  - Lower Barrier示例: {len(self.barrier_images)} 张")
        self.logger.info(f"  - Red Truck示例: {len(self.truck_images)} 张")
        self.logger.info(f"  - 总计: {len(self.few_shot_images)} 张")
        
        for i, video_id in enumerate(test_videos):
            try:
                self.logger.info(f"处理测试视频 {i+1}/{num_videos}: {video_id}")
                
                # 视频路径
                video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
                if not os.path.exists(video_path):
                    self.logger.warning(f"视频不存在: {video_path}")
                    continue
                
                # 获取ground truth
                gt_row = self.ground_truth[self.ground_truth['video_id'] == video_id]
                if gt_row.empty:
                    self.logger.warning(f"未找到ground truth: {video_id}")
                    continue
                
                ground_truth_label = gt_row.iloc[0]['ground_truth_label']
                
                # 分析视频（使用图像few-shot增强）
                self.logger.info(f"开始分析 {video_id}，预期标签: {ground_truth_label}")
                result = self.analyze_with_gpt4o(video_path, video_id)
                
                if result:
                    key_actions = self.extract_key_actions(result)
                    evaluation = self.evaluate_result(video_id, key_actions, ground_truth_label)
                    self.logger.info(f"✅ 分析完成: {video_id}")
                    self.logger.info(f"   GT={ground_truth_label}, 检测={key_actions}, 评估={evaluation}")
                else:
                    key_actions = ""
                    evaluation = "ERROR"
                    self.logger.error(f"❌ 分析失败: {video_id}")
                
                # 记录结果
                result_entry = {
                    "video_id": video_id,
                    "ground_truth": ground_truth_label,
                    "key_actions": key_actions,
                    "evaluation": evaluation,
                    "raw_result": result
                }
                
                self.results["detailed_results"].append(result_entry)
                
            except Exception as e:
                self.logger.error(f"处理测试视频失败 {video_id}: {str(e)}")
                continue
        
        # 保存测试结果
        self.save_test_results()
        self.generate_test_metrics()
        
    def save_test_results(self):
        """保存测试结果"""
        test_file = os.path.join(self.output_dir, f"run9_test_results_{self.timestamp}.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"测试结果已保存: {test_file}")
    
    def generate_test_metrics(self):
        """生成测试性能指标"""
        from collections import Counter
        
        evaluations = [r['evaluation'] for r in self.results["detailed_results"]]
        eval_counts = Counter(evaluations)
        
        tp = eval_counts.get('TP', 0)
        fp = eval_counts.get('FP', 0)
        tn = eval_counts.get('TN', 0)
        fn = eval_counts.get('FN', 0)
        errors = eval_counts.get('ERROR', 0)
        
        total_valid = tp + fp + tn + fn
        
        if total_valid > 0:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total_valid
        else:
            precision = recall = f1 = accuracy = 0
        
        self.logger.info("=== Run 9 Test Results (5 Videos) ===")
        self.logger.info(f"成功处理: {total_valid} 个视频")
        self.logger.info(f"错误数量: {errors} 个")
        if total_valid > 0:
            self.logger.info(f"精确度: {precision:.3f} ({precision*100:.1f}%)")
            self.logger.info(f"召回率: {recall:.3f} ({recall*100:.1f}%)")
            self.logger.info(f"F1分数: {f1:.3f} ({f1*100:.1f}%)")
            self.logger.info(f"准确率: {accuracy:.3f} ({accuracy*100:.1f}%)")
            self.logger.info(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        self.logger.info(f"使用图像few-shot示例: {len(self.few_shot_images)} 张")
        
        # 验证图像few-shot加载状态
        self.logger.info("=== Few-shot Images Status ===")
        for category, images in [("Ghost Probing", self.ghost_images), 
                                 ("Lower Barrier", self.barrier_images), 
                                 ("Red Truck", self.truck_images)]:
            self.logger.info(f"{category}: {len(images)} 张 - {list(images.keys())}")

if __name__ == "__main__":
    # 创建输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run9_gpt4o_ghost_probing_image_fewshot"
    
    # 运行测试实验
    test_experiment = GPT4oRun9Test(output_dir)
    test_experiment.run_test_experiment(num_videos=5)
    
    print("🧪 Run 9 测试实验完成!")
    print(f"📁 测试结果保存在: {output_dir}")
    print("✅ 如果测试成功，可以运行完整的100视频实验")