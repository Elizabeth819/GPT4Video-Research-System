#!/usr/bin/env python3

"""
Run GPT-5: Ghost Probing Detection on DADA-100 with Paper_Batch + Few-shot (Temperature=0)
- 从文件读取复杂Prompt与Few-shot示例
- 复用帧抽取与Azure OpenAI请求流程
- 输出中间结果、最终结果与性能指标到当前目录
"""

import os
import sys
import cv2
import json
import time
import base64
import re
import logging
import datetime
import traceback
import requests
import tqdm
import pandas as pd
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class GPT5GhostProbingRunner:
    def __init__(self,
                 output_dir: str,
                 complex_prompt_path: str,
                 fewshot_prompt_path: str,
                 chunk_size: int = 10,
                 frame_interval: int = 10,
                 frames_per_interval: int = 10):
        self.output_dir = output_dir
        self.complex_prompt_path = complex_prompt_path
        self.fewshot_prompt_path = fewshot_prompt_path
        self.chunk_size = chunk_size
        self.frame_interval = frame_interval
        self.frames_per_interval = frames_per_interval

        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()
        self._setup_openai()
        self._load_ground_truth()
        self._initialize_results()
        self._load_prompts()

    def _setup_logging(self) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_file = os.path.join(self.output_dir, f"run_gpt5_ghost_probing_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Run: GPT-5 Ghost Probing + Paper_Batch + Few-shot (temp=0) 开始")

    def _setup_openai(self) -> None:
        # 使用 Azure OpenAI 配置
        self.openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not self.openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY未设置")

        self.vision_endpoint = os.environ.get("AZURE_OPENAI_API_ENDPOINT", "")
        # 使用 GPT-5 部署
        self.vision_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_5", "gpt-5")
        if not self.vision_endpoint:
            raise ValueError("AZURE_OPENAI_API_ENDPOINT未设置")

        self.logger.info(f"Azure OpenAI Endpoint: {self.vision_endpoint}")
        self.logger.info(f"Deployment: {self.vision_deployment}")
        self.logger.info("Temperature: 1 (GPT-5 默认值)")

    def _load_ground_truth(self) -> None:
        # 尝试多个可能的标签文件位置
        possible_paths = [
            "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/supplementary_material/data/groundtruth_labels.csv",
            "/Users/wanmeng/repository/GPT4Video-cobra-auto/test_dada_videos/groundtruth_labels.csv",
            "/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/labels.csv"
        ]
        
        gt_path = None
        for path in possible_paths:
            if os.path.exists(path):
                gt_path = path
                break
                
        if not gt_path:
            raise FileNotFoundError(f"未找到ground truth标签文件，尝试了: {possible_paths}")
            
        # 加载CSV文件，确保正确解析分隔符
        self.ground_truth = pd.read_csv(gt_path, encoding='utf-8-sig')
        
        # 如果没有正确分割列，尝试手动分割
        if len(self.ground_truth.columns) == 1 and ',' in self.ground_truth.columns[0]:
            # 需要手动处理分隔符
            lines = []
            with open(gt_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    # 移除行尾逗号和空白
                    line = line.rstrip().rstrip(',')
                    parts = line.split(',')
                    if len(parts) >= 2:
                        lines.append({'video_id': parts[0], 'ground_truth_label': parts[1]})
            self.ground_truth = pd.DataFrame(lines)
        
        # 清理列名（去除BOM和空格）
        self.ground_truth.columns = self.ground_truth.columns.str.strip()
        
        # 只取前100个视频
        self.ground_truth = self.ground_truth.head(100)
                
        self.logger.info(f"加载ground truth标签: {len(self.ground_truth)} 条（限制100个视频），来源: {gt_path}")
        self.logger.info(f"标签文件列名: {list(self.ground_truth.columns)}")
        self.logger.info(f"前5行数据: {self.ground_truth.head()}")

    def _initialize_results(self) -> None:
        self.results = {
            "experiment_info": {
                "run_id": "gpt-5-experiment",
                "timestamp": self.timestamp,
                "video_count": 100,  # 限制处理100个视频
                "model": "GPT-5 (Azure)",
                "prompt_version": "Paper_Batch Complex + Few-shot (file-based)",
                "temperature": 1,  # GPT-5 只支持默认值
                "max_completion_tokens": 3000,
                "output_directory": self.output_dir,
                "note": "使用真正的 GPT-5 部署进行测试"
            },
            "detailed_results": []
        }

    def _load_prompts(self) -> None:
        if not os.path.exists(self.complex_prompt_path):
            raise FileNotFoundError(f"未找到复杂Prompt文件: {self.complex_prompt_path}")
        if not os.path.exists(self.fewshot_prompt_path):
            raise FileNotFoundError(f"未找到Few-shot示例文件: {self.fewshot_prompt_path}")

        with open(self.complex_prompt_path, 'r', encoding='utf-8') as f:
            complex_prompt_raw = f.read()
        with open(self.fewshot_prompt_path, 'r', encoding='utf-8') as f:
            fewshot_raw = f.read()

        # 仅保留复杂Prompt主体（去掉标题/指标说明等中文前言，若存在则直接使用全文）
        self.complex_prompt_template = complex_prompt_raw
        self.fewshot_examples_text = fewshot_raw
        self.logger.info("已加载复杂Prompt与Few-shot示例文本")

    def _extract_frames(self, video_path: str) -> list[str]:
        frames_dir = os.path.join(self.output_dir, "frames_temp")
        os.makedirs(frames_dir, exist_ok=True)
        frames: list[str] = []
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            num_intervals = max(1, int(duration / self.frame_interval))
            for interval_idx in range(num_intervals):
                start_time = interval_idx * self.frame_interval
                end_time = min((interval_idx + 1) * self.frame_interval, duration)
                for frame_idx in range(self.frames_per_interval):
                    if self.frames_per_interval == 1:
                        frame_time = start_time + (end_time - start_time) / 2
                    else:
                        frame_time = start_time + (frame_idx / (self.frames_per_interval - 1)) * (end_time - start_time)
                    if frame_time >= duration:
                        break
                    frame = clip.get_frame(frame_time)
                    frame_filename = f"frame_{interval_idx}_{frame_idx}_{frame_time:.1f}s.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frames.append(frame_path)
            clip.close()
            return frames
        except Exception as e:
            self.logger.error(f"帧提取失败 {video_path}: {str(e)}")
            return []

    def _build_prompt(self, video_id: str) -> str:
        # 将复杂Prompt参数化 + 附加Few-shot示例
        try:
            filled_complex = self.complex_prompt_template.format(
                frame_interval=self.frame_interval,
                frames_per_interval=self.frames_per_interval,
                video_id=video_id,
                segment_id_str="full_video"
            )
        except Exception:
            # 若格式化失败，直接使用原文，避免伪造
            filled_complex = self.complex_prompt_template

        prompt = (
            f"{filled_complex}\n\n"
            f"Few-shot Examples (from file):\n"
            f"{self.fewshot_examples_text}\n\n"
            f"Remember: Always and only return a single JSON object strictly following the schema."
        )
        return prompt

    def _send_request(self, prompt: str, images: list[str]) -> str | None:
        # Temperature=0 固定
        encoded_images: list[str] = []
        for image_path in images:
            try:
                with open(image_path, 'rb') as f:
                    encoded_images.append(base64.b64encode(f.read()).decode('utf-8'))
            except Exception as e:
                self.logger.error(f"图像编码失败 {image_path}: {str(e)}")

        if not encoded_images:
            return None

        content = [{"type": "text", "text": prompt}]
        for enc in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{enc}"}
            })

        data = {
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": 3000  # GPT-5 只支持默认 temperature=1
        }
        headers = {"Content-Type": "application/json", "api-key": self.openai_api_key}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.vision_endpoint}/openai/deployments/{self.vision_deployment}/chat/completions?api-version=2024-02-01",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                resp.raise_for_status()
                result = resp.json()
                return result['choices'][0]['message']['content']
            except requests.exceptions.Timeout:
                self.logger.warning(f"API请求超时，重试 {attempt+1}/{max_retries}")
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"API请求失败(尝试{attempt+1}/{max_retries}): {str(e)}")
                time.sleep(5)
        return None

    def _extract_key_actions(self, result_text: str) -> str:
        try:
            text = result_text.strip()
            if text.startswith('```json'):
                text = text.replace('```json', '').replace('```', '').strip()
            obj = json.loads(text)
            return str(obj.get('key_actions', '')).lower()
        except Exception:
            m = re.search(r'"key_actions":\s*"([^"]*)"', result_text)
            if m:
                return m.group(1).lower()
            return ""

    def _evaluate(self, key_actions: str, gt_label: str) -> str:
        has_ghost = "ghost probing" in key_actions
        gt_has_ghost = gt_label != "none"
        if has_ghost and gt_has_ghost:
            return "TP"
        if has_ghost and not gt_has_ghost:
            return "FP"
        if (not has_ghost) and gt_has_ghost:
            return "FN"
        return "TN"

    def run(self) -> None:
        video_ids = self.ground_truth['video_id'].tolist()
        self.logger.info(f"开始处理 {len(video_ids)} 个视频 (DADA-100 前100个)")
        for i, video_id in enumerate(tqdm.tqdm(video_ids, desc="处理视频")):
            try:
                video_path = f"/Users/wanmeng/repository/GPT4Video-cobra-auto/result/DADA-100-videos/{video_id}"
                if not os.path.exists(video_path):
                    self.logger.warning(f"视频不存在: {video_path}")
                    # 仍记录跳过以保持统计
                    self.results["detailed_results"].append({
                        "video_id": video_id,
                        "ground_truth": str(self._get_gt_label(video_id)),
                        "key_actions": "file_not_found",
                        "evaluation": "SKIP",
                        "raw_result": None
                    })
                    continue

                frames = self._extract_frames(video_path)
                if not frames:
                    self.logger.error(f"帧提取失败，跳过: {video_id}")
                    self._cleanup_frames(frames)
                    continue

                prompt = self._build_prompt(video_id)
                result = self._send_request(prompt, frames)
                self._cleanup_frames(frames)

                gt_label = self._get_gt_label(video_id)
                if result:
                    key_actions = self._extract_key_actions(result)
                    evaluation = self._evaluate(key_actions, gt_label)
                else:
                    key_actions = ""
                    evaluation = "ERROR"

                self.results["detailed_results"].append({
                    "video_id": video_id,
                    "ground_truth": gt_label,
                    "key_actions": key_actions,
                    "evaluation": evaluation,
                    "raw_result": result
                })

                # 每5个保存一次中间结果
                if (i + 1) % 5 == 0:
                    self._save_intermediate(i + 1)

            except Exception as e:
                self.logger.error(f"处理失败 {video_id}: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue

        self._save_final()
        self._save_metrics()

    def _get_gt_label(self, video_id: str) -> str:
        row = self.ground_truth[self.ground_truth['video_id'] == video_id]
        if row.empty:
            return "none"
        return str(row.iloc[0]['ground_truth_label'])

    def _cleanup_frames(self, frames: list[str]) -> None:
        for fp in frames or []:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass

    def _save_intermediate(self, processed_count: int) -> None:
        path = os.path.join(self.output_dir, f"gpt5_intermediate_{processed_count}videos_{self.timestamp}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"中间结果已保存: {path}")

    def _save_final(self) -> None:
        path = os.path.join(self.output_dir, f"gpt5_final_results_{self.timestamp}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"最终结果已保存: {path}")

    def _save_metrics(self) -> None:
        from collections import Counter
        evals = [r['evaluation'] for r in self.results["detailed_results"]]
        c = Counter(evals)
        tp, fp, tn, fn = c.get('TP', 0), c.get('FP', 0), c.get('TN', 0), c.get('FN', 0)
        errors = c.get('ERROR', 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "errors": errors,
            "total_videos": len(self.results["detailed_results"])
        }
        path = os.path.join(self.output_dir, f"gpt5_metrics_{self.timestamp}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        self.logger.info(f"性能指标已保存: {path}")
        self.logger.info(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Acc={accuracy:.3f}")


if __name__ == "__main__":
    # 输出目录
    output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/gpt-5"
    # Prompt文件
    complex_prompt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/prompts/paper_batch_original_complex_prompt.txt"
    fewshot_prompt_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/VIP/prompts/few-shot prompt/run8_gpt4o_fewshot_examples.txt"

    runner = GPT5GhostProbingRunner(
        output_dir=output_dir,
        complex_prompt_path=complex_prompt_path,
        fewshot_prompt_path=fewshot_prompt_path,
        chunk_size=10,
        frame_interval=10,
        frames_per_interval=10
    )
    print("开始: GPT-5 Ghost Probing (DADA-100) with Paper_Batch + Few-shot, temp=0")
    print(f"输出目录: {output_dir}")
    print(f"复杂Prompt: {complex_prompt_path}")
    print(f"Few-shot示例: {fewshot_prompt_path}")
    runner.run()
    print("完成: 结果与指标已写入输出目录") 