#!/usr/bin/env python3
"""
Few-shot Examples for Ablation Study
从JSON文件加载few-shot样本，用于消融实验
"""

import json
import os

def load_examples_data():
    """加载few-shot样本数据"""
    data_file = os.path.join(os.path.dirname(__file__), 'fewshot_examples_data_realistic.json')
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_fewshot_examples(num_samples=3):
    """
    获取指定数量的few-shot样本
    Args:
        num_samples: 1, 2, 3, 或 5
    Returns:
        str: 格式化的few-shot样本文本
    """
    
    # 加载样本数据
    data = load_examples_data()
    examples = data['examples'][:num_samples]
    
    # 格式化为prompt文本
    fewshot_text = "**Few-shot Examples:**\n\n"
    
    for i, example in enumerate(examples, 1):
        fewshot_text += f"Example {i} - {example['name']}:\n"
        fewshot_text += "{\n"
        
        example_data = example['data']
        fewshot_text += f'    "video_id": "{example_data["video_id"]}",\n'
        fewshot_text += f'    "segment_id": "{example_data["segment_id"]}",\n'
        fewshot_text += f'    "Start_Timestamp": "{example_data["Start_Timestamp"]}",\n'
        fewshot_text += f'    "End_Timestamp": "{example_data["End_Timestamp"]}",\n'
        fewshot_text += f'    "sentiment": "{example_data["sentiment"]}",\n'
        fewshot_text += f'    "scene_theme": "{example_data["scene_theme"]}",\n'
        fewshot_text += f'    "characters": "{example_data["characters"]}",\n'
        fewshot_text += f'    "summary": "{example_data["summary"]}",\n'
        fewshot_text += f'    "actions": "{example_data["actions"]}",\n'
        fewshot_text += f'    "key_objects": "{example_data["key_objects"]}",\n'
        fewshot_text += f'    "key_actions": "{example_data["key_actions"]}",\n'
        fewshot_text += f'    "next_action": {{\n'
        fewshot_text += f'        "speed_control": "{example_data["next_action"]["speed_control"]}",\n'
        fewshot_text += f'        "direction_control": "{example_data["next_action"]["direction_control"]}",\n'
        fewshot_text += f'        "lane_control": "{example_data["next_action"]["lane_control"]}"\n'
        fewshot_text += f'    }}\n'
        fewshot_text += "}\n\n"
    
    return fewshot_text

def get_example_info(num_samples=3):
    """
    获取样本信息摘要
    Args:
        num_samples: 样本数量
    Returns:
        list: 样本信息列表
    """
    data = load_examples_data()
    examples = data['examples'][:num_samples]
    
    info = []
    for i, example in enumerate(examples, 1):
        info.append({
            'id': i,
            'name': example['name'],
            'type': example['type'],
            'description': example['description'],
            'key_actions': example['data']['key_actions']
        })
    
    return info

def show_examples_summary():
    """显示所有样本摘要"""
    data = load_examples_data()
    
    print("=== Few-shot样本库总览 ===")
    print(f"总共{len(data['examples'])}个样本:\n")
    
    for i, example in enumerate(data['examples'], 1):
        print(f"📌 Example {i}: {example['name']}")
        print(f"   类型: {example['type']} ({'positive' if example['type'] == 'positive' else 'negative'}样本)")
        print(f"   描述: {example['description']}")
        print(f"   关键动作: {example['data']['key_actions']}")
        print(f"   场景主题: {example['data']['scene_theme']}")
        print()

if __name__ == "__main__":
    # 显示样本库总览
    show_examples_summary()
    
    # 测试不同数量的样本生成
    print("=== 消融实验配置测试 ===")
    for num in [1, 2, 3, 5]:
        info = get_example_info(num)
        print(f"\n🧪 {num}样本实验配置:")
        for sample_info in info:
            print(f"  - Example {sample_info['id']}: {sample_info['name']} ({sample_info['type']})")
        
        # 显示类型分布
        positive_count = sum(1 for x in info if x['type'] == 'positive')
        negative_count = sum(1 for x in info if x['type'] == 'negative')
        print(f"  📊 样本分布: {positive_count}个positive + {negative_count}个negative")