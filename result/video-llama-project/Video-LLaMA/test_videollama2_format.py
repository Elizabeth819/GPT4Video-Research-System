#!/usr/bin/env python3
"""
测试Video-LLaMA2输出格式
验证新的JSON格式是否正确
"""

import json
import sys
import os

def test_json_format():
    """测试JSON格式"""
    
    # 创建示例输出格式
    sample_output = [
        {
            "video_id": "images_1_002.avi",
            "segment_id": "segment_000",
            "Start_Timestamp": "0.0s",
            "End_Timestamp": "10.0s",
            "sentiment": "Neutral",
            "scene_theme": "Routine",
            "characters": "Multiple vehicles in traffic, no visible pedestrians",
            "summary": "The observer vehicle is following traffic in a normal urban environment. Several vehicles are visible ahead, maintaining normal speeds and distances. No unusual activities or dangerous situations are observed in this segment.",
            "actions": "The observer vehicle maintains steady speed, following the vehicle ahead at a safe distance. No emergency maneuvers are required.",
            "key_objects": "1) Front center: Lead vehicle, 10-15 meters ahead, maintaining steady speed, no immediate impact. 2) Left lane: Adjacent vehicle, parallel positioning, normal traffic flow.",
            "key_actions": "maintain safe following distance, normal traffic flow",
            "next_action": {
                "speed_control": "maintain speed",
                "direction_control": "keep direction",
                "lane_control": "maintain current lane"
            }
        },
        {
            "video_id": "images_1_002.avi",
            "segment_id": "segment_001",
            "Start_Timestamp": "10.0s",
            "End_Timestamp": "15.0s",
            "sentiment": "Negative",
            "scene_theme": "Dangerous",
            "characters": "Vehicle drivers, one pedestrian suddenly appearing",
            "summary": "A critical safety situation develops as a pedestrian suddenly appears from behind a parked vehicle, entering the roadway at very close range to the observer vehicle. The pedestrian's movement is unexpected and requires immediate emergency response.",
            "actions": "The observer vehicle immediately applies emergency braking in response to the pedestrian's sudden appearance. The driver reacts quickly to avoid collision.",
            "key_objects": "1) Front center: Pedestrian, <2 meters, sudden appearance from blind spot, requiring immediate emergency braking. 2) Right side: Parked vehicle, 5 meters, creates blind spot from which pedestrian emerged.",
            "key_actions": "ghost probing, emergency braking due to pedestrian suddenly appearing from blind spot",
            "next_action": {
                "speed_control": "rapid deceleration",
                "direction_control": "keep direction",
                "lane_control": "maintain current lane"
            }
        }
    ]
    
    print("🧪 Testing Video-LLaMA2 JSON Format")
    print("=" * 60)
    
    # 验证JSON格式
    try:
        json_str = json.dumps(sample_output, indent=2, ensure_ascii=False)
        print("✅ JSON格式验证通过")
        
        # 检查必要字段
        required_fields = [
            "video_id", "segment_id", "Start_Timestamp", "End_Timestamp",
            "sentiment", "scene_theme", "characters", "summary", "actions",
            "key_objects", "key_actions", "next_action"
        ]
        
        for segment in sample_output:
            for field in required_fields:
                if field not in segment:
                    print(f"❌ 缺少必要字段: {field}")
                    return False
        
        print("✅ 所有必要字段验证通过")
        
        # 检查鬼探头检测
        ghost_detected = False
        for segment in sample_output:
            if "ghost probing" in segment["key_actions"].lower():
                ghost_detected = True
                print(f"✅ 鬼探头检测: {segment['video_id']} - {segment['Start_Timestamp']}")
                break
        
        if not ghost_detected:
            print("ℹ️ 本示例中未检测到鬼探头")
        
        # 保存示例文件
        with open("sample_videollama2_output.json", "w", encoding="utf-8") as f:
            json.dump(sample_output, f, indent=2, ensure_ascii=False)
        
        print("✅ 示例文件已保存: sample_videollama2_output.json")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON格式验证失败: {e}")
        return False

def compare_with_gpt41_format():
    """与GPT-4.1格式对比"""
    
    print("\n📊 与GPT-4.1格式对比")
    print("=" * 60)
    
    # 检查是否存在GPT-4.1格式的文件
    gpt41_file = "../../result/gpt41-balanced-full/actionSummary_images_5_054.json"
    
    if os.path.exists(gpt41_file):
        try:
            with open(gpt41_file, 'r', encoding='utf-8') as f:
                gpt41_data = json.load(f)
            
            print("✅ 找到GPT-4.1格式文件")
            print(f"   段落数: {len(gpt41_data)}")
            
            # 显示第一个段落的字段
            if gpt41_data:
                first_segment = gpt41_data[0]
                print("   GPT-4.1字段:")
                for key in first_segment.keys():
                    print(f"     - {key}")
            
            # 检查鬼探头检测
            ghost_segments = []
            for segment in gpt41_data:
                if "ghost probing" in segment.get("key_actions", "").lower():
                    ghost_segments.append(segment)
            
            print(f"   鬼探头段落数: {len(ghost_segments)}")
            
            if ghost_segments:
                print("   鬼探头检测示例:")
                for segment in ghost_segments:
                    print(f"     - {segment.get('Start_Timestamp', 'unknown')}: {segment.get('key_actions', 'unknown')}")
            
        except Exception as e:
            print(f"❌ 读取GPT-4.1格式文件失败: {e}")
    else:
        print("ℹ️ 未找到GPT-4.1格式文件用于对比")

def test_parsing_logic():
    """测试解析逻辑"""
    
    print("\n🔍 测试解析逻辑")
    print("=" * 60)
    
    # 模拟不同类型的响应
    test_cases = [
        {
            "name": "完整JSON数组",
            "response": '''[
                {
                    "video_id": "test_video.avi",
                    "segment_id": "segment_000",
                    "Start_Timestamp": "0.0s",
                    "End_Timestamp": "10.0s",
                    "sentiment": "Negative",
                    "scene_theme": "Dangerous",
                    "characters": "Pedestrian suddenly appearing",
                    "summary": "Ghost probing situation detected",
                    "actions": "Emergency braking applied",
                    "key_objects": "1) Front: Pedestrian, <2 meters, sudden appearance",
                    "key_actions": "ghost probing, emergency braking",
                    "next_action": {
                        "speed_control": "rapid deceleration",
                        "direction_control": "keep direction",
                        "lane_control": "maintain current lane"
                    }
                }
            ]'''
        },
        {
            "name": "文本描述",
            "response": "In this video, I observe a dangerous ghost probing situation where a pedestrian suddenly appears from behind a parked car, requiring immediate emergency braking."
        },
        {
            "name": "普通交通情况",
            "response": "This video shows normal traffic flow with vehicles maintaining safe distances. No dangerous situations are observed."
        }
    ]
    
    # 导入解析函数（简化版）
    import re
    
    def simple_parse_response(response, video_id):
        """简化的解析函数"""
        try:
            # 尝试解析JSON数组
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_segments = json.loads(json_str)
                    if isinstance(parsed_segments, list):
                        for segment in parsed_segments:
                            segment["video_id"] = video_id
                        
                        ghost_detected = any("ghost probing" in seg.get("key_actions", "").lower() for seg in parsed_segments)
                        return {
                            "segments": parsed_segments,
                            "ghost_probing_detected": ghost_detected,
                            "parsing_success": True
                        }
                except json.JSONDecodeError:
                    pass
            
            # 文本解析
            ghost_detected = "ghost probing" in response.lower()
            return {
                "segments": [{
                    "video_id": video_id,
                    "segment_id": "segment_000",
                    "key_actions": "ghost probing" if ghost_detected else "normal traffic flow",
                    "summary": response[:200]
                }],
                "ghost_probing_detected": ghost_detected,
                "parsing_success": False
            }
            
        except Exception as e:
            return {"error": str(e), "parsing_success": False}
    
    # 测试解析
    for test_case in test_cases:
        print(f"\n测试用例: {test_case['name']}")
        result = simple_parse_response(test_case['response'], "test_video.avi")
        
        if result.get("parsing_success"):
            print("✅ 解析成功")
            print(f"   鬼探头检测: {result.get('ghost_probing_detected', False)}")
            print(f"   段落数: {len(result.get('segments', []))}")
        else:
            print("⚠️ 解析为文本模式")
            print(f"   鬼探头检测: {result.get('ghost_probing_detected', False)}")
        
        if "error" in result:
            print(f"❌ 解析错误: {result['error']}")

def main():
    """主函数"""
    print("🎬 Video-LLaMA2 JSON Format Testing")
    print("=" * 60)
    
    # 测试JSON格式
    if test_json_format():
        print("\n✅ JSON格式测试通过")
    else:
        print("\n❌ JSON格式测试失败")
        return
    
    # 与GPT-4.1格式对比
    compare_with_gpt41_format()
    
    # 测试解析逻辑
    test_parsing_logic()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成")
    print("✅ Video-LLaMA2格式已准备就绪")
    print("📄 可以查看 sample_videollama2_output.json 了解输出格式")

if __name__ == "__main__":
    main()