#!/usr/bin/env python3
"""
Test Claude 4 API Connection
"""

import json
import http.client
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_claude4_api():
    """测试Claude 4 API连接"""
    api_key = os.environ.get("CLAUDE_API_KEY", "")
    if not api_key:
        print("❌ CLAUDE_API_KEY未设置")
        return False
    
    try:
        # 构建简单测试请求
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, please respond with 'Claude 4 API working' to confirm you are Claude 4."
                }
            ],
            "max_tokens": 100,
            "temperature": 0
        })
        
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Host': 'globalai.vip',
            'Connection': 'keep-alive'
        }
        
        # 发送请求
        conn = http.client.HTTPSConnection("globalai.vip")
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        
        response_data = json.loads(data.decode("utf-8"))
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            content = response_data['choices'][0]['message']['content']
            print(f"✅ Claude 4 API连接成功!")
            print(f"🔮 模型: claude-sonnet-4-20250514")
            print(f"📝 响应: {content}")
            return True
        else:
            print(f"❌ API响应格式错误: {response_data}")
            return False
            
    except Exception as e:
        print(f"❌ Claude 4 API连接失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 测试 Claude 4 API 连接...")
    success = test_claude4_api()
    if success:
        print("🎯 准备开始 Run 20: Claude 4 Ghost Probing Detection 实验")
    else:
        print("⚠️ 请检查API配置后重试")