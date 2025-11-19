#!/usr/bin/env python3

"""
Real GPT-5 Ghost Probing Test Script
使用正确的 GPT-5 配置进行视频分析测试
"""

import os
import sys
import json
import time
import requests
import logging
import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class GPT5TestRunner:
    def __init__(self):
        self.setup_logging()
        self.setup_gpt5_config()
        
    def setup_logging(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        log_file = f"gpt5_test_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== GPT-5 Ghost Probing Test 开始 ===")
        
    def setup_gpt5_config(self):
        """配置 GPT-5 API 访问"""
        # 方案1: 使用 OpenAI GPT-5 API （如果可用）
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = "https://api.openai.com/v1"
        
        # 方案2: 使用 Azure OpenAI GPT-5 部署（如果配置了）
        self.azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_API_ENDPOINT")
        self.azure_gpt5_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME_5", "gpt-5")
        
        self.logger.info(f"OpenAI API Key: {'已设置' if self.openai_api_key else '未设置'}")
        self.logger.info(f"Azure API Key: {'已设置' if self.azure_api_key else '未设置'}")
        self.logger.info(f"Azure GPT-5 Deployment: {self.azure_gpt5_deployment}")
        
    def test_gpt5_availability(self):
        """测试 GPT-5 可用性"""
        self.logger.info("测试 GPT-5 API 可用性...")
        
        test_prompt = "Hello, this is a test message. Please respond with 'GPT-5 is working'."
        
        # 先测试 OpenAI GPT-5
        if self.openai_api_key:
            result = self.test_openai_gpt5(test_prompt)
            if result:
                return "openai", result
        
        # 然后测试 Azure GPT-5
        if self.azure_api_key and self.azure_endpoint:
            result = self.test_azure_gpt5(test_prompt)
            if result:
                return "azure", result
                
        return None, "GPT-5 不可用"
    
    def test_openai_gpt5(self, prompt):
        """测试 OpenAI GPT-5"""
        try:
            self.logger.info("测试 OpenAI GPT-5...")
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-5",  # 或者 "gpt-5-turbo" 等
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 100  # GPT-5 只支持默认 temperature=1
            }
            
            response = requests.post(
                f"{self.openai_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.logger.info(f"OpenAI GPT-5 响应: {content}")
                return content
            else:
                self.logger.error(f"OpenAI GPT-5 错误: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"OpenAI GPT-5 测试失败: {str(e)}")
            return None
    
    def test_azure_gpt5(self, prompt):
        """测试 Azure GPT-5"""
        try:
            self.logger.info("测试 Azure GPT-5...")
            headers = {
                "api-key": self.azure_api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 100  # GPT-5 只支持默认 temperature=1
            }
            
            url = f"{self.azure_endpoint}/openai/deployments/{self.azure_gpt5_deployment}/chat/completions?api-version=2024-02-01"
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.logger.info(f"Azure GPT-5 响应: {content}")
                return content
            else:
                self.logger.error(f"Azure GPT-5 错误: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Azure GPT-5 测试失败: {str(e)}")
            return None
    
    def run_ghost_probing_test(self):
        """运行鬼探头检测测试"""
        self.logger.info("开始 GPT-5 Ghost Probing 测试...")
        
        # 首先测试 GPT-5 可用性
        provider, response = self.test_gpt5_availability()
        
        if not provider:
            self.logger.error("GPT-5 不可用，无法进行测试")
            return False
            
        self.logger.info(f"使用 {provider} GPT-5 进行测试")
        
        # 加载测试用的 prompt 和示例
        ghost_probing_prompt = self.load_ghost_probing_prompt()
        
        # 进行简单的文本测试
        test_scenario = """
        请分析这个驾驶场景：一辆车正常行驶在城市道路上，突然从右侧停车的后面窜出一个骑自行车的人，
        直接横穿到车道上。请判断这是否为"ghost probing"行为。
        """
        
        if provider == "openai":
            result = self.test_openai_gpt5(ghost_probing_prompt + test_scenario)
        else:
            result = self.test_azure_gpt5(ghost_probing_prompt + test_scenario)
        
        if result:
            self.logger.info("GPT-5 Ghost Probing 测试成功")
            self.logger.info(f"测试结果: {result}")
            
            # 保存测试结果
            test_result = {
                "timestamp": self.timestamp,
                "provider": provider,
                "model": "GPT-5",
                "test_prompt": ghost_probing_prompt + test_scenario,
                "response": result,
                "status": "success"
            }
            
            with open(f"gpt5_test_result_{self.timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(test_result, f, ensure_ascii=False, indent=2)
                
            return True
        else:
            self.logger.error("GPT-5 Ghost Probing 测试失败")
            return False
    
    def load_ghost_probing_prompt(self):
        """加载 Ghost Probing 检测的提示词"""
        return """
你是一个专业的驾驶安全分析专家。请分析给定的驾驶场景，重点识别是否存在"Ghost Probing"（鬼探头）行为。

Ghost Probing 定义：
1. 行人或车辆从视觉障碍物后面突然出现
2. 给驾驶员的反应时间极短
3. 通常从停车、建筑物、树木等遮挡物后面窜出

请分析以下场景并回答：
1. 是否存在 Ghost Probing 行为？
2. 如果是，请说明具体的危险程度
3. 建议的应对措施

场景描述：
"""

def main():
    print("=== GPT-5 Ghost Probing 测试 ===")
    print("正在测试 GPT-5 API 可用性和 Ghost Probing 检测功能...")
    
    runner = GPT5TestRunner()
    success = runner.run_ghost_probing_test()
    
    if success:
        print("✅ GPT-5 测试完成！")
        print(f"📄 查看详细日志: gpt5_test_{runner.timestamp}.log")
        print(f"📊 查看测试结果: gpt5_test_result_{runner.timestamp}.json")
    else:
        print("❌ GPT-5 测试失败")
        print("请检查 API 配置和网络连接")
        
        # 输出配置建议
        print("\n配置建议：")
        print("1. 设置 OPENAI_API_KEY 环境变量（如果使用 OpenAI）")
        print("2. 或设置 Azure OpenAI 相关环境变量：")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_API_ENDPOINT") 
        print("   - AZURE_OPENAI_DEPLOYMENT_NAME_5")

if __name__ == "__main__":
    main()