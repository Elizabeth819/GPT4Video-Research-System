#!/usr/bin/env python3
"""测试LLaVA模型是否可用"""

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llava_availability():
    """测试LLaVA模型可用性"""
    
    print("🔧 测试LLaVA模型可用性...")
    
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        print("✅ LLaVA transformers可导入")
        
        # 测试加载小模型
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        
        print(f"📥 尝试加载模型: {model_id}")
        
        processor = LlavaNextProcessor.from_pretrained(model_id)
        print("✅ Processor加载成功")
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model加载成功")
        
        print("🎉 真正的LLaVA模型可用!")
        return True
        
    except Exception as e:
        print(f"❌ LLaVA模型不可用: {e}")
        return False

def test_clip_fallback():
    """测试CLIP回退"""
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        print("✅ CLIP回退模式可用")
        return True
        
    except Exception as e:
        print(f"❌ CLIP也不可用: {e}")
        return False

if __name__ == "__main__":
    print("🚀 模型可用性测试")
    print("=" * 50)
    
    llava_ok = test_llava_availability()
    if not llava_ok:
        print("\n🔄 测试CLIP回退...")
        clip_ok = test_clip_fallback()
        
        if not clip_ok:
            print("❌ 所有模型都不可用")
        else:
            print("⚠️  将使用CLIP回退模式")
    
    print("=" * 50)