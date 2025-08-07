#!/usr/bin/env python3
"""
测试Azure ML环境的基础依赖和LLaVA模型加载
"""

import sys
import os

def test_basic_imports():
    """测试基础包导入"""
    print("🔍 测试基础包导入...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers导入失败: {e}")
        return False
    
    try:
        import decord
        print(f"✅ Decord {decord.__version__}")
    except ImportError as e:
        print(f"❌ Decord导入失败: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV导入失败: {e}")
        return False
    
    return True

def test_llava_model():
    """测试LLaVA模型加载"""
    print("\n🤖 测试LLaVA模型组件...")
    
    try:
        # 添加LLaVA-NeXT到路径
        llava_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLaVA-NeXT')
        sys.path.append(llava_path)
        
        # 测试导入LLaVA组件
        from llava.model.builder import load_pretrained_model
        print("✅ LLaVA builder导入成功")
        
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        print("✅ LLaVA mm_utils导入成功")
        
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        print("✅ LLaVA constants导入成功")
        
        from llava.conversation import conv_templates
        print("✅ LLaVA conversation导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ LLaVA组件导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ LLaVA测试失败: {e}")
        return False

def test_data_access():
    """测试数据访问"""
    print("\n📁 测试数据访问...")
    
    video_folder = "./inputs/video_data"
    if os.path.exists(video_folder):
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.avi')]
        print(f"✅ 找到 {len(video_files)} 个视频文件")
        if video_files:
            print(f"   示例: {video_files[0]}")
        return len(video_files) > 0
    else:
        print(f"❌ 视频文件夹不存在: {video_folder}")
        return False

def main():
    """主测试函数"""
    print("🧪 Azure ML LLaVA环境测试")
    print("=" * 50)
    
    success = True
    
    # 测试基础导入
    success &= test_basic_imports()
    
    # 测试数据访问
    success &= test_data_access()
    
    # 测试LLaVA模型 (如果基础测试通过)
    if success:
        success &= test_llava_model()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！环境配置正确")
        sys.exit(0)
    else:
        print("❌ 测试失败，需要修复环境问题")
        sys.exit(1)

if __name__ == "__main__":
    main()