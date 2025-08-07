#!/usr/bin/env python3
import os
import sys

# Create the image-fewshot directory
output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot"
os.makedirs(output_dir, exist_ok=True)

print(f"✅ Created directory: {output_dir}")
print(f"Directory exists: {os.path.exists(output_dir)}")
print(f"Directory writable: {os.access(output_dir, os.W_OK)}")