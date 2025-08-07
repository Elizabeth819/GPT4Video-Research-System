#!/usr/bin/env python3
import subprocess
import sys
import os

# Change to the correct directory
os.chdir('/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/image-fewshot')

# Execute the extraction script
try:
    result = subprocess.run([sys.executable, 'test_extraction.py'], 
                          capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    print(f"\nReturn code: {result.returncode}")
    
except Exception as e:
    print(f"Error executing script: {e}")