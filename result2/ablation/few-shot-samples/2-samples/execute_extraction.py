#!/usr/bin/env python3
import subprocess
import os
import sys

def run_extraction():
    # Change to the script directory
    script_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
    os.chdir(script_dir)
    
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Run the extraction script
    script_path = os.path.join(script_dir, "run_frame_extraction.py")
    
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=script_dir)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running extraction: {e}")

if __name__ == "__main__":
    run_extraction()