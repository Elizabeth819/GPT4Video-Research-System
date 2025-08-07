#!/usr/bin/env python3
"""
Simple runner to extract ghost probing frames
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to the script directory
    script_dir = Path(__file__).parent
    script_path = script_dir / "extract_ghost_probing_frames.py"
    
    print(f"Running: {script_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    try:
        # Run the extraction script
        result = subprocess.run([sys.executable, str(script_path)], 
                              cwd=str(script_dir),
                              capture_output=True, 
                              text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running script: {e}")

if __name__ == "__main__":
    main()