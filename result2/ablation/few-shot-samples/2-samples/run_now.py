import subprocess
import sys
import os

# Change to the directory with the script
os.chdir('/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples')

# Run the manual extraction script
try:
    result = subprocess.run([sys.executable, 'manual_extraction.py'], 
                          capture_output=True, text=True, timeout=300)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("Script timed out after 5 minutes")
except Exception as e:
    print(f"Error running script: {e}")

# Also try to list any generated files
output_dir = "/Users/wanmeng/repository/GPT4Video-cobra-auto/result2/ablation/few-shot-samples/2-samples"
ghost_files = []
try:
    all_files = os.listdir(output_dir)
    ghost_files = [f for f in all_files if f.startswith("ghost_probing_") and f.endswith(".jpg")]
    ghost_files.sort()
except Exception as e:
    print(f"Error listing files: {e}")

if ghost_files:
    print(f"\nGenerated ghost probing files:")
    for f in ghost_files:
        filepath = os.path.join(output_dir, f)
        try:
            size = os.path.getsize(filepath)
            print(f"  {f} ({size:,} bytes)")
        except:
            print(f"  {f} (size unknown)")
else:
    print("\nNo ghost probing files found in output directory")