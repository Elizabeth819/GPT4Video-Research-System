#!/usr/bin/env python3
"""
Submit Video-LLaMA-2 inference job to Azure ML
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def upload_dada_videos():
    """Upload DADA videos to Azure ML datastore"""
    print("📤 Uploading DADA videos to Azure ML...")
    
    dada_path = "/Users/wanmeng/repository/GPT4Video-cobra-auto/DADA-2000-videos"
    
    # Create data asset from DADA videos
    upload_cmd = f"""
    az ml data create --name dada-2000-videos \\
        --version 1 \\
        --type uri_folder \\
        --path {dada_path} \\
        --description "DADA-2000 driving videos for ghost probing detection" \\
        --resource-group video-llama2-ghost-probing-rg \\
        --workspace-name video-llama2-ghost-probing-ws
    """
    
    try:
        result = subprocess.run(upload_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Successfully uploaded DADA videos")
            return True
        else:
            print(f"❌ Upload failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def submit_inference_job():
    """Submit Video-LLaMA-2 inference job to Azure ML"""
    print("🚀 Submitting Video-LLaMA-2 inference job...")
    
    # Submit job using Azure ML CLI
    submit_cmd = """
    az ml job create --file video_llama2_azure_job.yml \\
        --resource-group video-llama2-ghost-probing-rg \\
        --workspace-name video-llama2-ghost-probing-ws \\
        --stream
    """
    
    try:
        print("Executing:", submit_cmd)
        result = subprocess.run(submit_cmd, shell=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully submitted inference job")
            return True
        else:
            print(f"❌ Job submission failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Job submission error: {e}")
        return False

def check_job_status():
    """Check the status of running jobs"""
    print("📊 Checking job status...")
    
    status_cmd = """
    az ml job list \\
        --resource-group video-llama2-ghost-probing-rg \\
        --workspace-name video-llama2-ghost-probing-ws \\
        --max-results 5
    """
    
    try:
        result = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("📋 Recent jobs:")
            print(result.stdout)
        else:
            print(f"❌ Failed to check status: {result.stderr}")
    except Exception as e:
        print(f"❌ Status check error: {e}")

def main():
    """Main function to submit Video-LLaMA-2 job"""
    print("🎯 Video-LLaMA-2 Azure ML Job Submission")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check required files
    required_files = [
        "azure_video_llama2_inference.py",
        "video_llama2_azure_job.yml", 
        "video_llama2_azure_env.yml"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return
    
    print("✅ All required files found")
    
    # Step 1: Upload DADA videos (if not already uploaded)
    print("\n📤 Step 1: Upload DADA videos")
    if upload_dada_videos():
        print("✅ DADA videos ready")
    else:
        print("⚠️  Video upload failed, but continuing (videos might already exist)")
    
    # Step 2: Submit inference job
    print("\n🚀 Step 2: Submit inference job")
    if submit_inference_job():
        print("✅ Job submitted successfully")
        
        # Step 3: Check job status
        print("\n📊 Step 3: Check job status")
        check_job_status()
        
        print(f"""
        
🎯 Video-LLaMA-2 Inference Job Submitted!

📋 What happens next:
   1. Azure ML will provision V100 compute cluster
   2. Download Video-LLaMA-2-7B-Finetuned model
   3. Process 20 DADA videos for ghost probing detection
   4. Generate results with confidence scores
   5. Save results to Azure ML outputs

🌐 Monitor job progress:
   • Azure ML Studio: https://ml.azure.com
   • Check logs and outputs in the job details
   • Expected runtime: 2-4 hours

💰 Estimated cost: $50-100 for V100 cluster usage
        """)
        
    else:
        print("❌ Job submission failed")

if __name__ == "__main__":
    main()