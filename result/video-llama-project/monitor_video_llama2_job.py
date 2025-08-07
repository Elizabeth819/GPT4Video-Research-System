#!/usr/bin/env python3
"""
Monitor Video-LLaMA-2 Azure ML job progress and results
"""

import subprocess
import json
import time
from datetime import datetime

def check_job_status(job_name):
    """Check the status of an Azure ML job"""
    cmd = f"""
    az ml job show --name {job_name} \\
        --resource-group video-llama2-ghost-probing-rg \\
        --workspace-name video-llama2-ghost-probing-ws \\
        --query "status" --output tsv
    """
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error checking job status: {e}")
        return "Error"

def get_job_details(job_name):
    """Get detailed job information"""
    cmd = f"""
    az ml job show --name {job_name} \\
        --resource-group video-llama2-ghost-probing-rg \\
        --workspace-name video-llama2-ghost-probing-ws
    """
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return None
    except Exception as e:
        print(f"Error getting job details: {e}")
        return None

def get_job_logs(job_name):
    """Get job logs"""
    cmd = f"""
    az ml job stream --name {job_name} \\
        --resource-group video-llama2-ghost-probing-rg \\
        --workspace-name video-llama2-ghost-probing-ws
    """
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error getting logs: {result.stderr}"
    except Exception as e:
        return f"Error getting logs: {e}"

def monitor_job(job_name, check_interval=30):
    """Monitor job progress with regular status checks"""
    
    print(f"🔍 Starting monitoring for job: {job_name}")
    print(f"📊 Check interval: {check_interval} seconds")
    print(f"🌐 Azure ML Studio: https://ml.azure.com/runs/{job_name}")
    print("=" * 60)
    
    start_time = datetime.now()
    last_status = None
    
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        
        # Check job status
        status = check_job_status(job_name)
        
        if status != last_status:
            print(f"\n🔄 [{current_time.strftime('%H:%M:%S')}] Status changed: {last_status} → {status}")
            last_status = status
            
            # Get additional details on status change
            if status in ["Running", "Completed", "Failed", "Canceled"]:
                details = get_job_details(job_name)
                if details:
                    print(f"📋 Job Details:")
                    print(f"   • Display Name: {details.get('display_name', 'N/A')}")
                    print(f"   • Experiment: {details.get('experiment_name', 'N/A')}")
                    print(f"   • Compute: {details.get('compute', 'N/A')}")
                    
                    # Show timing information
                    props = details.get('properties', {})
                    if 'StartTimeUtc' in props:
                        print(f"   • Start Time: {props['StartTimeUtc']}")
                    if 'EndTimeUtc' in props:
                        print(f"   • End Time: {props['EndTimeUtc']}")
        
        # Check if job is complete
        if status in ["Completed", "Failed", "Canceled"]:
            print(f"\n🎯 Job {status.lower()}!")
            print(f"⏱️  Total runtime: {elapsed}")
            
            if status == "Completed":
                print("✅ Job completed successfully!")
                print("\n📄 Final logs:")
                logs = get_job_logs(job_name)
                if logs:
                    print(logs[-2000:])  # Show last 2000 characters
                else:
                    print("No logs available")
            elif status == "Failed":
                print("❌ Job failed!")
                print("\n📄 Error logs:")
                logs = get_job_logs(job_name)
                if logs:
                    print(logs[-2000:])  # Show last 2000 characters
                else:
                    print("No logs available")
            
            break
        
        # Show progress
        print(f"⏳ [{current_time.strftime('%H:%M:%S')}] Status: {status} | Elapsed: {elapsed}")
        
        # Wait before next check
        time.sleep(check_interval)

def main():
    """Main monitoring function"""
    # The job name from our submission
    job_name = "stoic_chain_1d015wswm1"
    
    print("🎯 Video-LLaMA-2 Job Monitor")
    print("=" * 50)
    print(f"Job Name: {job_name}")
    print(f"Resource Group: video-llama2-ghost-probing-rg")
    print(f"Workspace: video-llama2-ghost-probing-ws")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        monitor_job(job_name, check_interval=30)
        
        print(f"\n📊 Monitoring Summary:")
        print(f"   • Job: {job_name}")
        print(f"   • Final Status: {check_job_status(job_name)}")
        print(f"   • Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n🌐 View full results in Azure ML Studio:")
        print(f"   https://ml.azure.com/runs/{job_name}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Monitoring stopped by user")
        print(f"   Current status: {check_job_status(job_name)}")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    main()