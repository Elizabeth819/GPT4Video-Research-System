# Azure ML Job Failure Investigation Report

## Job Information
- **Job ID**: stoic_kitchen_mz0lhlfq29
- **Job Name**: llava-ghost-probing-fixed-0721_144428
- **Status**: Failed
- **Start Time**: 2025-07-21T06:45:08
- **End Time**: 2025-07-21T06:45:33
- **Duration**: ~25 seconds (very short, indicating early failure)
- **Compute**: llava-a100-low-priority
- **Environment**: AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10

## Root Cause Analysis

### Primary Issue: Data Access Failure
The job failed during the data mounting phase, specifically when trying to access the input data path:

```
azureml://datastores/workspaceblobstore/paths/DADA-100-videos/
```

**Error Details:**
- **Error Code**: `ScriptExecution.StreamAccess.NotFound`
- **Error Message**: "The requested stream was not found. Please make sure the request uri is correct."
- **Category**: UserError
- **Target**: `UriMountSession:INPUT_video_data`

### Detailed Error Analysis

1. **Data Capability Failure**:
   ```
   Failed to mount URI azureml://subscriptions/.../datastores/workspaceblobstore/paths/DADA-100-videos/
   due to exception of type ExecutionError with message:
   Error Code: ScriptExecution.StreamAccess.NotFound
   Native Error: error in streaming from input data sources
   StreamError(NotFound) => stream not found
   ```

2. **Job Configuration**:
   - The job was configured to read from `./inputs/video_data` (mounted from the DADA-100-videos path)
   - The script tried to execute: `python llava_ghost_probing_batch.py --video-folder ./inputs/video_data`
   - However, the data mounting failed before the script could execute

3. **Authentication Issue**:
   - Storage account access denied: "Key based authentication is not permitted"
   - This suggests the storage account requires managed identity or SAS token authentication

## Technical Issues Identified

### 1. Missing Data Asset
The path `DADA-100-videos` does not exist in the workspaceblobstore or is not accessible with current authentication.

### 2. Authentication Problem
- Storage account `llavaworstoragea223ed2d6` has key-based authentication disabled
- The job needs to use Azure AD authentication or proper credentials

### 3. Job Configuration Mismatch
- The job expects to find `llava_ghost_probing_batch.py` in the uploaded code
- The command references input data that wasn't successfully mounted

### 4. Environment Setup
- Job failed before reaching dependency installation phase
- GPU availability check never executed
- No actual code execution occurred

## Timeline of Failure

1. **06:45:08** - Job started, began capability initialization
2. **06:45:11** - Execution wrapper started successfully
3. **06:45:12** - CS, Metrics, and Snapshot capabilities started successfully
4. **06:45:12** - **Data capability failed** to mount input URI
5. **06:45:13** - Job marked as failed and began cleanup
6. **06:45:33** - Job terminated

## Actionable Recommendations

### Immediate Actions (High Priority)

1. **Fix Data Access**:
   ```bash
   # Check if DADA-100-videos data asset exists
   az ml data list --resource-group llava-resourcegroup --workspace-name llava-workspace
   
   # If it doesn't exist, create it first
   az ml data create --file data_asset_config.yml
   ```

2. **Verify Storage Authentication**:
   - Ensure the workspace has proper permissions to access the storage account
   - Consider using managed identity instead of keys
   - Check if RBAC permissions are correctly configured

3. **Create Missing Data Asset**:
   ```yaml
   # Create dada_data_asset.yml
   name: dada-100-videos
   description: DADA dataset with 100 videos for LLaVA ghost probing detection
   type: uri_folder
   path: azureml://datastores/workspaceblobstore/paths/DADA-100-videos/
   ```

### Code Fixes Required

1. **Update Job Submission Script**:
   - Remove dependency on non-existent external data
   - Use local code upload instead of external data mounting
   - Implement self-contained test scenario

2. **Alternative Approach - Self-Contained Job**:
   ```python
   # Remove inputs section entirely
   inputs={
       # Remove this problematic input
       # "video_data": {
       #     "mode": "ro_mount",
       #     "path": "azureml://datastores/workspaceblobstore/paths/DADA-100-videos/",
       #     "type": "uri_folder"
       # }
   }
   ```

3. **Update Command Script**:
   - Remove references to `./inputs/video_data`
   - Create test videos within the job itself
   - Focus on validating LLaVA functionality rather than processing specific dataset

### Long-term Solutions

1. **Data Pipeline Setup**:
   - Upload DADA videos to proper blob storage location
   - Create registered data assets in Azure ML
   - Implement proper data versioning

2. **Authentication Improvements**:
   - Configure storage account for Azure AD authentication
   - Use workspace managed identity
   - Implement proper RBAC roles

3. **Job Monitoring**:
   - Add health checks for data availability before job submission
   - Implement better error handling and retry logic
   - Add validation steps in job configuration

## Next Steps

1. **Immediate**: Use the fixed job submission script that doesn't depend on external data
2. **Short-term**: Create and upload the DADA-100-videos data asset properly
3. **Long-term**: Implement robust data pipeline and authentication strategy

## Files to Review/Update

1. `/Users/wanmeng/repository/GPT4Video-cobra-auto/result/LlaVA/submit_azure_llava_job_fixed.py` - Already contains working solution
2. Create proper data asset configuration file
3. Update job monitoring scripts to validate data accessibility

## Conclusion

The job failed due to a data access issue, specifically the inability to mount the DADA-100-videos dataset from Azure blob storage. The quickest resolution is to use a self-contained job that doesn't depend on external data mounting, as implemented in the "fixed" version of the submission script. For production use, proper data asset creation and authentication setup will be required.