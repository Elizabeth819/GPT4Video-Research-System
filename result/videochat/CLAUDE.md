# CLAUDE.md in videochat
This file provides guidance to Claude Code (claude.ai/code) when working with code in this videochat repository.

# NO MOCK RULES
- Strictly no fabrication of non-existent data, models, or so-called “fallbacks”; no “mock data”, no placeholder, “proof of concept,” and “simulation” are forbidden. If you cannot verify something, reply “I don’t know.” No mocks. Do not simulate or invent data/models/results, do not deceive me, and do not lie for rewards—otherwise I’ll have your boss pull the plug.
- All output must be based on the project’s code or user-provided documentation; inventing information out of thin air is prohibited.
- Cite the exact file path for every code change.
- Require citations for critical claims
- Build production ready code instead of placeholders and to-dos.

## Workspace
in Azure ML:
export AZURE_SUBSCRIPTION_ID="0d3f39ba-7349-4bd7-8122-649ff18f0a4a"
export AZURE_RESOURCE_GROUP="videochat-group"
export AZURE_WORKSPACE_NAME="videochat-workspace"
use GPU compute cluster
videochat2-a100-low-priority
Virtual machine size
Standard_NC24s_v3 (24 cores, 448 GB RAM, 2948 GB disk)
Processing unit
    Standard_NC24ads_A100_v4 (24 cores, 220 GB RAM, 64 GB disk)
Environment
    AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu:10

## Video files
/Users/wanmeng/repository/GPT4Video-cobra-auto/result/videochat/DADA-100-videos

## Goal of coding
You should use the ghost probing detection prompt ,a balanced version which is the same as the prompt used for testing gpt-4.1

## Evaluate
You should measure accuracy, precision and recall for the 100 video test result, comparing with ground truth label file.

## Debug
Use the Playwright MCP to see the browser screenshot, if something isn’t working, test it on the browser using Playwright and understand what the issue is, and then fix it, you should be able to find errors that you can find inside the Azure machine learning.

