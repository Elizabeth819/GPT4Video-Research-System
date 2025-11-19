# DriveLM vs AutoDrive-GPT 对比项目

这个目录包含了DriveLM与AutoDrive-GPT系统的完整对比分析项目。

## 📁 目录结构

```
drivelm_comparison/
├── README.md                    # 项目说明文件
├── drivelm_gpt41_results/       # DriveLM风格GPT-4.1处理结果
├── analysis/                    # 对比分析结果
├── configs/                     # 配置文件
├── outputs/                     # 原始输出数据
└── reports/                     # 分析报告
```

## 🎯 项目目标

1. **方法对比**: 比较DriveLM的Graph VQA方法与AutoDrive-GPT的Balanced Prompt Engineering
2. **性能评估**: 在相同的DADA-2000数据集上评估两种方法的Ghost Probing检测性能
3. **公平比较**: 使用相同的视频数据、评估标准和基础模型进行对比

## 🔬 实验设计

### DriveLM适配方案
- **方法**: Prompt-based Adaptation
- **基础模型**: GPT-4.1 (与AutoDrive-GPT相同)
- **Prompt风格**: Graph Visual Question Answering
- **数据集**: DADA-2000 (images_1_001 - images_5_XXX)

### 对比维度
1. **精确度 (Precision)**: 检测准确性
2. **召回率 (Recall)**: 检测完整性  
3. **F1分数**: 综合性能指标
4. **方法论差异**: 技术路径对比

## 📊 主要文件

### 处理脚本
- `ActionSummary-drivelm-gpt41.py`: DriveLM风格处理脚本
- `run_drivelm_style_test.py`: 批处理测试脚本

### 结果文件
- `drivelm_gpt41_results/`: DriveLM处理的JSON结果
- `analysis/drivelm_gpt41_comparison_test.csv`: 初步对比分析
- `reports/`: 详细分析报告

### 配置文件
- `configs/drivelm_config.json`: DriveLM系统配置
- `configs/autodrive_config.json`: AutoDrive-GPT配置
- `configs/comparison_config.json`: 对比实验配置

## 🚀 使用方法

### 1. 单视频测试
```bash
python ActionSummary-drivelm-gpt41.py --single DADA-2000-videos/images_1_001.avi
```

### 2. 批量测试
```bash
python run_drivelm_style_test.py
```

### 3. 完整100视频处理
```bash
python ActionSummary-drivelm-gpt41.py --folder DADA-2000-videos --interval 10 --frames 10
```

## 📈 进展记录

- ✅ 2025-07-12: DriveLM风格prompt设计完成
- ✅ 2025-07-12: 单视频测试成功 (images_1_001)
- ✅ 2025-07-12: 3视频批处理测试完成
- ⏳ 下一步: 完整100视频处理和详细对比分析

## 🎯 预期成果

1. **技术对比**: DriveLM Graph VQA vs AutoDrive-GPT Balanced Prompt
2. **性能分析**: 在Ghost Probing检测任务上的量化对比
3. **论文贡献**: 为AAAI 2026提供系统性的baseline对比
4. **方法论洞察**: 通用性方法 vs 任务特定优化的效果差异

## 📝 注意事项

- 所有DriveLM相关文件都集中在此目录中
- 与其他模型结果完全分离，便于管理和查找
- 保持实验的一致性和可重现性
- 确保公平对比的科学性