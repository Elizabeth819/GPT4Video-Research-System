# Run 13: Gemini 2.0 Flash DADA-100 Videos Analysis

## 实验目标
使用Gemini 2.0 Flash模型，采用与VIP目录下相同的prompt，对DADA-100视频数据集进行ghost probing检测分析。

## 实验设计

### 模型配置
- **模型**: `gemini-2.0-flash-exp`
- **Temperature**: 0 (确保结果一致性)
- **Max Output Tokens**: 4000

### 数据集
- **路径**: `/result/DADA-100-videos/`
- **视频数量**: 100个标准测试视频
- **格式**: `images_*.avi`

### 处理参数
- **帧提取间隔**: 10秒
- **每间隔最大帧数**: 10帧
- **基于**: VIP/ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py

### Prompt设计
使用与VIP脚本完全相同的prompt模板，包括：
1. **Ghost Probing定义**: 详细的鬼探头检测标准
2. **Cut-in vs Ghost Probing区分**: 明确的分类规则
3. **四项任务**:
   - 场景描述与人物识别
   - 当前驾驶行为解释
   - 下一步行为预测
   - 关键对象与行为一致性检查

## 运行方式

```bash
cd /Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run13-gemini-2.0-flash-dada100
python run13_gemini_2_0_flash_dada100.py
```

## 输出文件

### 单个视频结果
- **格式**: `actionSummary_{video_name}.json`
- **内容**: 标准化JSON格式，包含所有检测字段

### 汇总报告
- **文件**: `run13_gemini_2_0_flash_summary_{timestamp}.json`
- **内容**: 
  - 实验元数据
  - 处理统计信息
  - 所有视频结果汇总

### 日志文件
- **路径**: `logs/run13_gemini_2_0_flash_{timestamp}.log`
- **内容**: 详细的处理日志和错误信息

## 与其他Run的对比

| Run | 模型 | F1得分 | 精确度 | 召回率 | 视频覆盖 |
|-----|------|--------|--------|--------|----------|
| Run 8 | GPT-4o + Few-shot | 68.2% | 59.5% | 80.0% | 100/100 |
| Run 6 | GPT-4o + Paper_Batch | 63.6% | 55.4% | 74.5% | 100/100 |
| **Run 13** | **Gemini 2.0 Flash** | **待测** | **待测** | **待测** | **100/100** |
| result/ | Gemini 2.0 Flash (历史) | 19.7% | 36.8% | 13.5% | 98/101 |

## 实验假设
基于result/目录下Gemini 2.0 Flash的低性能表现 (F1=19.7%)，本次实验旨在验证：
1. 使用改进的prompt是否能提升Gemini 2.0 Flash性能
2. VIP脚本的详细指令是否对Gemini模型更有效
3. 相同prompt下不同模型的性能差异

## 预期改进方向
- 更详细的ghost probing定义
- 严格的cut-in vs ghost probing区分规则
- 四任务结构化分析
- 关键对象与行为一致性要求

## 技术细节
- **帧提取**: 基于moviepy，每10秒interval提取10帧
- **图像编码**: Base64编码传输给Gemini API
- **错误处理**: 完整的异常捕获和日志记录
- **资源清理**: 自动清理临时帧文件