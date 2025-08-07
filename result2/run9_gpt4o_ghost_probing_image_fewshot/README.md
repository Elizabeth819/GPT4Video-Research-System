# Run 9: GPT-4o Ghost Probing Detection with Image Few-shot Learning

## 概述

Run 9 是基于 Run 8 架构的增强版本，集成了图像 few-shot learning 技术来提升 ghost probing 检测性能。该实现结合了文本 few-shot examples 和视觉 few-shot examples，为 GPT-4o 提供更丰富的视觉模式识别能力。

## 核心特性

### 🎯 Image Few-shot Learning 集成
- **14张示例图片**: 从 `/fsl/` 目录加载的高质量 ghost probing 示例
- **3类视觉场景**: 
  - Ghost Probing 序列 (6张): `frame_at_26s.jpg` ~ `frame_at_31s.jpg`
  - Lower Barrier 示例 (6张): `lowerbar_1s.jpg` ~ `lowerbar_9s.jpg`
  - Red Truck 示例 (2张): `redtruck-32s.png`, `redtruck-33s.png`

### 🧠 多模态 Few-shot 架构
- **文本 Few-shot**: 3个详细的JSON格式示例（继承自Run 8）
- **图像 Few-shot**: 14张不同场景的视觉示例
- **组合效应**: 文本描述 + 视觉模式识别

### ⚙️ 技术规格
- **模型**: GPT-4o (Azure OpenAI)
- **Temperature**: 0 (确保一致性)
- **Max Tokens**: 3000
- **API Timeout**: 90秒 (增加以适应更多图片)
- **Prompt**: Paper_Batch Complex (4-Task) + 双重 Few-shot

## 文件结构

```
run9_gpt4o_ghost_probing_image_fewshot/
├── run9_gpt4o_ghost_probing_image_fewshot_100videos.py  # 主实验脚本
├── test_run9_5videos.py                                 # 测试脚本
├── README.md                                            # 说明文档
└── [运行时生成的结果文件]
    ├── run9_test_results_*.json                         # 测试结果
    ├── run9_final_results_*.json                        # 完整实验结果
    ├── run9_metrics_*.json                              # 性能指标
    └── run9_image_fewshot_*.log                         # 日志文件
```

## 关键实现亮点

### 1. 智能图片加载系统
```python
def load_few_shot_images(self):
    """加载和编码few-shot示例图片"""
    # 自动分类组织图片
    ghost_probing_examples = ["frame_at_26s.jpg", ...]
    lower_barrier_examples = ["lowerbar_1s.jpg", ...]
    red_truck_examples = ["redtruck-32s.png", ...]
    
    # Base64编码存储
    for img_name in all_examples:
        with open(img_path, 'rb') as f:
            base64_img = base64.b64encode(f.read()).decode('utf-8')
            self.few_shot_images[img_name] = base64_img
```

### 2. 增强的API请求结构
```python
def send_azure_openai_request_with_image_fewshot(self, prompt, video_frames):
    """发送包含图像few-shot的API请求"""
    content = [{"type": "text", "text": prompt}]
    
    # 添加图像few-shot examples
    content.append({"type": "text", "text": "VISUAL FEW-SHOT EXAMPLES:"})
    
    # 按类别添加示例图片
    for category_images in [ghost_images, barrier_images, truck_images]:
        for img_name, base64_img in sorted(category_images.items()):
            content.append({"type": "text", "text": f"Example frame: {img_name}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
    
    # 添加当前视频帧
    content.append({"type": "text", "text": "NOW ANALYZE THESE VIDEO FRAMES:"})
    # ... (添加待分析的视频帧)
```

### 3. 增强的Prompt设计
```python
def get_paper_batch_prompt_with_image_fewshot(self, video_id):
    """包含文本+图像few-shot的综合prompt"""
    return f'''
    [基础任务说明...]
    
    **Text Few-shot Examples:**
    [3个详细的JSON示例...]
    
    **Visual Few-shot Examples:**
    The following visual examples demonstrate actual ghost probing scenarios:
    
    VISUAL EXAMPLE SET 1 - Sequential Ghost Probing Emergence
    VISUAL EXAMPLE SET 2 - Lower Barrier Ghost Probing  
    VISUAL EXAMPLE SET 3 - Vehicle Ghost Probing
    
    [详细的视觉识别指导...]
    '''
```

## 测试结果

### 5视频测试验证 (2025-07-27)
- **成功处理**: 5/5 视频 (100% 成功率)
- **性能指标**:
  - 精确度: 60.0% (3 TP, 2 FP)
  - 召回率: 100.0% (0 FN)
  - F1分数: 75.0%
  - 准确率: 60.0%
- **Few-shot加载**: 14张图片成功加载
- **处理时间**: 约23秒/视频 (包含图像few-shot处理)

## 与 Run 8 的对比

| 特性 | Run 8 | Run 9 |
|------|-------|-------|
| Few-shot类型 | 仅文本 | 文本 + 图像 |
| 示例数量 | 3个文本示例 | 3个文本 + 14张图片 |
| API请求大小 | 标准 | 增大 (包含图片) |
| 处理时间 | ~16秒/视频 | ~23秒/视频 |
| 视觉识别能力 | 基于描述 | 直接视觉模式匹配 |
| 预期改进 | - | 更好的视觉模式识别 |

## 运行方式

### 1. 测试运行 (推荐先运行)
```bash
cd /Users/wanmeng/repository/GPT4Video-cobra-auto/result2/run9_gpt4o_ghost_probing_image_fewshot
python test_run9_5videos.py
```

### 2. 完整实验 (100视频)
```bash
python run9_gpt4o_ghost_probing_image_fewshot_100videos.py
```

## 环境要求

### 必需的环境变量
```bash
OPENAI_API_KEY=your_azure_openai_key
VISION_ENDPOINT=https://your-endpoint.openai.azure.com/
VISION_DEPLOYMENT_NAME=gpt-4o-global
```

### Python依赖
- opencv-python>=4.8.1.78
- moviepy==1.0.3
- pandas
- tqdm
- requests
- python-dotenv

## 实验目标

### 主要研究问题
1. **图像few-shot是否显著提升ghost probing检测性能？**
2. **视觉模式识别vs文本描述的效果对比**
3. **不同类型障碍物的识别准确率改善程度**
4. **处理时间增加vs性能提升的权衡分析**

### 预期改进方向
- **更精确的障碍物识别**: 基于视觉相似性匹配
- **减少误判**: 通过视觉例子区分真假ghost probing
- **时序理解增强**: 通过序列图片理解动态过程
- **多场景适应**: 不同环境下的ghost probing模式识别

## 技术创新点

### 1. 多模态Few-shot Learning
- 首次在DADA-2000数据集上应用图像+文本双重few-shot
- 14张精选示例涵盖主要ghost probing场景类型

### 2. 分类图像示例系统
- 按场景类型组织示例图片 (序列、障碍、车辆)
- 自动加载和验证系统确保数据完整性

### 3. 增强API架构
- 智能内容组织：示例→当前帧的逻辑顺序
- 超时优化适应更大的请求负载

## 下一步计划

1. **完整100视频实验**: 获得完整性能数据
2. **与Run 8性能对比**: 量化图像few-shot的提升效果
3. **错误案例分析**: 分析FP/FN案例的视觉特征
4. **参数优化**: 测试不同的图片组合和顺序
5. **学术论文更新**: 将结果纳入AAAI26论文

## 贡献和创新

Run 9 代表了在自动驾驶视频分析领域中多模态few-shot learning的重要探索，为后续研究奠定了技术基础，特别是在ghost probing这一安全关键场景的识别上。