# Ghost Probing Only Version - README

## 📁 文件对比

| 文件 | 用途 | 大小 | 说明 |
|------|------|------|------|
| `ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch.py` | 原始版本 | 96,482 字符 | 包含ghost probing和cut-in双重检测 |
| `ActionSummary-predict_explain-fsl-sys-En-cutin-time-paper-batch-GHOST_ONLY.py` | 专注版本 | 93,012 字符 | **仅专注ghost probing检测** |

## 🎯 修改目的

根据Run 7对比分析发现，**cut-in分支会干扰ghost probing的准确检测**：

### 问题分析
1. **边界案例混淆**: 原始脚本在images_1_011、images_1_012、images_1_017等视频中将ghost probing误判为cut-in
2. **分类干扰**: cut-in和ghost probing的判断逻辑存在重叠，导致模型困惑
3. **性能影响**: 双重分类增加了判断复杂度，影响了ghost probing检测精度

### 对比结果
- **原始脚本**: F1=0.667, 存在FN(漏检)和分类错误
- **Run 7重新实现**: F1=0.741, 更准确的ghost probing检测
- **主要差异**: 7个视频中有分类不一致，多数与cut-in/ghost probing混淆有关

## 🔧 具体修改内容

### 1. **Task 1标题简化**
```diff
- **Task 1: Identify and Predict potential "Ghost Probing(专业术语：鬼探头)",Cut-in(加塞) etc behavior**
+ **Task 1: Identify and Predict potential "Ghost Probing(专业术语：鬼探头)" behavior**
```

### 2. **完全移除Cut-in定义**
删除了以下内容：
- Cut-in的详细定义（约2000字符）
- Cut-in vs Ghost probing的区别说明
- Cut-in分类流程和示例
- Cut-in相关的惩罚机制

### 3. **简化key_actions分类**
```diff
- ghost probing
- cut-in                    # 删除
- overtaking
+ none (if no dangerous behavior)
```

### 4. **更新prompt示例**
```diff
- "key_actions": "cut-in: an object from an adjacent lane moves into the self-vehicle's lane"
+ "key_actions": "ghost probing: an object suddenly emerges from behind a physical obstruction into the vehicle's path"
```

### 5. **专注化指导原则**
```diff
- **Penalty for Mislabeling**: cut-in相关惩罚机制
+ **Focus on Ghost Probing Only**: 
  - 优先准确的ghost probing检测
  - ghost probing必须涉及从物理遮挡物后出现
  - 如不确定是否为ghost probing，使用"none"
```

## 🎯 预期效果

使用Ghost-Only版本预期能够：

1. **提高准确率**: 消除cut-in干扰，专注ghost probing检测
2. **减少误判**: 避免将ghost probing错误分类为cut-in
3. **增强一致性**: 简化分类逻辑，提高模型输出一致性
4. **改善性能**: 预期F1分数从0.667提升到接近0.741的水平

## 🚀 使用建议

1. **优先使用Ghost-Only版本**进行ghost probing检测实验
2. **对比测试**: 可以用相同视频对比两个版本的结果
3. **性能验证**: 建议先在20个测试视频上验证效果
4. **大规模应用**: 验证效果良好后可应用于完整数据集

## 📊 技术实现

修改通过`cut_in_removal_script.py`自动化完成：
- 使用正则表达式精确定位和删除cut-in相关内容
- 保持代码结构完整性
- 自动更新相关prompt和示例
- 生成详细的修改日志

这个版本将帮助你获得更纯净、更准确的ghost probing检测结果。