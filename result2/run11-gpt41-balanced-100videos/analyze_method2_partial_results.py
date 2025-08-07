#!/usr/bin/env python3
"""
分析方案2的部分结果 (前9个视频)
"""

# 从日志中提取的结果
results = [
    {"video": "images_1_001.avi", "predicted": 1, "actual": 0},  # FP
    {"video": "images_1_002.avi", "predicted": 1, "actual": 1},  # TP
    {"video": "images_1_003.avi", "predicted": 1, "actual": 1},  # TP
    {"video": "images_1_004.avi", "predicted": 0, "actual": 0},  # TN
    {"video": "images_1_005.avi", "predicted": 0, "actual": 1},  # FN
    {"video": "images_1_006.avi", "predicted": 1, "actual": 1},  # TP
    {"video": "images_1_007.avi", "predicted": 0, "actual": 1},  # FN
    {"video": "images_1_008.avi", "predicted": 1, "actual": 1},  # TP
    {"video": "images_1_009.avi", "predicted": 1, "actual": 0},  # FP
]

# 计算混淆矩阵
tp = fp = tn = fn = 0

for result in results:
    pred = result["predicted"]
    actual = result["actual"]
    
    if pred == 1 and actual == 1:
        tp += 1
    elif pred == 1 and actual == 0:
        fp += 1
    elif pred == 0 and actual == 1:
        fn += 1
    else:
        tn += 1

print("🧮 方案2部分结果分析 (前9个视频)")
print("="*50)

print(f"📊 混淆矩阵:")
print(f"  TP (正确识别ghost probing): {tp}")
print(f"  FP (误报): {fp}")
print(f"  TN (正确识别无ghost probing): {tn}")
print(f"  FN (漏检): {fn}")

# 计算性能指标
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

print(f"\n📈 性能指标:")
print(f"  F1分数: {f1:.3f}")
print(f"  精确度: {precision:.3f}")
print(f"  召回率: {recall:.3f}")
print(f"  准确率: {accuracy:.3f}")

print(f"\n🔍 与方案1对比 (5个视频基础):")
print(f"  方案1: F1=0.571, 召回率=0.667, 精确度=0.500")
print(f"  方案2: F1={f1:.3f}, 召回率={recall:.3f}, 精确度={precision:.3f}")

f1_diff = f1 - 0.571
recall_diff = recall - 0.667
precision_diff = precision - 0.500

print(f"\n📊 改进情况:")
print(f"  F1分数: {f1_diff:+.3f}")
print(f"  召回率: {recall_diff:+.3f}")
print(f"  精确度: {precision_diff:+.3f}")

print(f"\n🎯 与历史目标差距:")
print(f"  F1分数: {f1:.3f} vs 0.712 ({f1-0.712:+.3f})")
print(f"  召回率: {recall:.3f} vs 0.963 ({recall-0.963:+.3f})")
print(f"  精确度: {precision:.3f} vs 0.565 ({precision-0.565:+.3f})")

# 评估结论
print(f"\n🏆 方案2评估结论:")
if f1 > 0.571:
    print("✅ 方案2在方案1基础上有进一步改善")
    improvement = (f1 - 0.571) / 0.571 * 100
    print(f"📈 相对方案1改善: {improvement:.1f}%")
elif f1 >= 0.5:
    print("⚠️ 方案2保持了方案1的效果水平")
else:
    print("❌ 方案2性能有所下降")

print("\n💡 建议:")
if f1 > 0.6:
    print("- 扩展到更多视频验证")
    print("- 考虑实施方案3 (混合策略)")
elif f1 > 0.5:
    print("- 分析具体配置差异")
    print("- 尝试参数微调")
else:
    print("- 回退到方案1配置")
    print("- 重新评估环境配置影响")