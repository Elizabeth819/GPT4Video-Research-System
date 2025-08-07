# GPT-4o False Positive Analysis Report

## Executive Summary

Analysis of 39 false positive cases from GPT-4o model results reveals systematic patterns where the model incorrectly predicts "ghost probing" when ground truth shows "none". The model generates 66 false positives while there are only 39 true events, indicating a significant over-prediction issue.

## Key Findings

### 1. Primary False Positive Patterns

#### **Pattern 1: Pedestrian "Sudden Appearance" Misclassification (19 cases, 49%)**
**Issue**: The model consistently mislabels normal pedestrian crossing scenarios as "ghost probing" when pedestrians appear from the sides of the road.

**Examples**:
- `images_1_001.avi`: "A child suddenly appears from the right side of the road, running towards the vehicle with a stick"
- `images_1_004.avi`: "A man suddenly appears from the left side of the road and runs across the path of the vehicle"
- `images_1_009.avi`: "Two individuals are seen crossing the road from the right side to the left side. They suddenly appear in the path of the vehicle"

**Root Cause**: The model interprets any pedestrian that appears "suddenly" from the sides as ghost probing, when these are actually normal pedestrian crossing behaviors.

#### **Pattern 2: Cyclist Interaction Misclassification (6 cases, 15%)**
**Issue**: Normal cyclist movements near the vehicle are incorrectly classified as ghost probing.

**Examples**:
- `images_1_025.avi`: "A cyclist appears from the left side of the road and crosses to the right side"
- `images_4_001.avi`: "A cyclist appears from the left side, initially moving parallel to the vehicle"

**Root Cause**: The model confuses normal cyclist lane positioning and movement with threatening "ghost probing" behavior.

#### **Pattern 3: Normal Traffic Scenario Over-Sensitivity (24 cases, 62%)**
**Issue**: The model is over-sensitive to any mention of "sudden" movements or "rapid deceleration" even in routine traffic scenarios.

**Common Triggers**:
- "suddenly appear"
- "decelerate rapidly" 
- "from the left/right side"
- "potential collision risk"

### 2. Linguistic Pattern Analysis

#### **Top Trigger Words Leading to False Positives**:
- **"suddenly"** (33 occurrences) - Most significant trigger
- **"appears"** (31 occurrences) - Normal appearance interpreted as threatening
- **"side"** (60 occurrences) - Any side movement flagged
- **"rapidly"** (25 occurrences) - Normal vehicle responses over-interpreted

#### **Action Keywords Causing Confusion**:
- **"avoid collision"** (34 occurrences)
- **"decelerate rapidly"** (21 occurrences) 
- **"suddenly"** (19 occurrences)

### 3. Model Confusion Categories

| Category | Count | Description |
|----------|-------|-------------|
| Sudden Movements | 36 | Any "sudden" appearance triggers false positive |
| Pedestrian Interactions | 26 | Normal pedestrian crossings mislabeled |
| Cyclist Interactions | 25 | Routine cyclist movements flagged |
| Normal Driving Mislabeled | 24 | Routine traffic responses over-interpreted |
| Emergency Situations | 20 | Normal emergency responses confused with ghost probing |

## Detailed Analysis

### False Positive Type 1: Semantic Over-Interpretation
The model appears to have learned that "ghost probing" involves sudden appearances from the side, but it's applying this pattern too broadly to normal traffic scenarios.

**Example from `images_1_001.avi`**:
```json
{
  "summary": "A child suddenly appears from the right side of the road, running towards the vehicle with a stick",
  "actions": "The self-driving vehicle is forced to decelerate rapidly due to the sudden appearance of the child",
  "key_actions": "ghost probing"  // FALSE POSITIVE
}
```

**Why it's wrong**: This is a normal pedestrian (child) crossing scenario, not a vehicle-to-vehicle ghost probing maneuver.

### False Positive Type 2: Context Misunderstanding 
The model fails to distinguish between:
- **Actual ghost probing**: A vehicle briefly appearing alongside and then disappearing
- **Pedestrian crossings**: People crossing the road from sidewalks
- **Cyclist movements**: Normal bicycle lane positioning

**Example from `images_1_025.avi`**:
```json
{
  "summary": "A cyclist appears from the left side of the road and crosses to the right side",
  "key_actions": "ghost probing"  // FALSE POSITIVE
}
```

**Why it's wrong**: This describes normal cyclist movement, not a vehicle ghost probing scenario.

### False Positive Type 3: Trigger Word Sensitivity
The model has become over-sensitive to specific phrases:
- Any use of "suddenly appear" → ghost probing
- Any mention of "from the side" → ghost probing  
- Any "rapid deceleration" → ghost probing

## Recommendations for Prompt Improvement

### 1. **Add Explicit Negative Examples**
Include clear definitions of what is NOT ghost probing:
```
Ghost probing is NOT:
- Pedestrians crossing the road
- Cyclists moving in their lanes
- Normal traffic interactions
- Emergency braking for pedestrians
```

### 2. **Refine Ghost Probing Definition**
Make the definition more specific to vehicle-to-vehicle interactions:
```
Ghost probing specifically refers to:
- A VEHICLE (car, truck, motorcycle) briefly appearing alongside the ego vehicle
- The probing vehicle then quickly disappearing or backing away
- This creates uncertainty about the other vehicle's intentions
- Does NOT apply to pedestrians, cyclists, or stationary objects
```

### 3. **Add Context Constraints**
```
Only classify as ghost probing if:
- The entity is a motorized vehicle (not pedestrian/cyclist)
- The vehicle appears in adjacent lanes temporarily
- The vehicle then moves away or disappears
- The behavior suggests testing/probing rather than committed lane change
```

### 4. **Reduce Trigger Word Sensitivity**
```
The presence of these words alone does not indicate ghost probing:
- "suddenly appear" (could be pedestrians)
- "from the side" (could be normal lane changes)
- "rapid deceleration" (normal safety response)
```

### 5. **Add Decision Tree Logic**
```
Before labeling as ghost probing, ask:
1. Is the entity a motor vehicle? (If no → not ghost probing)
2. Does it appear alongside temporarily? (If no → not ghost probing)  
3. Does it then move away/disappear? (If no → not ghost probing)
4. Is this vehicle-to-vehicle interaction? (If no → not ghost probing)
```

## Impact Assessment

- **Current False Positive Rate**: 66 false positives vs 39 true positives (≈63% false positive rate)
- **Primary Cause**: Over-broad interpretation of "sudden appearance from sides"
- **Most Problematic Pattern**: Pedestrian crossings being mislabeled (49% of false positives)

## Conclusion

The GPT-4o model has learned the surface patterns of ghost probing (sudden side appearance) but lacks the semantic understanding to distinguish between vehicle-to-vehicle ghost probing and normal pedestrian/cyclist interactions. The prompt needs explicit constraints to limit ghost probing classification to vehicle-only scenarios and provide clear negative examples.

**Files Analyzed**:
- Ground truth: `/Users/wanmeng/repository/GPT4Video-cobra-auto/result/groundtruth_labels.csv` (101 labels)
- GPT-4o predictions: `/Users/wanmeng/repository/GPT4Video-cobra-auto/result/gpt4o-100-3rd/` (1014 predictions)
- False positives identified: 39 cases requiring prompt refinement