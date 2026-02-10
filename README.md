# Pairwise Learning + Voting Classifiers (MNIST)

## Overview
This project reformulates multiclass digit recognition (0–9) as a **binary similarity task**:

> Given two images, predict whether they belong to the **same class** (1) or **different classes** (0).

Then, for a new test image, we predict its digit using a **voting-based k-shot inference** procedure:
- sample *k* reference images per class
- score pairwise similarity between the test image and each reference
- predict the class with the most “same-class” matches

This mirrors real-world setups where:
- labeled data per class is limited
- we rely on similarity comparisons and retrieval-like classification

---

## Problem Setting
### Task A: Pairwise (Binary) Classification
Input: a pair of MNIST images (x_i, x_j)  
Label: 1 if y_i == y_j else 0

### Task B: Voting-based Multiclass Inference
For each test image:
1. select k reference images per class (0–9)
2. predict “same/different” for each pair
3. choose the class with the highest match count (majority vote)

---

## Methods
- Dataset: MNIST (digits 0–9)
- Training subset: balanced sample of 100 images per class (1000 total)
- Pair construction:
  - positive pairs: same digit
  - negative pairs: different digits
- Model: MLP on concatenated/paired features
- Loss: binary cross-entropy
- Inference: k-shot voting over classes

---

## Key Findings (from experiments)
- Standard multiclass MLP reaches high accuracy (~98%).
- Pairwise similarity learning is harder: accuracy can fall near chance (~50%) without careful design.
- Voting inference improves with k up to a point, but may saturate or degrade with large k.

Hence, transforming multiclass classification into similarity learning introduces a harder generalization problem unless the representation is well-structured.
---

## Origin
Adapted and extended from IIT ML coursework on MNIST pairwise learning and k-voting inference.
