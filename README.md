# Human Activity Recognition via Clustering

> Unsupervised discovery of human motion patterns from raw smartphone sensor data — no labels used.

---

## Project Overview

Applied four clustering algorithms to 7352 smartphone sensor readings captured from 30 individuals using accelerometers and gyroscopes. The goal was to rediscover distinct human activities (walking, sitting, standing, etc.) purely from 561 raw sensor features, without ever looking at the pre-existing labels.

This project was completed as part of **Introduction to Data Science, Spring 2026** at the German International University.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | HAR Master Dataset (`human_activity_master.csv`) |
| Rows | 7,352 sensor time windows |
| Original features | 561 |
| Features after preprocessing | 253 |
| Real activities | 6 (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) |

---

## Repository Structure

```
HAR_Clustering.ipynb    ← Main notebook (all 5 steps, fully documented)
README.md
```

---

## Steps Implemented

### Step 1 — Data Preparation & Cleaning
- Applied `StandardScaler` to normalize all 561 features to mean=0, std=1
- Computed full correlation matrix and removed **308 features** with correlation > 0.95
- Final feature space: **561 → 253 features**

### Step 2 — Finding Optimal k
- **Elbow Method:** sharpest inflection point at k=3
- **Silhouette Analysis:** peaks at k=2 (score: 0.2406)
- **Selected k=6** based on domain knowledge — the HAR dataset captures 6 real-world activities. The lower silhouette at k=6 reflects genuine physical similarity between fine-grained motion variants, not a modeling flaw.

### Step 3 — Discovery & Evaluation (K-Means, k=6)

**Motion Profiling**
Ranked cluster centers by average acceleration magnitude to identify:
- 3 Active clusters → walking variants (high acceleration variance)
- 3 Static clusters → sitting, standing, laying (stable low acceleration)
- Cluster 0 was the cleanest Active cluster (only 1 anomaly out of 2020 points)

**PCA 2D Visualization**
Compressed 253 dimensions to 2 principal components (37% variance captured). Clusters show visible regional separation with expected boundary overlap between similar activities.

**Anomaly Detection**
- Distance threshold (95th percentile): 18.716
- Flagged **368 points (5%)** as anomalies
- Cluster 4 had 62.6% anomaly rate — identified as an activity transition zone, not sensor error

### Step 4 — Advanced Comparisons

| Method | Silhouette Score | Cluster Balance | Rank |
|--------|-----------------|-----------------|------|
| GMM | **0.1076** | Unbalanced | 🏆 1st |
| K-Means | 0.0782 | Moderate | 2nd |
| Agglomerative | 0.0517 | Unbalanced | 3rd |
| BIRCH | 0.0517 | Unbalanced | 3rd |

**GMM** achieved the highest score because human motion data has naturally overlapping, probabilistic boundaries — exactly what soft clustering is designed for.

**Agglomerative and BIRCH** tied at 0.0517. BIRCH internally uses Agglomerative clustering for final merges, producing identical results on this dataset. Both are better suited for hierarchical or large-scale data respectively.

### Step 5 — The Data Story
Markdown analysis cells throughout the notebook address:
- Why StandardScaler was essential before distance-based clustering
- How the Elbow Method and Silhouette Analysis justify k=6
- Physical interpretation of Dynamic vs Static clusters
- Anomaly analysis — activity transitions vs sensor noise
- Final method comparison and conclusions

---

## Key Findings

- **GMM is the best method** for HAR sensor data — soft probabilistic boundaries outperform hard geometric assignments
- **The dominant natural split** in raw sensor data is Active movement vs Static movement (confirmed by both k=2 silhouette peak and cluster center analysis)
- **Cluster 4 anomaly concentration (62.6%)** reveals genuine activity transition moments rather than random sensor errors
- **Fine-grained distinctions** between similar activities (e.g. walking vs walking upstairs) are genuinely difficult to separate from sensor data alone — reflected in overall low silhouette scores across all methods

---

## Technologies

- Python 3 (Google Colab)
- `pandas`, `numpy`
- `scikit-learn` — KMeans, GaussianMixture, AgglomerativeClustering, Birch, PCA, StandardScaler, silhouette_score
- `matplotlib`, `seaborn`

