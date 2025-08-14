# 🌍 Landslide Classification Project

🌐 **[Launch Web App](https://landslideclassification.streamlit.app/)** – Try the interactive Landslide Classification tool.  
📓 **[View Notebook & Datasets](https://www.kaggle.com/code/arthur254/cnn-xgbm-sotr-efficientnetv2b0173ef58638-8f7654)** – Access the noteboo and datasets.  

---

## 1️⃣ Business Understanding
Landslides are among the most devastating natural hazards, causing significant loss of life, damage to infrastructure, and economic disruption. Detecting and classifying landslides quickly is critical for **early warning** and **disaster response systems**—especially in regions with persistent cloud cover.

### 🎯 Objectives
1. Build an accurate classification system using **optical** and **Synthetic Aperture Radar (SAR)** satellite data.
2. Compare traditional ML and deep learning approaches, using **F1 score** as the primary metric.
3. Deliver a scalable, deployable model ready for integration into real-world disaster monitoring pipelines.

---

## 2️⃣ Data Understanding
**Source:** ESA Copernicus **Sentinel-1** (SAR) & **Sentinel-2** (Optical) satellites.

- **Sentinel-1 SAR** → VV & VH polarizations; unaffected by cloud cover; sensitive to surface roughness and moisture.  
- **Sentinel-2 Optical** → RGB + NIR bands; captures vegetation health, terrain changes, and water coverage.

**Target Variable:**  
- `1` – Landslide  
- `0` – No Landslide  

<img src="https://github.com/user-attachments/assets/414b3faa-82a9-467d-ada0-5efd482d9154" width="451" height="320" />

---

## 3️⃣ Data Preparation

### 🧹 Cleaning
- Removed corrupted/incomplete tiles.
- Standardized coordinate reference systems.
- Addressed missing values in derived indices.

### ⚙️ Feature Engineering

**SAR-Derived Features:**
- **Backscatter Ratios:** VV/VH for structural contrast.
- **Texture Measures:** GLCM contrast, homogeneity, entropy.
- **Speckle Filtering:** Reduced noise for stable backscatter values.

**Optical-Derived Features:**
- **Vegetation Indices:** NDVI, SAVI for vegetation cover.
- **Water Index:** NDWI to detect water-logged areas.
- **Burn Ratio (NBR):** Soil disturbance detection.

**Statistical Features:**
- Per-band means, standard deviations, percentiles.
- Ratios between key bands to highlight spectral patterns.

**For CNN Models:**
- Raw patch extraction from multi-band imagery.
- Data augmentation: flips, rotations, brightness & contrast adjustments.
- Conversion to TFRecords for efficient TensorFlow training.

---

## 4️⃣ Modelling

### 🟦 4.1 XGBoost
Regularized gradient boosting with strong overfitting control.  

- **F1 Score:** 0.8466  
- **Accuracy:** 0.9463  
- **AUC:** 0.9834  

<img src="https://github.com/user-attachments/assets/521557c8-15d9-48ad-9a80-bcb8d3d9bbe3" width="520" height="500" />  
<img src="https://github.com/user-attachments/assets/4015c86b-490e-42fe-889b-d68a3ec8713a" width="520" height="500" />  

---

### 🟩 4.2 LightGBM – **Best Performer**
Faster training, lower memory usage, and excellent accuracy.  

- **F1 Score:** **0.8567**  
- **Accuracy:** **0.9519**  
- **AUC:** 0.9838  

<img src="https://github.com/user-attachments/assets/318d5c5f-4219-4bcf-bfff-7e1cee86097a" width="520" height="500" />  
<img src="https://github.com/user-attachments/assets/936e164a-ad40-4b48-bd99-eb6d6ef0159b" width="520" height="500" />   

---

### 🟨 4.3 Stacking Classifier
Meta-ensemble combining LightGBM & XGBoost predictions.  

- **F1 Score:** 0.8529  
- **Accuracy:** 0.9502  
- **AUC:** 0.9828  

<img src="https://github.com/user-attachments/assets/e1aaa132-e71d-4ff0-9b73-eec5d4fabd84" width="520" height="500" />   
<img src="https://github.com/user-attachments/assets/0d2a7f44-d514-4506-bb9c-453e0983eae8" width="520" height="500" />  

---

### 🟥 4.4 CNN – EfficientNetV2-B0
Transfer learning on SAR + optical bands.  

- **F1 Score:** 0.7796  
- **Accuracy:** 0.9200  
- **AUC:** 0.9647  

<img src="https://github.com/user-attachments/assets/33ff5aa1-169e-444c-9e0b-9fdad0c458df" width="520" height="500" />    
<img src="https://github.com/user-attachments/assets/f1d5f888-7022-4dba-a632-485fbe15606b" width="520" height="500" />   

---

## 5️⃣ Evaluation Summary

**Metric:** F1 score – chosen to balance false positives and false negatives in a high-risk disaster response scenario.

| Rank | Model                   | F1 Score   | Accuracy   | AUC    |
|------|------------------------|-----------|-----------|--------|
| **1** | LightGBM               | **0.8567** | **0.9519** | 0.9838 |
| 2     | Stacking Classifier    | 0.8529     | 0.9502     | 0.9828 |
| 3     | XGBoost                | 0.8466     | 0.9463     | 0.9834 |
| 4     | CNN (EffNetV2-B0)      | 0.7796     | 0.9200     | 0.9647 |

✅ **Recommendation:** Use **LightGBM** for deployment—it delivers the best F1 score and accuracy while maintaining fast inference speeds.

---

## 6️⃣ Deployment
- **Model:** LightGBM + preprocessing pipeline.
- **Backend:** FastAPI, hosted on Railway.
- **Frontend:** Streamlit web app for image uploads & predictions, hosted on Streamlit Cloud.

---

## 7️⃣ Future Work
- Incorporate **temporal sequences** for early detection.
- Experiment with **Vision Transformers (ViTs)** for richer feature extraction.
- Extend to **multi-class severity prediction** (low, medium, high impact).

---
