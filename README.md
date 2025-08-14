Landslide Detection

## 1. Business Understanding
Landslides are among the most destructive natural hazards, causing loss of life, infrastructure damage, and significant economic disruption.  
This project aims to **detect and classify landslides** from satellite imagery using models that can operate effectively even in persistent cloud cover.

### Objectives
1. Develop a detection pipeline using both **optical** and **Synthetic Aperture Radar (SAR)** satellite data.
2. Evaluate traditional ML and deep learning models using **F1 score** as the primary metric.
3. Provide a deployable and scalable model for integration into early-warning and disaster response systems.

---

## 2. Data Understanding
**Source:** ESA Copernicus Sentinel-1 & Sentinel-2 satellites.

- **Sentinel-1 (SAR)** → Radar data (VV & VH polarizations), works through clouds, sensitive to surface roughness and moisture.
- **Sentinel-2 (Optical)** → RGB + NIR bands, provides vegetation and terrain information.

**Target Variable:**  
- `1` – Landslide  
- `0` – No Landslide

<img width="800" height="500" alt="my_plot_highres" src="https://github.com/user-attachments/assets/414b3faa-82a9-467d-ada0-5efd482d9154" />

---

## 3. Data Preparation

### 3.1 Cleaning
- Removed corrupted or incomplete image tiles.
- Checked and maintained consistent coordinate reference systems.
- Handled missing values in derived indices.

### 3.2 Feature Engineering
To improve model performance, multiple SAR and optical features were engineered:

**SAR-Derived Features**
- **Backscatter Ratios:** VV/VH ratio to highlight terrain structure.
- **Texture Measures:** GLCM texture features (contrast, homogeneity, entropy).
- **Speckle Filtering:** Enhanced signal-to-noise for more stable backscatter values.

**Optical-Derived Features**
- **Vegetation Indices:** NDVI, SAVI to capture vegetation cover changes.
- **Water Index:** NDWI to identify water-logged regions.
- **Burn Ratio:** NBR for detecting recent soil disturbances.

**Band Statistics**
- Per-band means, standard deviations, and percentiles.
- Ratios between selected bands to capture spectral relationships.

**For CNN**
- Raw patch extraction from multi-band imagery.
- Data augmentation: flips, rotations, brightness/contrast adjustments.
- Conversion to TFRecords for efficient training in TensorFlow.

---

## 4. Modelling

### 4.1 XGBoost
Regularized gradient boosting algorithm known for high performance and overfitting control.

**Validation Performance**  
- **F1 Score:** 0.8528
- Accuracy: 0.9463  
- AUC: 0.9834

**AUC **

<img width="450" height="319" alt="xgb" src="https://github.com/user-attachments/assets/521557c8-15d9-48ad-9a80-bcb8d3d9bbe3" />

**Confusion Matrix**

<img width="450" height="315" alt="xgb" src="https://github.com/user-attachments/assets/4015c86b-490e-42fe-889b-d68a3ec8713a" />

---

### 4.2 LightGBM
Gradient boosting decision tree model optimized for speed and accuracy.

**Validation Performance**  
- **F1 Score:** 0.8567
- Accuracy: 0.9519
- AUC: 0.9838

**AUC**

<img width="451" height="317" alt="xgb" src="https://github.com/user-attachments/assets/318d5c5f-4219-4bcf-bfff-7e1cee86097a" />

**Confusion Matrix**

<img width="451" height="317" alt="xgb" src="https://github.com/user-attachments/assets/936e164a-ad40-4b48-bd99-eb6d6ef0159b" />

---

### 4.3 Stacking Classifier
Ensemble model combining LightGBM and XGBoost outputs for improved generalization.

**Validation Performance**  
- F1 Score: 0.8529
- Accuracy: 0.9502
- AUC: 0.9828

**AUC**

<img width="451" height="317" alt="xgb" src="https://github.com/user-attachments/assets/e1aaa132-e71d-4ff0-9b73-eec5d4fabd84" />

**Confusion Matrix**

<img width="450" height="311" alt="xgb" src="https://github.com/user-attachments/assets/0d2a7f44-d514-4506-bb9c-453e0983eae8" />


---

### 4.4 CNN (EfficientNetV2-B0)
Transfer learning–based deep convolutional neural network trained with SAR + optical bands.

**Validation Performance**  
- F1 Score: 0.7796 
- Accuracy: 0.92
- AUC: 0.9647 

**AUC**

<img width="422" height="326" alt="xgb" src="https://github.com/user-attachments/assets/33ff5aa1-169e-444c-9e0b-9fdad0c458df" />

**Confusion Matrix**

<img width="396" height="325" alt="xgb" src="https://github.com/user-attachments/assets/f1d5f888-7022-4dba-a632-485fbe15606b" />


## 5. Evaluation Summary

**Primary Metric:** **F1 Score** – chosen to balance false positives and false negatives in a disaster response context.

| Rank | Model                   | F1 Score   | Accuracy   | AUC    |
| ---- | ----------------------- | ---------- | ---------- | -------|
| 1    | LightGBM                | **0.8567** | **0.9519** | 0.9800 |
| 1    | Stacking Classifier     | **0.8567** | **0.9519** | 0.9800 |
| 3    | XGBoost                 | 0.8466     | 0.9463     | 0.9800 |
| 4    | CNN (EfficientNetV2-B0) | 0.7796     | 0.9200     | 0.9647 |


---

## 6. Deployment
- Packaged the CNN model and preprocessing pipeline for inference.
- Provided a REST API for batch predictions from satellite data.
- Integrated into GIS dashboards for decision support.

---

## 7. Future Work
- Integrate temporal sequences for early landslide detection.
- Explore vision transformers (ViTs) for improved feature extraction.
- Extend to multi-class severity classification.
