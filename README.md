# Landslide Detection

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

<img width="1600" height="800" alt="my_plot_highres" src="https://github.com/user-attachments/assets/414b3faa-82a9-467d-ada0-5efd482d9154" />

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

### 4.1 LightGBM
Gradient boosting decision tree model optimized for speed and accuracy.

**Validation Performance**  
- **F1 Score:** 0.8567
- Accuracy: 0.9519   
- AUC: 0.98 


---

### 4.2 XGBoost
Regularized gradient boosting algorithm known for high performance and overfitting control.

**Validation Performance**  
- **F1 Score:** 0.8466
- Accuracy: 0.9463  
- AUC: 0.98

---

### 4.3 Stacking Classifier
Ensemble model combining LightGBM and XGBoost outputs for improved generalization.

**Validation Performance**  
- F1 Score: 0.8567
- Accuracy: 0.9519 
- AUC: 0.98
- 
---

### 4.4 CNN (EfficientNetV2-B0)
Transfer learning–based deep convolutional neural network trained with SAR + optical bands.

**Validation Performance**  
- F1 Score: 0.7796 
- Accuracy: 0.92
- AUC: 0.9647 


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
