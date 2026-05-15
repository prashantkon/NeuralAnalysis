### Team Outliers - fMRI Cognitive State Classification
A machine learning project that classifies brain states(rest vs. active cognitive task) from fMRI signals using the <u>Haxby et. al (2001)</u> dataset.

Built for the UIUC Data Science Club Spring 2026 Data Dive competition.
_______________________________________________________________________________________________________________________________________________________

## Overview 
This project applies several machine learning models to fMRI time series data in order to predict whether a subject is at rest or engaged in a visual 
recognition task. Brain activity is extracted from 100 cortical regions using the Schaefer 2018 atlas, and models are trained and evaluated across all 
6 subjects in the Haxby dataset.
_______________________________________________________________________________________________________________________________________________________

## Dataset
**Haxby et. al (2001)** — a publicly available neuroimaging dataset included in [nilearn](https://nilearn.github.io/).
  * 6 subjects, each with whole-brain fMRI recordings
  * Stimuli categories: faces, houses, cats, bottles, scissors, shoes, chairs, and rest
  * Target: binary classification - rest(0) vs. any active task (1)
  * Brain parcellation: Schaefer 2018 atlas, 100 ROIs (regions of interest)
Data is downloaded automatically via "nilearn.datasets.fetch_haxby()" - no manual download required.
_______________________________________________________________________________________________________________________________________________________

## Models
| Model | Notes |
|---|---|
| Random Forest | 100 estimators, max depth 12 |
| SVM (RBF kernel) | Preceded by StandardScaler + PCA (80 components) |
| Neural Network (PyTorch) | 4-layer MLP (512→256→128→64→1), BatchNorm + Dropout, trained with SMOTE oversampling |
| XGBoost | Gradient boosted trees on preprocessed features |

## Project Structure
- datadive_final.ipynb        # Main notebook: data loading, EDA, Random
Forest, SVM, Neural Net       
- XGBOOST.ipynb               # XGBoost model and multi-subject data pipeline
- new.ipynb                   # Visualization utilities (PCA, correlation
heatmaps, time series)
- OlD_DATA__Data.ipynb        # Earlier exploratory work using NHANES diabetes
data (archived)
- README.md
_______________________________________________________________________________________________________________________________________________________
## Approach 
1. Data extraction - fMRI volumes are parcellated using "NiftiLabelsMasker" with the
Schaefer atlas, producing a time series matrix of shape (timepoints * 100
regions) per subject

3. Label encoding - stimulus labels are binarized: "rest=0", any task stimulus "=1".
 
4. EDA - PCA projections, correlations heatmaps, and mean activity comparisons between
rest and task states.

6. Modeling - models are trained on an 80/20 train/test split. The neural networks uses SMOTE to handle class imbalance.
 
7. Feature importance - top contributing brain regions are identified from the Random Forest and visualized on a glass brain using "nilearn.plotting".
   
