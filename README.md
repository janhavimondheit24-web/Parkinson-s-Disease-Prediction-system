# Parkinson's Disease Detection & Prediction System

A machine learning project that detects Parkinson's disease from voice measurements
and predicts a patient's risk through a simple web app.

## Overview
- **Dataset:** UCI Parkinson's dataset (`parkinsons.data`), with 195 voice recordings and 22 biomedical voice features such as jitter, shimmer, HNR and PPE.
- **Model:** Support Vector Machine (linear kernel) in a scikit-learn pipeline with feature scaling.
- **Output:** Predicts Healthy or Parkinson's and gives a risk probability.
- **Interface:** A Streamlit app where the user enters voice measurements and gets a prediction.

> Disclaimer: This is an educational screening aid, not a medical diagnosis.

## Files
| File | Purpose |
|------|---------|
| `parkinsons.data` | Dataset with voice features and the `status` label (1 = Parkinson's, 0 = healthy) |
| `AIML_miniproj.ipynb` | Detection notebook: data exploration, training and comparing models, saving the SVM |
| `AIML_miniproj_PKLE.ipynb` | Prediction notebook: loads the saved model and predicts on the data |
| `train_model.py` | Trains the scaler + SVM pipeline, evaluates it on a held-out test set, saves `parkinson_pipeline.pkl` |
| `parkinson_pipeline.pkl` | Saved trained pipeline used by the app |
| `deploy_SVM.pkl` | Earlier saved SVM model (without the scaler), used by the prediction notebook |
| `app.py` | Streamlit web app for predicting on new patient input |

## How to run
`streamlit run app.py` to launch the app

## Limitations
- The dataset is small and has multiple recordings per person, so test results may be optimistic.
- Results should not be used for clinical decisions.
