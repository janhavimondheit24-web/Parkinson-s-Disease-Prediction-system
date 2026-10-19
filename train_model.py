# Run as a notebook cell or: python train_model.py   (parkinsons.data must be in the same folder)
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report)

df = pd.read_csv("parkinsons.data")
X = df.drop(columns=["status", "name"])
Y = df["status"]

# Stratified split keeps the healthy/Parkinson's ratio the same in train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=40, stratify=Y
)

# Scaler + SVM in ONE pipeline, so new inputs are scaled automatically
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=40)),
])
pipe.fit(X_train, Y_train)

# Evaluate ONLY on unseen test data
pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
print("Test accuracy :", accuracy_score(Y_test, pred))
print("Recall (PD)   :", recall_score(Y_test, pred))
print("ROC-AUC       :", roc_auc_score(Y_test, proba))
print("Confusion matrix:\n", confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))

# Save pipeline + feature order + default values for the app
bundle = {
    "model": pipe,
    "features": list(X.columns),
    "defaults": X.median().to_dict(),
    "mins": X.min().to_dict(),
    "maxs": X.max().to_dict(),
}
with open("parkinson_pipeline.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("Saved parkinson_pipeline.pkl")
