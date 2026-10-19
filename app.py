# Run with:  streamlit run app.py
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Parkinson's Disease Prediction", page_icon="🧠")

@st.cache_resource
def load_bundle():
    with open("parkinson_pipeline.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model, features = bundle["model"], bundle["features"]

st.title("🧠 Parkinson's Disease Prediction")
st.caption("Enter the voice measurements below to estimate Parkinson's risk.")
st.warning("Screening aid only — this is NOT a medical diagnosis. Consult a doctor.")

values = {}
cols = st.columns(2)
for i, feat in enumerate(features):
    with cols[i % 2]:
        values[feat] = st.number_input(
            feat,
            value=float(bundle["defaults"][feat]),
            format="%.6f",
            help=f"Dataset range: {bundle['mins'][feat]:.5g} to {bundle['maxs'][feat]:.5g}",
        )

if st.button("Predict", type="primary"):
    row = pd.DataFrame([values], columns=features)
    prob = model.predict_proba(row)[0][1]
    pred = int(model.predict(row)[0])
    st.metric("Parkinson's risk probability", f"{prob:.0%}")
    st.progress(float(prob))
    if pred == 1:
        st.error("Prediction: Parkinson's likely. Please consult a neurologist.")
    else:
        st.success("Prediction: Healthy (low Parkinson's likelihood).")
