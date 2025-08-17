import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
st.title("Insurance Analytics Dashboard")
# Load models
severity_model = joblib.load("notebooks/saved_models/xgboost_severity_model.pkl")
premium_model = joblib.load("notebooks/saved_models/randomforest_best_model.pkl")
with open("notebooks/saved_models/xgb_claim_occurred_model.pkl", "rb") as f:
    claim_model = pickle.load(f)
with open("notebooks/saved_models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
# Section 1: Claim Severity Prediction
st.header("1️ Claim Severity Prediction")
severity_features = [
    "RecordID", "UnderwrittenCoverID", "PolicyID", "PostalCode", "mmcode",
    "RegistrationYear", "Cylinders", "cubiccapacity", "kilowatts",
    "NumberOfDoors", "CapitalOutstanding", "SumInsured",
    "CalculatedPremiumPerTerm", "TotalPremium", "TransactionYear",
    "VehicleAge", "ClaimRatio"
]
severity_input = {}
for feat in severity_features:
    if feat in ["RecordID", "UnderwrittenCoverID", "PolicyID", "PostalCode", "mmcode"]:
        severity_input[feat] = st.number_input(f"{feat}", value=0, step=1, key="sev_"+feat)
    else:
        severity_input[feat] = st.number_input(f"{feat}", value=0.0, key="sev_"+feat)

if st.button("Predict Claim Severity"):
    try:
        X_severity = pd.DataFrame([severity_input])
        pred = severity_model.predict(X_severity)[0]
        st.success(f"Predicted Claim Severity: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Section 2: Premium Price Prediction
st.header("2️ Premium Price Prediction")

premium_features = [
    "RecordID", "UnderwrittenCoverID", "PolicyID", "PostalCode", "mmcode",
    "RegistrationYear", "Cylinders", "cubiccapacity", "kilowatts",
    "NumberOfDoors", "CapitalOutstanding", "SumInsured", "TotalPremium"
]
premium_input = {}
for feat in premium_features:
    if feat in ["RecordID", "UnderwrittenCoverID", "PolicyID", "PostalCode", "mmcode"]:
        premium_input[feat] = st.number_input(f"{feat}", value=0, step=1, key="prem_"+feat)
    else:
        premium_input[feat] = st.number_input(f"{feat}", value=0.0, key="prem_"+feat)

if st.button("Predict Premium Price"):
    try:
        X_premium = pd.DataFrame([premium_input])
        pred = premium_model.predict(X_premium)[0]
        st.success(f"Predicted Premium Price: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Section 3: Claim Occurrence Prediction 
st.header("3️ Claim Occurrence Prediction")
claim_features = [
    "RecordID", "UnderwrittenCoverID", "PolicyID", "PostalCode", "mmcode",
    "RegistrationYear", "Cylinders", "cubiccapacity", "kilowatts",
    "NumberOfDoors", "CapitalOutstanding", "SumInsured", "TotalPremium"
]
claim_input = {}
for feat in claim_features:
    # Integer features
    if feat in ["RecordID", "UnderwrittenCoverID", "PolicyID", "PostalCode", "mmcode",
                "RegistrationYear", "NumberOfDoors", "Cylinders"]:
        claim_input[feat] = st.number_input(f"{feat}", value=0, step=1, key="claim_"+feat)
    else:
        claim_input[feat] = st.number_input(f"{feat}", value=0.0, key="claim_"+feat)
if st.button("Predict Claim Occurrence Probability"):
    try:
        X_claim = pd.DataFrame([claim_input])
        X_claim_scaled = scaler.transform(X_claim)
        proba = claim_model.predict_proba(X_claim_scaled)[0][1]
        pred_class = claim_model.predict(X_claim_scaled)[0]
        st.success(f"Prediction: {'Claim Occurred' if pred_class == 1 else 'No Claim Occurred'}")
        st.info(f"Predicted Probability of Claim Occurrence: {proba:.2%}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
