from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
from scipy.sparse import hstack, csr_matrix

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ----------------------
# App & CORS
# ----------------------
app = Flask(__name__)
CORS(app)

# ----------------------
# Load models
# ----------------------
claim_model = joblib.load("models/logisticregression_claim_model.pkl")
claim_scaler = joblib.load("models/scaler_claim.pkl")
claim_ohe = joblib.load("models/ohe_claim.pkl")

severity_model = joblib.load("models/random_forest_severity_model.pkl")
premium_model = joblib.load("models/randomforest_premium_model_fast.pkl")

# ----------------------
# Store original feature names
# ----------------------
NUM_COLS_CLAIM = None
CAT_COLS_CLAIM = None

# ----------------------
# Preprocessing functions
# ----------------------
def preprocess_claim_data(df):
    global NUM_COLS_CLAIM, CAT_COLS_CLAIM

    drop_cols = [
        "RecordID", "UnderwrittenCoverID", "PolicyID",
        "TransactionMonth", "VehicleIntroDate",
        "CalculatedPremiumPerTerm", "TotalPremium",
        "SumInsured", "CapitalOutstanding"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    if NUM_COLS_CLAIM is None:
        NUM_COLS_CLAIM = df.select_dtypes(include=[np.number]).columns.tolist()
    if CAT_COLS_CLAIM is None:
        CAT_COLS_CLAIM = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

    # Ensure numeric columns exist
    for col in NUM_COLS_CLAIM:
        if col not in df.columns:
            df[col] = 0

    # Ensure categorical columns exist
    for col in CAT_COLS_CLAIM:
        if col not in df.columns:
            df[col] = "__NA__"

    # Numeric features
    X_num = csr_matrix(df[NUM_COLS_CLAIM].fillna(0))

    # Categorical features
    if CAT_COLS_CLAIM:
        X_cat = df[CAT_COLS_CLAIM].fillna("__NA__").astype(str)
        X_cat_sparse = claim_ohe.transform(X_cat)

        # Adjust feature count to match scaler
        expected_features = claim_scaler.scale_.shape[0]
        total_features = X_num.shape[1] + X_cat_sparse.shape[1]

        if total_features != expected_features:
            logging.warning(f"[Claim] Feature mismatch: scaler expects {expected_features}, got {total_features}")
            if total_features < expected_features:
                diff = expected_features - total_features
                X_cat_sparse = hstack([X_cat_sparse, csr_matrix((X_cat_sparse.shape[0], diff))])
            elif total_features > expected_features:
                X_cat_sparse = X_cat_sparse[:, :expected_features - X_num.shape[1]]

        X_full = hstack([X_num, X_cat_sparse]).tocsr()
    else:
        X_full = X_num

    X_scaled = claim_scaler.transform(X_full)
    return X_scaled

def preprocess_severity_data(df):
    drop_cols = [
        "RecordID", "UnderwrittenCoverID", "PolicyID", "TransactionMonth",
        "Title", "Bank", "AccountType", "Gender", "Country", "Province",
        "PostalCode", "MainCrestaZone", "SubCrestaZone", "ItemType", "mmcode",
        "VehicleType", "make", "Model", "bodytype", "VehicleIntroDate",
        "AlarmImmobiliser", "TrackingDevice",
        "CapitalOutstanding", "SumInsured", "TotalPremium"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = df.drop(columns=["TotalClaims"], errors='ignore')
    logging.info(f"[Severity] Columns after drop: {X.columns.tolist()}")
    return X.fillna(0)

def preprocess_premium_data(df):
    drop_cols = ["RecordID", "UnderwrittenCoverID", "PolicyID", "TransactionMonth"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    logging.info(f"[Premium] Columns used: {X.columns.tolist()}")
    return X.fillna(0)

# ----------------------
# Routes
# ----------------------
# Upload CSV and get preview
@app.route("/api/predict_csv", methods=["POST"])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        df = pd.read_csv(file)
        logging.info(f"Uploaded CSV columns: {df.columns.tolist()}")

        # Only keep first 10 rows
        df = df.head(10)
        logging.info(f"Processing first 10 rows")

        # ----------------------
        # Claim probability
        claim_X = preprocess_claim_data(df)
        claim_preds = claim_model.predict_proba(claim_X)[:, 1]

        # ----------------------
        # Claim severity
        severity_X = preprocess_severity_data(df)
        severity_preds = severity_model.predict(severity_X)

        # ----------------------
        # Premium prediction
        premium_X = preprocess_premium_data(df)
        premium_preds = premium_model.predict(premium_X)

        # ----------------------
        # Save all predictions to CSV
        df_results = df.copy()
        df_results["ClaimProbability"] = claim_preds
        df_results["ClaimSeverity"] = severity_preds
        df_results["PremiumPrediction"] = premium_preds
        csv_path = "predictions.csv"
        df_results.to_csv(csv_path, index=False)
        logging.info(f"Predictions saved for {len(df_results)} rows to {csv_path}")

        # ----------------------
        # Prepare preview for frontend
        preview = df_results.to_dict(orient="records")

        return jsonify({
            "message": "Predictions generated successfully",
            "total_rows": len(df_results),
            "preview": preview,
            "csv_path": csv_path
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Predict for a single row by index
@app.route("/api/predict_row", methods=["POST"])
def predict_row():
    try:
        data = request.json
        row_data = pd.DataFrame([data])

        claim_X = preprocess_claim_data(row_data)
        claim_pred = claim_model.predict_proba(claim_X)[:, 1][0]

        severity_X = preprocess_severity_data(row_data)
        severity_pred = severity_model.predict(severity_X)[0]

        premium_X = preprocess_premium_data(row_data)
        premium_pred = premium_model.predict(premium_X)[0]

        return jsonify({
            "ClaimProbability": claim_pred,
            "ClaimSeverity": severity_pred,
            "PremiumPrediction": premium_pred
        })

    except Exception as e:
        logging.error(f"Single row prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Insurance Risk Analytics API is running."

# ----------------------
# Run server
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
