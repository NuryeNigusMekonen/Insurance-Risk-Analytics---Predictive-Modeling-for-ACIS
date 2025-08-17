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
# Store uploaded CSV globally for pagination
# ----------------------
uploaded_df = None

# ----------------------
# Preprocessing functions
# ----------------------
NUM_COLS_CLAIM = None
CAT_COLS_CLAIM = None

def preprocess_claim_data(df):
    global NUM_COLS_CLAIM, CAT_COLS_CLAIM
    drop_cols = ["RecordID","UnderwrittenCoverID","PolicyID","TransactionMonth","VehicleIntroDate",
                 "CalculatedPremiumPerTerm","TotalPremium","SumInsured","CapitalOutstanding"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    if NUM_COLS_CLAIM is None:
        NUM_COLS_CLAIM = df.select_dtypes(include=[np.number]).columns.tolist()
    if CAT_COLS_CLAIM is None:
        CAT_COLS_CLAIM = df.select_dtypes(include=['object','bool','category']).columns.tolist()

    # Ensure numeric columns
    for col in NUM_COLS_CLAIM:
        if col not in df.columns:
            df[col] = 0

    # Ensure categorical columns
    for col in CAT_COLS_CLAIM:
        if col not in df.columns:
            df[col] = "__NA__"

    X_num = csr_matrix(df[NUM_COLS_CLAIM].fillna(0))
    if CAT_COLS_CLAIM:
        X_cat = df[CAT_COLS_CLAIM].fillna("__NA__").astype(str)
        X_cat_sparse = claim_ohe.transform(X_cat)
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
    drop_cols = ["RecordID","UnderwrittenCoverID","PolicyID","TransactionMonth","Title","Bank",
                 "AccountType","Gender","Country","Province","PostalCode","MainCrestaZone",
                 "SubCrestaZone","ItemType","mmcode","VehicleType","make","Model","bodytype",
                 "VehicleIntroDate","AlarmImmobiliser","TrackingDevice","CapitalOutstanding",
                 "SumInsured","TotalPremium"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = df.drop(columns=["TotalClaims"], errors='ignore')
    return X.fillna(0)

def preprocess_premium_data(df):
    drop_cols = ["RecordID","UnderwrittenCoverID","PolicyID","TransactionMonth"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return X.fillna(0)

# ----------------------
# Prediction & EDA helpers
# ----------------------
def make_predictions(df_chunk):
    if df_chunk.empty:
        return []

    claim_X = preprocess_claim_data(df_chunk)
    claim_preds = claim_model.predict_proba(claim_X)[:,1]

    severity_X = preprocess_severity_data(df_chunk)
    severity_preds = severity_model.predict(severity_X)

    premium_X = preprocess_premium_data(df_chunk)
    premium_preds = premium_model.predict(premium_X)

    df_res = df_chunk[["RecordID","UnderwrittenCoverID","PolicyID","TransactionMonth"]].copy()
    df_res["ClaimProbability"] = claim_preds
    df_res["ClaimSeverity"] = severity_preds
    df_res["PremiumPrediction"] = premium_preds

    return df_res.replace({np.nan: None}).to_dict(orient="records")

def get_eda_preview(df_chunk):
    numeric_stats = df_chunk.describe().replace({np.nan: None}).to_dict()
    cat_cols = df_chunk.select_dtypes(include=['object','bool','category']).columns.tolist()
    cat_summary = {}
    for col in cat_cols:
        top = df_chunk[col].value_counts().head(5)
        cat_summary[col] = top.to_dict()
    return {
        "numeric_summary": numeric_stats,
        "top_categories": cat_summary
    }

# ----------------------
# Routes
# ----------------------
@app.route("/api/predict_csv", methods=["POST"])
def predict_csv():
    global uploaded_df
    try:
        if 'file' not in request.files:
            return jsonify({"error":"No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error":"No selected file"}), 400

        df = pd.read_csv(file)
        logging.info(f"Uploaded CSV columns: {df.columns.tolist()}")
        uploaded_df = df  # store globally

        preview_chunk = df.head(10)
        eda_chunk = df.sample(1000) if len(df) > 1000 else df.copy()

        predictions = make_predictions(preview_chunk)
        eda_preview = get_eda_preview(eda_chunk)

        return jsonify({
            "total_rows": len(df),
            "preview": predictions,
            "eda_preview": eda_preview,
            "page": 0
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/get_chunk", methods=["POST"])
def get_chunk():
    global uploaded_df
    try:
        if uploaded_df is None:
            return jsonify({"error":"No CSV uploaded"}), 400

        data = request.get_json()
        page = data.get("page", 0)
        start = page * 10
        end = start + 10
        chunk = uploaded_df.iloc[start:end]
        predictions = make_predictions(chunk)
        eda_preview = get_eda_preview(chunk)

        return jsonify({
            "rows": predictions,
            "eda_preview": eda_preview,
            "page": page
        })

    except Exception as e:
        logging.error(f"Chunk error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Insurance Risk Analytics API is running."

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
