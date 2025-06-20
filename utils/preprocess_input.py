import pandas as pd

def prepare_features(df):
    df = df.copy()
    # Convert booleans
    df["TrackingDevice"] = df["TrackingDevice"].map({"Yes": 1, "No": 0})
    df["NewVehicle"] = df["NewVehicle"].map({"Yes": 1, "No": 0})

    # One-hot or label encode as used in training
    categorical_cols = ["Gender", "Province", "CoverType", "VehicleType"]
    df = pd.get_dummies(df, columns=categorical_cols)

    return df
