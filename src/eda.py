import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Load Cleaned Data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath, parse_dates=['TransactionMonth', 'VehicleIntroDate'])
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# Descriptive Statistics
def summarize_data(df):
    print("\n Data Info ")
    print(df.info())
    print("\nDescriptive Statistics (Numerical)")
    print(df.describe().T)

# Missing Value Assessment 
def check_missing(df):
    print("\n Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)

# Univariate Plots
def plot_distributions(df, save_dir="plots/univariate"):
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm', 'CapitalOutstanding']
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{col}_hist.png")
        plt.show()
        plt.close()

    cat_cols = ['Gender', 'VehicleType', 'Province', 'CoverType']
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Frequency of {col}")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{col}_bar.png")
        plt.show()
        plt.close()

# Bivariate Analysis: Loss Ratio by Segment
def analyze_loss_ratio(df, save_dir="plots/bivariate"):
    os.makedirs(save_dir, exist_ok=True)

    df['LossRatio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], np.nan)

    for col in ['Province', 'VehicleType', 'Gender']:
        plt.figure(figsize=(8, 4))
        avg_loss = df.groupby(col)['LossRatio'].mean().sort_values()
        counts = df[col].value_counts()

        title = f"Average Loss Ratio by {col}"
        if col == 'Gender':
            title += f" ( Skewed: {counts.to_dict()})"

        avg_loss.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(title)
        plt.ylabel("Loss Ratio")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"loss_ratio_by_{col}.png")
        plt.savefig(save_path)
        print(f" Plot saved to: {save_path}")
        plt.show()
        plt.close()

#  Correlation Matrix 
def correlation_analysis(df, save_path="plots/multivariate/correlation_heatmap.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    corr = df.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

# Temporal Analysis
def temporal_trends(df, save_dir="plots/time"):
    os.makedirs(save_dir, exist_ok=True)
    df['Month'] = df['TransactionMonth'].dt.to_period('M')

    monthly = df.groupby('Month')[['TotalClaims', 'TotalPremium']].sum().reset_index()
    monthly['Month'] = monthly['Month'].astype(str)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly, x='Month', y='TotalClaims', label='Claims')
    sns.lineplot(data=monthly, x='Month', y='TotalPremium', label='Premiums')
    plt.title("Total Claims and Premiums Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/claims_vs_premiums_over_time.png")
    plt.show()
    plt.close()

# Vehicle Make/Model Risk Analysis
def top_vehicle_risks(df, save_dir="plots/vehicle_risks"):
    os.makedirs(save_dir, exist_ok=True)
    top_makes = df.groupby('make')['TotalClaims'].sum().sort_values(ascending=False).head(10)
    top_models = df.groupby('Model')['TotalClaims'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 4))
    top_makes.plot(kind='bar', color='teal', edgecolor='black')
    plt.title("Top 10 Vehicle Makes by Claim Amount")
    plt.ylabel("Total Claims")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top_makes.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    top_models.plot(kind='bar', color='orange', edgecolor='black')
    plt.title("Top 10 Vehicle Models by Claim Amount")
    plt.ylabel("Total Claims")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top_models.png")
    plt.show()
    plt.close()

# Outlier Detection
def detect_outliers(df, save_dir="plots/outliers"):
    os.makedirs(save_dir, exist_ok=True)
    outlier_cols = ['TotalClaims', 'TotalPremium', 'CustomValueEstimate']
    for col in outlier_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Outlier Detection - {col}")
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"boxplot_{col}.png")
            plt.savefig(save_path)
            print(f" Outlier boxplot saved to: {save_path}")
            plt.show()
            plt.close()

# Additional Insight: Loss Ratio by ZipCode 
def postalcode_analysis(df, save_dir="plots/geography"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    grouped = df.groupby('PostalCode')['LossRatio'].mean().sort_values()
    grouped.tail(30).plot(kind='barh', color='purple')
    plt.title("Top 30 Zip Codes with Highest Average Loss Ratio")
    plt.xlabel("Loss Ratio")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "top_zipcodes_lossratio.png")
    plt.savefig(save_path)
    print(f" Zipcode plot saved to: {save_path}")
    plt.show()
    plt.close()

# Additional Insight: Skewness of Financial Columns
def skewness_summary(df):
    print("\nSkewness of Numerical Features ")
    numeric = df.select_dtypes(include=np.number)
    skewness = numeric.skew().sort_values(ascending=False)
    print(skewness)
    return skewness
