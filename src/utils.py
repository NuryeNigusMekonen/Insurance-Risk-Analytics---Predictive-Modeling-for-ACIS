def calculate_loss_ratio(df):
    """Compute the loss ratio = TotalClaims / TotalPremium."""
    if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    else:
        raise KeyError("Required columns missing.")
    return df
