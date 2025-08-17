def run_all_eda(df):
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'column_names': df.columns.tolist()
    }
