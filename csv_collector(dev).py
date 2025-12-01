import pandas as pd
import numpy as np

def collect_dataset_info(file_path, target_columns=None, sample_rows=5):
    info = {}

    try:
        data = pd.read_csv(file_path)
        info["file_loaded"] = True
    except Exception as e:
        info["file_loaded"] = False
        info["error"] = str(e)
        return info

    info["num_samples"], info["num_features"] = data.shape
    info["columns"] = dict(data.dtypes.astype(str))
    info["memory_usage_MB"] = round(data.memory_usage(deep=True).sum() / (1024**2), 3)
    info["missing_values"] = data.isnull().sum().to_dict()
    info["duplicate_rows"] = int(data.duplicated().sum())
    info["unique_values"] = {col: data[col].nunique() for col in data.columns}
    info["sample_data"] = data.head(sample_rows).to_dict(orient="records")

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    info["numeric_columns"] = numeric_cols
    info["categorical_columns"] = categorical_cols

    if target_columns:
        target_candidates = [c for c in target_columns if c in data.columns]
    else:
        # Try common target names
        guess_targets = ["target", "label", "class", "y"]
        target_candidates = [c for c in data.columns if c.lower() in guess_targets]

    info["detected_target_columns"] = target_candidates

    if target_candidates:
        for t in target_candidates:
            info[f"target_distribution_{t}"] = data[t].value_counts(dropna=False).to_dict()

    info["descriptive_stats"] = data.describe(include="all").to_dict()
    if len(numeric_cols) > 1:
        info["correlation_matrix"] = data[numeric_cols].corr().round(3).to_dict()
    else:
        info["correlation_matrix"] = "Not enough numeric columns"

    for key, val in info.items():
        print(f"{key.upper()}:")
        print(val)
        print()

    return info


info = collect_dataset_info('path')
