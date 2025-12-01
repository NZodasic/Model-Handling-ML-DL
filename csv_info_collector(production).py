import pandas as pd
import numpy as np
import os
import json

# 1. SETUP: Ensure we have a fallback if the env var isn't set
dataset_path = os.getenv('DATASET_PATH') 
# For testing purposes, you can hardcode a path if env var is missing:
if not dataset_path:
    dataset_path = 'your_data.csv' 

def collect_dataset_info(file_path, target_columns=None, sample_rows=5):
    """
    Collect detailed metadata about a CSV dataset with JSON-safe output.
    """
    info = {}

    # 2. ROBUSTNESS: Check if file exists before trying to read
    if not file_path or not os.path.exists(file_path):
        info["file_loaded"] = False
        info["error"] = f"File path not found or empty: {file_path}"
        return info

    try:
        data = pd.read_csv(file_path)
        info["file_loaded"] = True
    except Exception as e:
        info["file_loaded"] = False
        info["error"] = str(e)
        return info

    # --- Basic Metadata ---
    info["num_samples"] = int(data.shape[0]) # Convert to Python int
    info["num_features"] = int(data.shape[1])
    info["columns"] = data.dtypes.astype(str).to_dict()
    
    # Calculate memory and round
    mem_bytes = data.memory_usage(deep=True).sum()
    info["memory_usage_MB"] = round(mem_bytes / (1024**2), 3)
    
    # 3. TYPE CASTING: Convert numpy int64 to python int for JSON safety
    info["missing_values"] = data.isnull().sum().astype(int).to_dict()
    info["duplicate_rows"] = int(data.duplicated().sum())
    info["unique_values"] = {col: int(data[col].nunique()) for col in data.columns}
    
    # Get sample data but handle NaN (replace with None for valid JSON)
    info["sample_data"] = data.head(sample_rows).where(pd.notnull(data), None).to_dict(orient="records")

    # --- Column Types ---
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    info["numeric_columns"] = numeric_cols
    info["categorical_columns"] = categorical_cols

    # --- Target Detection ---
    if target_columns:
        target_candidates = [c for c in target_columns if c in data.columns]
    else:
        guess_targets = ["target", "label", "class", "y", "outcome", "churn"]
        target_candidates = [c for c in data.columns if c.lower() in guess_targets]

    info["detected_target_columns"] = target_candidates

    if target_candidates:
        for t in target_candidates:
            # Check if target is categorical or numeric to decide how to summarize
            if data[t].dtype == 'object' or data[t].nunique() < 20:
                dist = data[t].value_counts(dropna=False).to_dict()
                # Ensure keys/values are native types
                info[f"target_distribution_{t}"] = {str(k): int(v) for k, v in dist.items()}
            else:
                info[f"target_distribution_{t}"] = "Continuous target (distribution skipped)"

    # --- Advanced Stats ---
    # describe() produces NaNs for missing stats (e.g., std of a string). 
    # We strip those or replace them to ensure clean output.
    desc = data.describe(include="all")
    info["descriptive_stats"] = desc.where(pd.notnull(desc), None).to_dict()

    if len(numeric_cols) > 1:
        # Correlation matrix
        corr = data[numeric_cols].corr().round(3)
        info["correlation_matrix"] = corr.where(pd.notnull(corr), None).to_dict()
    else:
        info["correlation_matrix"] = "Not enough numeric columns"

    # 4. READABILITY: Pretty print using JSON
    print("-" * 30)
    print(f"REPORT FOR: {os.path.basename(file_path)}")
    print("-" * 30)
    
    # We use a custom encoder or simple dumps to print nicely
    print(json.dumps(info, indent=4, default=str))

    return info


# Example Usage
# Ensure you create a dummy csv or point to a real one for this to work
if __name__ == "__main__":
    # Create a dummy file for demonstration if it doesn't exist
    if not os.path.exists('test_data.csv'):
        df = pd.DataFrame({
            'feature_A': np.random.rand(100),
            'feature_B': np.random.randint(0, 100, 100),
            'category': np.random.choice(['red', 'blue', 'green'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        df.to_csv('test_data.csv', index=False)
        dataset_path = 'test_data.csv'

    info = collect_dataset_info(dataset_path)
