#Collect data information
import pandas as pd

def collect_dataset_info(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    info = {}

    # Basic shape
    info['num_samples'], info['num_features'] = data.shape

    # Column names and types
    info['columns'] = dict(data.dtypes)

    # Count missing values per column
    info['missing_values'] = data.isnull().sum().to_dict()
    
    # Get class balance if "target" column exists
    if 'target' in data.columns:
        info['target_distribution'] = data['target'].value_counts().to_dict()
    
    # Describe numerics
    info['descriptive_stats'] = data.describe().to_dict()

    # Print all collected metadata
    for key, val in info.items():
        print(f"{key}:")
        print(val)
        print()
        
    return info



info = collect_dataset_info('/content/drive/MyDrive/ML/mcol.csv')
