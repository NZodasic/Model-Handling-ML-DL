import os
import shutil
from google.colab import drive
import kagglehub

# Download latest version
path = kagglehub.dataset_download("maitam/vietnamese-traffic-signs")
print("Path to dataset files:", path)

# Define source and destination paths
# 'path' variable holds the Kaggle dataset path from the previous step
source_path = path
drive_path = "/content/drive/MyDrive/xxx-xxx"

# 'drive_path' variable holds the base Google Drive path
# Create a specific folder for the dataset within the drive_path
destination_folder_name = os.path.basename(source_path) # e.g., 'vietnamese-traffic-signs'
destination_path = os.path.join(drive_path, destination_folder_name)

print(f"Source dataset path: {source_path}")
print(f"Destination Google Drive path: {destination_path}")

# Copy the dataset to Google Drive
if os.path.exists(destination_path):
    print(f"Warning: Destination folder '{destination_path}' already exists. Skipping copy.")
    print("If you want to overwrite, please manually delete the folder first and re-run.")
else:
    try:
        shutil.copytree(source_path, destination_path)
        print(f"Dataset successfully copied from '{source_path}' to '{destination_path}'")
    except Exception as e:
        print(f"Error copying dataset: {e}")
