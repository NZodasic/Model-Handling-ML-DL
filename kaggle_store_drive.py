# ============================================
# 1. Mount Google Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# Create a folder in Google Drive to store dataset
save_dir = "/content/drive/MyDrive/datasets/xxx-xxx"
import os
os.makedirs(save_dir, exist_ok=True)

# ============================================
# 2. Install kagglehub
# ============================================
!pip install -q kagglehub

# ============================================
# 3. Download dataset using kagglehub
# ============================================
import kagglehub

print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("xxx-xxx")
print("Downloaded to:", dataset_path)

# ============================================
# 4. Copy dataset to Google Drive
# ============================================
!cp -r "{dataset_path}"/* "{save_dir}"

print("Dataset copied to Google Drive at:", save_dir)
