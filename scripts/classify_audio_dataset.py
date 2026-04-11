import pandas as pd
import os
import shutil

# =========================
# PROJECT ROOT PATH
# =========================

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================
# CSV FILE PATHS
# =========================

HEART_CSV = os.path.join(BASE_PATH, "metadata", "Mix.csv")
LUNG_CSV = os.path.join(BASE_PATH, "metadata", "Mix.csv")

# =========================
# DATASET PATHS
# =========================

HEART_RAW = os.path.join(BASE_PATH, "dataset", "spectrograms","heart")
HEART_NORMAL = os.path.join(BASE_PATH, "testset", "test_heart_audio","spect" ,"normal")
HEART_ABNORMAL = os.path.join(BASE_PATH, "testset", "test_heart_audio","spect", "abnormal")

LUNG_RAW = os.path.join(BASE_PATH, "dataset", "spectrograms","lung")
LUNG_NORMAL = os.path.join(BASE_PATH, "testset", "test_lung_audio","spect","normal")
LUNG_ABNORMAL = os.path.join(BASE_PATH, "testset", "test_lung_audio","spect","abnormal")

# =========================
# CREATE FOLDERS
# =========================

os.makedirs(HEART_NORMAL, exist_ok=True)
os.makedirs(HEART_ABNORMAL, exist_ok=True)

os.makedirs(LUNG_NORMAL, exist_ok=True)
os.makedirs(LUNG_ABNORMAL, exist_ok=True)

# =========================
# LOAD CSV FILES
# =========================

heart_df = pd.read_csv(HEART_CSV)
lung_df = pd.read_csv(LUNG_CSV)

# =========================
# HEART AUDIO CLASSIFICATION
# =========================

print("Classifying Heart Audio...")

for _, row in heart_df.iterrows():

    sound_type = str(row["Heart Sound Type"]).lower()
    sound_id = row["Heart Sound ID"]

    file_name = sound_id + ".png"

    src = os.path.join(HEART_RAW, file_name)

    if not os.path.exists(src):
        continue

    # NORMAL
    if "normal" in sound_type:
        dst = os.path.join(HEART_NORMAL, file_name)

    # ABNORMAL
    else:
        dst = os.path.join(HEART_ABNORMAL, file_name)

    shutil.copy(src, dst)

print("Heart audio classification complete")

# =========================
# LUNG AUDIO CLASSIFICATION
# =========================

print("Classifying Lung Audio...")

for _, row in lung_df.iterrows():

    sound_type = str(row["Lung Sound Type"]).lower()
    sound_id = row["Lung Sound ID"]

    file_name = sound_id + ".png"

    src = os.path.join(LUNG_RAW, file_name)

    if not os.path.exists(src):
        continue

    # NORMAL
    if "normal" in sound_type:
        dst = os.path.join(LUNG_NORMAL, file_name)
        

    # ABNORMAL
    else:
        dst = os.path.join(LUNG_ABNORMAL, file_name)

    shutil.copy(src, dst)

print("Lung audio classification complete")

print("Dataset successfully organized!")