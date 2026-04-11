import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# FILE PATHS
# ==========================

metadata_path = "metadata/Mix.csv"

heart_audio_folder = "dataset/raw_audio/heart_raw"
lung_audio_folder = "dataset/raw_audio/lung_raw"

heart_spec_folder = "dataset/spectrograms/heart"
lung_spec_folder = "dataset/spectrograms/lung"

os.makedirs(heart_spec_folder, exist_ok=True)
os.makedirs(lung_spec_folder, exist_ok=True)

# ==========================
# LOAD METADATA
# ==========================

df = pd.read_csv(metadata_path)

# ==========================
# FUNCTION: CREATE SPECTROGRAM
# ==========================

def create_spectrogram(audio_path, save_path):

    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(3,3))
    librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(y))),
                            sr=sr,
                            x_axis='time',
                            y_axis='log',
                            cmap='magma')

    
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ==========================
# LOOP THROUGH DATASET
# ==========================

for index, row in df.iterrows():

    heart_id = row["Heart Sound ID"]
    lung_id = row["Lung Sound ID"]

    heart_audio = os.path.join(heart_audio_folder, heart_id + ".wav")
    lung_audio = os.path.join(lung_audio_folder, lung_id + ".wav")

    heart_output = os.path.join(heart_spec_folder, heart_id + ".png")
    lung_output = os.path.join(lung_spec_folder, lung_id + ".png")

    if os.path.exists(heart_audio):
        create_spectrogram(heart_audio, heart_output)

    if os.path.exists(lung_audio):
        create_spectrogram(lung_audio, lung_output)

print("All spectrograms generated successfully!")