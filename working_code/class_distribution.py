
# ------------------------------------------
# 2. EDA (Exploratory Data Analysis)
# ------------------------------------------

import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageFolder

heart_image_path = "dataset/heart_image/proc_data"
lung_image_path = "dataset/lung_images"

heart_image_dataset = ImageFolder(heart_image_path)
lung_image_dataset = ImageFolder(lung_image_path)

print("Heart image Classes:", heart_image_dataset.classes)
print("Lung image Classes:", lung_image_dataset.classes)

heart_audio_path = "dataset/heart_audio/spect"
lung_audio_path = "dataset/lung_audio/spect"

heart_audio_dataset = ImageFolder(heart_audio_path)
lung_audio_dataset = ImageFolder(lung_audio_path)

print("Heart audio Classes:", heart_audio_dataset.classes)
print("Lung audio Classes:", lung_audio_dataset.classes)
# ------------------------------------------
heart_img_labels = [label for _, label in heart_image_dataset]
lung_img_labels = [label for _, label in lung_image_dataset]
heart_aud_labels = [label for _, label in heart_audio_dataset]
lung_aud_labels = [label for _, label in lung_audio_dataset]

plt.figure(figsize=(10,4))

plt.subplot(2,2,1)
sns.countplot(x=heart_img_labels)
plt.title("Heart Image Class Distribution")

plt.subplot(2,2,2)
sns.countplot(x=lung_img_labels)
plt.title("Lung Image Class Distribution")

plt.subplot(2,2,3)
sns.countplot(x=heart_aud_labels)
plt.title("Heart Audio Class Distribution")

plt.subplot(2,2,4)
sns.countplot(x=lung_aud_labels)
plt.title("Lung Audio Class Distribution")

plt.show()