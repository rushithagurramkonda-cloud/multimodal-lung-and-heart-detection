import os
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
heart_img = cv2.imread("dataset/heart_image/proc_data/normal/img_147.png")
lung_img = cv2.imread("dataset/lung_images/pneumonia/person11_bacteria_45.jpeg")
heart_spec = cv2.imread("dataset/heart_audio/spect/normal/H0039.png")
lung_spec = cv2.imread("dataset/lung_audio/spect/normal/L0041.png")
plt.figure(figsize=(8,8))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(heart_img, cv2.COLOR_BGR2RGB))
plt.title("Heart Image Sample")

plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(lung_img, cv2.COLOR_BGR2RGB))
plt.title("Lung Image Sample")

plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(heart_spec, cv2.COLOR_BGR2RGB))
plt.title("Heart audio Sample")

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(lung_spec, cv2.COLOR_BGR2RGB))
plt.title("Lung Audio Sample")

plt.show()

img = cv2.imread("dataset/heart_image/proc_data/normal/img_147.png")

resized = cv2.resize(img,(128,128))

normalized = resized/255.0

plt.imshow(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB))
plt.title("Preprocessed Image")
plt.show()