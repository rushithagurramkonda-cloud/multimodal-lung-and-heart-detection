import h5py
import numpy as np
import cv2
import os

# -------- PATHS --------
raw_path = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\heart_image\raw_data"
normal_path = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\heart_image\normal"
abnormal_path = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\heart_image\abnormal"

os.makedirs(normal_path, exist_ok=True)
os.makedirs(abnormal_path, exist_ok=True)

img_id = 0

for file in os.listdir(raw_path):

    if file.endswith(".h5"):
        file_path = os.path.join(raw_path, file)

        with h5py.File(file_path, "r") as hf:

            # print keys to see datasets
            print("Keys:", list(hf.keys()))

            images = hf["image"][:]     # change if dataset name is different
            labels = hf["label"][:]     # change if dataset name is different

            for i in range(len(images)):

                img = images[i]
                label = labels[i].argmax()
                img = (img * 255).astype(np.uint8)

                if label == 0:
                    save_path = os.path.join(normal_path, f"img_{img_id}.png")
                else:
                    save_path = os.path.join(abnormal_path, f"img_{img_id}.png")

                cv2.imwrite(save_path, img)
                img_id += 1

print("Extraction completed")