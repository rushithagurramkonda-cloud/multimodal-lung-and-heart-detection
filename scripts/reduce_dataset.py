import os
import random

# path to dataset
dataset_path = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\lung_images"

# number of images you want to keep per class
MAX_IMAGES = 1500  # change this number

for class_name in ["normal","pneumonia"]:
    folder = os.path.join(dataset_path, class_name)

    images = [f for f in os.listdir(folder) if f.endswith((".png", ".jpg", ".jpeg"))]

    print(f"\n{class_name} before:", len(images))

    if len(images) > MAX_IMAGES:
        images_to_delete = random.sample(images, len(images) - MAX_IMAGES)

        for img in images_to_delete:
            os.remove(os.path.join(folder, img))

    print(f"{class_name} after:", len(os.listdir(folder)))