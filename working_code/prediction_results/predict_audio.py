import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt

device = "cpu"

# ----------------------------
# Load HEART AUDIO MODEL
# ----------------------------

heart_model = resnet18(weights=None)
heart_model.fc = nn.Linear(heart_model.fc.in_features, 2)

heart_model.load_state_dict(torch.load("Model_implementation/models/heart_audio_model.pth", map_location=device))
heart_model.eval()

# ----------------------------
# Load LUNG AUDIO MODEL
# ----------------------------

lung_model = resnet18(weights=None)
lung_model.fc = nn.Linear(lung_model.fc.in_features, 2)

lung_model.load_state_dict(torch.load("Model_implementation/models/lung_audio_model.pth", map_location=device))
lung_model.eval()

# ----------------------------
# Image Transform
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------
# Classes
# ----------------------------

classes = ["abnormal", "normal"]

# ----------------------------
# Prediction Functions
# ----------------------------

def predict_audio(model,image_path,classes):

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = heart_model(img_tensor)
        probs = torch.softmax(output,dim=1)
        confidence, pred = torch.max(probs, 1)

    return classes[pred.item()], confidence.item(),img


# ----------------------------
# Test Examples
# ----------------------------

heart_file = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\heart_audio\spect\normal\M_N_RUSB.png"
lung_file = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\lung_audio\spect\normal\F_N_RMA.png"

heart_label,heart_conf,heart_img=predict_audio(heart_model,heart_file,classes)
lung_label,lung_conf,lung_img=predict_audio(lung_model,lung_file,classes)

print("Heart Prediction:", heart_label, "| Confidence:", round(heart_conf*100,2), "%")
print("Lung Prediction:", lung_label, "| Confidence:", round(lung_conf*100,2), "%")

plt.figure(figsize=(10,4))

# Heart
plt.subplot(1,2,1)
plt.imshow(heart_img)
plt.title(f"Heart: {heart_label} ({heart_conf*100:.2f}%)")
plt.axis("off")

# Lung
plt.subplot(1,2,2)
plt.imshow(lung_img)
plt.title(f"Lung: {lung_label} ({lung_conf*100:.2f}%)")
plt.axis("off")

plt.show()