import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os

# ==============================
# CNN MODEL
# ==============================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# ==============================
# TRANSFORMS
# ==============================
img_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

audio_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ==============================
# LOAD MODELS
# ==============================
heart_img_model = CNN()
lung_img_model = CNN()

heart_img_model.load_state_dict(torch.load("Model_implementation/models/heart_image_model.pth"))
lung_img_model.load_state_dict(torch.load("Model_implementation/models/lung_image_model.pth"))

heart_audio_model = resnet18(weights=None)
heart_audio_model.fc = nn.Linear(heart_audio_model.fc.in_features, 2)
heart_audio_model.load_state_dict(torch.load("Model_implementation/models/heart_audio_model.pth"))

lung_audio_model = resnet18(weights=None)
lung_audio_model.fc = nn.Linear(lung_audio_model.fc.in_features, 2)
lung_audio_model.load_state_dict(torch.load("Model_implementation/models/lung_audio_model.pth"))

# Eval Mode
heart_img_model.eval()
lung_img_model.eval()
heart_audio_model.eval()
lung_audio_model.eval()

# ==============================
# LOADERS
# ==============================
heart_img_loader = DataLoader(
    datasets.ImageFolder("dataset/heart_image/proc_data", transform=img_transform),
    batch_size=32,
    shuffle=False
)

lung_img_loader = DataLoader(
    datasets.ImageFolder("dataset/lung_images", transform=img_transform),
    batch_size=32,
    shuffle=False
)

heart_audio_loader = DataLoader(
    datasets.ImageFolder("dataset/heart_audio/spect", transform=audio_transform),
    batch_size=32,
    shuffle=False
)

lung_audio_loader = DataLoader(
    datasets.ImageFolder("dataset/lung_audio/spect", transform=audio_transform),
    batch_size=32,
    shuffle=False
)

# ==============================
# METRICS FUNCTION
# ==============================
results = []
def evaluate_metrics(model, loader, class_names, model_name):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    weighted = report["weighted avg"]

    results.append({
        "Model": model_name,
        "Accuracy": round(acc*100, 2),
        "Precision": round(weighted["precision"], 4),
        "Recall": round(weighted["recall"], 4),
        "F1 Score": round(weighted["f1-score"], 4)
    })

    print(f"{model_name} Done")

# ==============================
# RUN EVALUATION
# ==============================
evaluate_metrics(heart_img_model, heart_img_loader, ["abnormal", "normal"], "Heart Image CNN")
evaluate_metrics(lung_img_model, lung_img_loader, ["normal", "pneumonia"], "Lung Image CNN")
evaluate_metrics(heart_audio_model, heart_audio_loader, ["abnormal", "normal"], "Heart Audio ResNet")
evaluate_metrics(lung_audio_model, lung_audio_loader, ["abnormal", "normal"], "Lung Audio ResNet")

os.makedirs("metadata", exist_ok=True)

df = pd.DataFrame(results)

csv_path = "metadata/precision_metrics.csv"
df.to_csv(csv_path, index=False)

print(f"\nMetrics saved to {csv_path}")
print(df)