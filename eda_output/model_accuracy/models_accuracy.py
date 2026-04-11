import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
# ==============================
# DEVICE
# ==============================
device = torch.device("cpu")

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
# DATASET PATHS
# ==============================
HEART_IMG_PATH   = "testset/test_heart_image/"
LUNG_IMG_PATH    = "testset/test_lung_image"
HEART_AUDIO_PATH = "testset/test_heart_audio/spect"
LUNG_AUDIO_PATH  = "testset/test_lung_audio/spect"

# ==============================
# LOAD DATASETS
# ==============================
heart_img_loader = DataLoader(
    datasets.ImageFolder(HEART_IMG_PATH, transform=img_transform),
    batch_size=32,
    shuffle=False
)

lung_img_loader = DataLoader(
    datasets.ImageFolder(LUNG_IMG_PATH, transform=img_transform),
    batch_size=32,
    shuffle=False
)

heart_audio_loader = DataLoader(
    datasets.ImageFolder(HEART_AUDIO_PATH, transform=audio_transform),
    batch_size=32,
    shuffle=False
)

lung_audio_loader = DataLoader(
    datasets.ImageFolder(LUNG_AUDIO_PATH, transform=audio_transform),
    batch_size=32,
    shuffle=False
)

# ==============================
# CNN MODEL (FOR IMAGE)
# ==============================
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

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
# LOAD IMAGE MODELS
# ==============================
heart_img_model = CNN()
heart_img_model.load_state_dict(
    torch.load("Model_implementation/models/heart_image_model.pth", map_location=device)
)
heart_img_model.eval()

lung_img_model = CNN()
lung_img_model.load_state_dict(
    torch.load("Model_implementation/models/lung_image_model.pth", map_location=device)
)
lung_img_model.eval()

# ==============================
# LOAD AUDIO MODELS
# ==============================
heart_audio_model = resnet18(weights=None)
heart_audio_model.fc = nn.Linear(heart_audio_model.fc.in_features, 2)
heart_audio_model.load_state_dict(
    torch.load("Model_implementation/models/heart_audio_model.pth", map_location=device)
)
heart_audio_model.eval()

lung_audio_model = resnet18(weights=None)
lung_audio_model.fc = nn.Linear(lung_audio_model.fc.in_features, 2)
lung_audio_model.load_state_dict(
    torch.load("Model_implementation/models/lung_audio_model.pth", map_location=device)
)
lung_audio_model.eval()

# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate_model(model, loader, model_name):

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100

    print(f"{model_name} Accuracy: {accuracy:.2f}%")

    return accuracy

# ==============================
# GET ALL REAL ACCURACIES
# ==============================
heart_img_acc = evaluate_model(
    heart_img_model,
    heart_img_loader,
    "Heart Image CNN"
)

lung_img_acc = evaluate_model(
    lung_img_model,
    lung_img_loader,
    "Lung Image CNN"
)

heart_audio_acc = evaluate_model(
    heart_audio_model,
    heart_audio_loader,
    "Heart Audio ResNet"
)

lung_audio_acc = evaluate_model(
    lung_audio_model,
    lung_audio_loader,
    "Lung Audio ResNet"
)

# ==============================
# FINAL LIST FOR GRAPH
# ==============================


# ==============================
# CREATE ACCURACY TABLE
# ==============================

results_df = pd.DataFrame({
    "Model": [
        "Heart Image CNN",
        "Lung Image CNN",
        "Heart Audio ResNet",
        "Lung Audio ResNet"
    ],
    "Accuracy (%)": [
        heart_img_acc,
        lung_img_acc,
        heart_audio_acc,
        lung_audio_acc
    ]
})

print("\n==============================")
print("MODEL ACCURACY TABLE FOR TESTING DATA")
print("==============================")
print(results_df)

# Save table to CSV
results_df.to_csv("model_accuracy_table.csv", index=False)

# ==============================
# PLOT COMPARISON GRAPH
# ==============================

plt.figure(figsize=(10,6))

bars = plt.bar(
    results_df["Model"],
    results_df["Accuracy (%)"],
    color=[
        'royalblue',
        'cornflowerblue',
        'seagreen',
        'mediumseagreen'
    ]
)

# Add percentage labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.5,
        f'{height:.2f}%',
        ha='center',
        fontsize=10
    )

plt.ylim(0,100)
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("CNN vs ResNet Model Comparison for testing data")
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()