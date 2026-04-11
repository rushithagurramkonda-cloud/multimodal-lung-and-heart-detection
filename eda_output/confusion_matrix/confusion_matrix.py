import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

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
# CNN MODEL (IMAGE)
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
# LOAD MODELS
# ==============================

heart_img_model = CNN()
heart_img_model.load_state_dict(torch.load("Model_implementation/models/heart_image_model.pth", map_location=device))
heart_img_model.eval()

lung_img_model = CNN()
lung_img_model.load_state_dict(torch.load("Model_implementation/models/lung_image_model.pth", map_location=device))
lung_img_model.eval()

heart_audio_model = resnet18(weights=None)
heart_audio_model.fc = nn.Linear(heart_audio_model.fc.in_features, 2)
heart_audio_model.load_state_dict(torch.load("Model_implementation/models/heart_audio_model.pth", map_location=device))
heart_audio_model.eval()

lung_audio_model = resnet18(weights=None)
lung_audio_model.fc = nn.Linear(lung_audio_model.fc.in_features, 2)
lung_audio_model.load_state_dict(torch.load("Model_implementation/models/lung_audio_model.pth", map_location=device))
lung_audio_model.eval()

# ==============================
# FUNCTION TO GENERATE CM
# ==============================

def plot_confusion_matrix(model, dataset_path, transform, title):

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=dataset.classes
    )

    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

# ==============================
# GENERATE ALL MATRICES
# ==============================

plot_confusion_matrix(
    heart_img_model,
    "testset/test_heart_image",
    img_transform,
    "Heart Image Confusion Matrix for testing data"
)

plot_confusion_matrix(
    lung_img_model,
    "testset/test_lung_image",
    img_transform,
    "Lung Image Confusion Matrix for testing data"
)

plot_confusion_matrix(
    heart_audio_model,
    "testset/test_heart_audio/spect",
    audio_transform,
    "Heart Audio Confusion Matrix for testing data"
)

plot_confusion_matrix(
    lung_audio_model,
    "testset/test_lung_audio/spect",
    audio_transform,
    "Lung Audio Confusion Matrix for testing data"
)