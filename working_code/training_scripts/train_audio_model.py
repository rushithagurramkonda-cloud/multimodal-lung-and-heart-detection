import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
# ==============================
# Device
# ==============================

device = torch.device("cpu")

# ==============================
# Paths
# ==============================

heart_path = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\heart_audio\spect"
lung_path = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\lung_audio\spect"

# ==============================
# Image Transform
# ==============================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor() #converting to vectors
])

# ==============================
# Load Datasets
# ==============================

heart_dataset = datasets.ImageFolder(
    root=heart_path,
    transform=transform
)

lung_dataset = datasets.ImageFolder(
    root=lung_path,
    transform=transform
)

heart_loader = DataLoader(heart_dataset, batch_size=32, shuffle=True)
lung_loader = DataLoader(lung_dataset, batch_size=32, shuffle=True)

print("Heart Classes:", heart_dataset.classes)
print("Lung Classes:", lung_dataset.classes)

#===============================
# Class Weights
#===============================

heart_class_counts = torch.bincount(
    torch.tensor([label for _, label in heart_dataset])
)

lung_class_counts = torch.bincount(
    torch.tensor([label for _, label in lung_dataset])
)

heart_weights = 1.0 / heart_class_counts.float()
lung_weights = 1.0 / lung_class_counts.float()

heart_weights = heart_weights / heart_weights.sum()
lung_weights = lung_weights / lung_weights.sum()

heart_weights = heart_weights.to(device)
lung_weights = lung_weights.to(device)

# ==============================
# Create Models
# ==============================

heart_model = resnet18(weights=ResNet18_Weights.DEFAULT)
heart_model.fc = nn.Linear(heart_model.fc.in_features, 2)
heart_model = heart_model.to(device)

lung_model = resnet18(weights=ResNet18_Weights.DEFAULT)
lung_model.fc = nn.Linear(lung_model.fc.in_features, 2)
lung_model = lung_model.to(device)

# ==============================
# Loss and Optimizer
# ==============================

heart_criterion = nn.CrossEntropyLoss(weight=heart_weights)
lung_criterion = nn.CrossEntropyLoss(weight=lung_weights)

heart_optimizer = optim.Adam(heart_model.parameters(), lr=0.0001)
lung_optimizer = optim.Adam(lung_model.parameters(), lr=0.0001)
lung_loss =[]
lung_acc=[]
heart_loss=[]
heart_acc=[]
# ==============================
# Train HEART AUDIO MODEL
# ==============================

print("\nTraining Heart Audio Model")

heart_model.train()

for epoch in range(10):

    running_loss = 0
    correct=0
    total =0
    for images, labels in heart_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = heart_model(images)
        loss = heart_criterion(outputs, labels)

        heart_optimizer.zero_grad()
        loss.backward()
        heart_optimizer.step()

        running_loss += loss.item()
        # Accuracy calculation
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)   

    epoch_loss = running_loss/len(heart_loader)
    epoch_acc= correct/total
    heart_loss.append(epoch_loss)
    heart_acc.append(epoch_acc)
    print(f"Heart Epoch {epoch+1} Loss: {running_loss/len(heart_loader):.4f} Accuracy: {epoch_acc:.4f}")

torch.save(heart_model.state_dict(), "Model_implementation/models/heart_audio_model.pth")

print("Heart audio model saved")

# ==============================
# Train LUNG AUDIO MODEL
# ==============================

print("\nTraining Lung Audio Model")

lung_model.train()

for epoch in range(10):
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in lung_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = lung_model(images)
        loss = lung_criterion(outputs, labels)

        lung_optimizer.zero_grad()
        loss.backward()
        lung_optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(lung_loader)
    epoch_acc = correct / total

    lung_loss.append(epoch_loss)
    lung_acc.append(epoch_acc)

    print(f"Lung Epoch {epoch+1} Loss: {running_loss/len(lung_loader):.4f} Accuracy: {epoch_acc:.4f}")

torch.save(lung_model.state_dict(), "Model_implementation/models/lung_audio_model.pth")

print("Lung audio model saved") 

plt.figure(figsize=(12,8))

#  Heart Accuracy
plt.subplot(2,2,1)
plt.plot(heart_acc, marker='o', color='blue')
plt.title("Heart Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

#  Lung Accuracy
plt.subplot(2,2,2)
plt.plot(lung_acc, marker='o', color='green')
plt.title("Lung Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

#  Heart Loss
plt.subplot(2,2,3)
plt.plot(heart_loss, marker='o', color='red')
plt.title("Heart Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

#  Lung Loss
plt.subplot(2,2,4)
plt.plot(lung_loss, marker='o', color='orange')
plt.title("Lung Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()