import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==============================
# DEVICE
# ==============================

device = torch.device("cpu")

# ==============================
# DATASET PATHS
# ==============================

HEART_PATH = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\heart_image\proc_data"
LUNG_PATH = r"C:\Users\SIRI\OneDrive\Desktop\fds_proj\dataset\lung_images"

# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ==============================
# LOAD DATASETS
# ==============================

heart_dataset = datasets.ImageFolder(root=HEART_PATH, transform=transform)
lung_dataset = datasets.ImageFolder(root=LUNG_PATH, transform=transform)

heart_loader = DataLoader(heart_dataset, batch_size=32, shuffle=True)
lung_loader = DataLoader(lung_dataset, batch_size=32, shuffle=True)

print("Heart Classes:", heart_dataset.classes)
print("Lung Classes:", lung_dataset.classes)

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
        x = self.fc(x)
        return x

# ==============================
# TRAIN FUNCTION
# ==============================

def train_model(loader, model_name):

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10

    loss_list = []
    acc_list = []

    for epoch in range(EPOCHS):

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)

        print(f"{model_name} Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), f"models/{model_name}.pth")
    print(f"{model_name} saved!")

    return loss_list, acc_list

# ==============================
# TRAIN BOTH MODELS
# ==============================

heart_loss, heart_acc = train_model(heart_loader, "heart_image_model")
lung_loss, lung_acc = train_model(lung_loader, "lung_image_model")

print("Training complete!")

# ==============================
# PLOT 4 GRAPHS
# ==============================

plt.figure(figsize=(12,8))

# 1️⃣ Heart Accuracy
plt.subplot(2,2,1)
plt.plot(heart_acc, marker='o', color='blue')
plt.title("Heart Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# 2️⃣ Lung Accuracy
plt.subplot(2,2,2)
plt.plot(lung_acc, marker='o', color='green')
plt.title("Lung Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# 3️⃣ Heart Loss
plt.subplot(2,2,3)
plt.plot(heart_loss, marker='o', color='red')
plt.title("Heart Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 4️⃣ Lung Loss
plt.subplot(2,2,4)
plt.plot(lung_loss, marker='o', color='orange')
plt.title("Lung Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()