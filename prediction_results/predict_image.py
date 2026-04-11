import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
device = torch.device("cpu")
# CNN architecture (same as training)
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

# Load models
heart_model = CNN()
lung_model = CNN()

heart_model.load_state_dict(torch.load("Model_implementation/models/heart_image_model.pth", map_location="cpu"))
lung_model.load_state_dict(torch.load("Model_implementation/models/lung_image_model.pth", map_location="cpu"))

heart_model.eval()
lung_model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def predict(image_path, model, classes):

    img = Image.open(image_path).convert("RGB")
    img_1 = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_1)
        probs = torch.softmax(output,dim=1)
        confidence, pred = torch.max(probs,1)

    return classes[pred.item()],confidence.item(),img

# Example images
heart_image = "testset/test_heart_image/normal/img_147.png"
lung_image = "testset/test_lung_image/pneumonia/person30_bacteria_150.jpeg"

# Class order (alphabetical)
heart_classes = ["abnormal","normal"]
lung_classes = ["normal","pneumonia"]

# Predictions
heart_lable,heart_prob,heart_img = predict(heart_image, heart_model, heart_classes)
lung_lable,lung_prob,lung_img = predict(lung_image, lung_model, lung_classes)

print("Heart Prediction:", heart_lable, "| Confidence:", round(heart_prob*100,2), "%")
print("Lung Prediction:", lung_lable, "| Confidence:", round(lung_prob*100,2), "%")

plt.figure(figsize=(10,4))

# Heart
plt.subplot(1,2,1)
plt.imshow(heart_img)
plt.title(f"Heart: {heart_lable} ({heart_prob*100:.2f}%)")
plt.axis("off")

# Lung
plt.subplot(1,2,2)
plt.imshow(lung_img)
plt.title(f"Lung: {lung_lable} ({lung_prob*100:.2f}%)")
plt.axis("off")

plt.show()