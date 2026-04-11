import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile

device = "cpu"

# -------------------------
# CNN MODEL (MATCH TRAINING)
# -------------------------

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # conv.0
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, 3, padding=1),  # conv.3
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, padding=1),  # conv.6
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16384,128),   # fc.0
            nn.ReLU(),
            nn.Linear(128,2)        # fc.2
        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# -------------------------
# LOAD MODELS
# -------------------------

heart_image_model = CNNModel()
heart_image_model.load_state_dict(
    torch.load("models/heart_image_model.pth", map_location=device)
)

lung_image_model = CNNModel()
lung_image_model.load_state_dict(
    torch.load("models/lung_image_model.pth", map_location=device)
)

heart_audio_model = resnet18(weights=None)
heart_audio_model.fc = nn.Linear(512,2)
heart_audio_model.load_state_dict(
    torch.load("models/heart_audio_model.pth", map_location=device)
)

lung_audio_model = resnet18(weights=None)
lung_audio_model.fc = nn.Linear(512,2)
lung_audio_model.load_state_dict(
    torch.load("models/lung_audio_model.pth", map_location=device)
)

heart_image_model.eval()
lung_image_model.eval()
heart_audio_model.eval()
lung_audio_model.eval()


# -------------------------
# IMAGE TRANSFORM
# -------------------------

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# -------------------------
# PREDICTION FUNCTION
# -------------------------

def predict(image, model, classes):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

        prob = torch.softmax(output, dim=1)

        conf, pred = torch.max(prob,1)

    return classes[pred.item()], conf.item()

def audio_to_spectrogram(audio_file):

    # Load audio
    y, sr = librosa.load(audio_file)

    plt.figure(figsize=(3,3))
    # Create spectrogram

    # Plot spectrogram
    librosa.display.specshow(librosa.amplitude_to_db(abs(librosa.stft(y))),
                            sr=sr,
                            x_axis='time',
                            y_axis='log',
                            cmap='magma')
    plt.axis("off")

    # Save temporary image
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)

    plt.savefig(temp_img.name, bbox_inches="tight", pad_inches=0)

    plt.close()

    # Open as PIL image
    img = Image.open(temp_img.name).convert("RGB")

    return img
# -------------------------
# STREAMLIT UI
# -------------------------

st.title("Heart & Lung Disease Detection System")

option = st.selectbox(
    "Select Prediction Type",
    ("Heart Image", "Lung Image", "Heart Audio", "Lung Audio")
)

uploaded_file = st.file_uploader(
    "Upload File",
    type=["png","jpg","jpeg","wav"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "wav":

        st.audio(uploaded_file)

        image = audio_to_spectrogram(uploaded_file)

        st.image(image, caption="Generated Spectrogram", width=300)

    else:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", width=300)


    if option == "Heart Image":

        result, conf = predict(
            image,
            heart_image_model,
            ["abnormal","normal"]
        )

    elif option == "Lung Image":

        result, conf = predict(
            image,
            lung_image_model,
            ["normal","pneumonia"]
        )

    elif option == "Heart Audio":

        result, conf = predict(
            image,
            heart_audio_model,
            ["abnormal","normal"]
        )

    elif option == "Lung Audio":

        result, conf = predict(
            image,
            lung_audio_model,
            ["abnormal","normal"]
        )

    st.success(f"Prediction: {result}")

    st.write(f"Confidence: {conf*100:.2f}%")