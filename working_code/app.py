import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

device = "cpu"

# -------------------------
# CNN MODEL
# -------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*16*16,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------
# LOAD MODELS (CACHED)
# -------------------------
@st.cache_resource
def load_models():

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

    return heart_image_model, lung_image_model, heart_audio_model, lung_audio_model

heart_image_model, lung_image_model, heart_audio_model, lung_audio_model = load_models()

# -------------------------
# TRANSFORM
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

# -------------------------
# AUDIO → SPECTROGRAM
# -------------------------
def audio_to_spectrogram(audio_file):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path)

    plt.figure(figsize=(3,3))

    librosa.display.specshow(
        librosa.amplitude_to_db(abs(librosa.stft(y))),
        sr=sr,
        x_axis='time',
        y_axis='log',
        cmap='magma'
    )

    plt.axis("off")

    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_img.name, bbox_inches="tight", pad_inches=0)
    plt.close()

    img = Image.open(temp_img.name).convert("RGB")

    return img

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("Heart & Lung Disease Detection System")

st.sidebar.title("About")
st.sidebar.write("Multimodal Detection using CNN & ResNet")

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

    # ---------------- IMAGE ----------------
    if file_type != "wav":
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

    # ---------------- AUDIO ----------------
    else:
        st.audio(uploaded_file)
        image = audio_to_spectrogram(uploaded_file)
        st.image(image, caption="Generated Spectrogram", width=300)

    # ---------------- PREDICTIONS ----------------
    if option == "Heart Image":
        result, conf = predict(image, heart_image_model, ["abnormal","normal"])

    elif option == "Lung Image":
        result, conf = predict(image, lung_image_model, ["normal","pneumonia"])

    elif option == "Heart Audio":
        result, conf = predict(image, heart_audio_model, ["abnormal","normal"])

    elif option == "Lung Audio":
        result, conf = predict(image, lung_audio_model, ["abnormal","normal"])

    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {conf*100:.2f}%")