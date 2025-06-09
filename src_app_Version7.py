import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
from src.models.simple_detector import SimpleDetector

CLASS_NAMES = ['person', 'car']  # Update as needed

@st.cache_resource
def load_model():
    model = SimpleDetector(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    return model

st.title("Automatic Object Detector")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Running inference...")
    model = load_model()
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    st.write(f"Prediction: **{CLASS_NAMES[pred]}**")