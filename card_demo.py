import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load the model
resnet_model = models.resnet50(pretrained=False)
resnet_model.fc = torch.nn.Linear(2048, 53)  # Assuming 53 classes for your model
checkpoint = torch.load("final_resnet_model.pth", map_location=torch.device('cpu'))  # Change 'cpu' to 'cuda' if you're using GPU
resnet_model.load_state_dict(checkpoint)
resnet_model.eval()

# Define label mappings
labels_mapping = {
    0: 'joker',
    1: 'ace of clubs', 2: 'ace of diamonds', 3: 'ace of hearts', 4: 'ace of spades',
    5: 'eight of clubs', 6: 'eight of diamonds', 7: 'eight of hearts', 8: 'eight of spades',
    9: 'five of clubs', 10: 'five of diamonds', 11: 'five of hearts', 12: 'five of spades',
    13: 'four of clubs', 14: 'four of diamonds', 15: 'four of hearts', 16: 'four of spades',
    17: 'jack of clubs', 18: 'jack of diamonds', 19: 'jack of hearts', 20: 'jack of spades',
    21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts', 24: 'king of spades',
    25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts', 28: 'nine of spades',
    29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts', 32: 'queen of spades',
    33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts', 36: 'seven of spades',
    37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts', 40: 'six of spades',
    41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts', 44: 'ten of spades',
    45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts', 48: 'three of spades',
    49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts', 52: 'two of spades',
}


def classify_card(image_bytes):
    try:
        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_bytes).convert("RGB")
        input_data = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = resnet_model(input_data)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = labels_mapping.get(predicted_class_idx, "Unknown")

        return predicted_class

    except Exception as e:
        return str(e)

# Set up the Streamlit app
st.title("Card Classification")
st.write("Upload an image of a card to predict its class.")

# Image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image and predict the class
    if st.button("Predict Class"):
        predicted_class = classify_card(uploaded_file)
        st.success(f"Predicted Class: {predicted_class}")
