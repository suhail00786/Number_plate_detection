import streamlit as st
import torch
from PIL import Image
import io
import cv2
import numpy as np
from torchvision import transforms

# -------------------------------
# 🔹 App Configuration
# -------------------------------
st.set_page_config(
    page_title="Number Plate Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Number Plate Detection")
st.write("Upload an image of a vehicle to detect its number plate using the trained RNN model.")

# -------------------------------
# 🔹 Model Loading
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "learned_plate_detection"  # update this to your actual path if needed
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

model = load_model()

# -------------------------------
# 🔹 Prediction Function
# -------------------------------
def predict(image_pil):
    # Convert PIL -> Tensor
    img_tensor = transforms.ToTensor()(image_pil)
    pred = model([img_tensor])[0]

    # Convert PIL -> OpenCV (for drawing boxes)
    img_cv = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    for box, score in zip(pred["boxes"], pred["scores"]):
        if score > 0.1:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert back to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_rgb

# -------------------------------
# 🔹 Image Upload Section
# -------------------------------
uploaded_file = st.file_uploader("Upload a vehicle image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and convert to PNG if needed
    image = Image.open(uploaded_file)
    if image.format and image.format.lower() in ["jpeg", "jpg"]:
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Number Plate"):
        with st.spinner("Processing..."):
            result_img = predict(image)
        st.image(result_img, caption="Detected Number Plate", use_column_width=True)
else:
    st.info("Please upload an image (.png, .jpg, or .jpeg) to begin.")

# -------------------------------
# 🔹 Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ by Md Suhail")
