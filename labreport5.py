# CPU-based Image Classification Web App
import streamlit as st
import torch
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import pandas as pd

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Image Classification (CPU)",
    layout="centered"
)

st.title("🖼️ Image Classification using ResNet18")
st.caption("CPU-only inference using a pre-trained deep learning model")

# -------------------------------
# Device configuration (CPU)
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# Load model and weights
# -------------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.to(device)
model.eval()

# Preprocessing and labels
transform = weights.transforms()
class_labels = weights.meta["categories"]

# -------------------------------
# File uploader
# -------------------------------
uploaded_image = st.file_uploader(
    "Upload an image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Inference
    # -------------------------------
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)

    probs = F.softmax(logits[0], dim=0)

    # Top-5 predictions
    top_probs, top_indices = torch.topk(probs, 5)

    result_df = pd.DataFrame({
        "Class": [class_labels[i] for i in top_indices],
        "Confidence": top_probs.cpu().numpy()
    })

    # -------------------------------
    # Display results
    # -------------------------------
    st.subheader("🔍 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Prediction**")
        st.success(
            f"{result_df.iloc[0]['Class']} "
            f"({result_df.iloc[0]['Confidence']:.2%})"
        )

    with col2:
        st.markdown("**Model Info**")
        st.write("Architecture: ResNet18")
        st.write("Inference Device: CPU")

    st.markdown("### Top-5 Class Probabilities")
    st.dataframe(result_df, use_container_width=True)

    # -------------------------------
    # Visualization
    # -------------------------------
    st.markdown("### Probability Distribution")
    st.bar_chart(
        result_df.set_index("Class"),
        horizontal=True
    )

    # -------------------------------
    # Interpretation note
    # -------------------------------
    st.info(
        "Probabilities represent the model's confidence for each predicted class. "
        "Higher confidence indicates stronger model belief."
    )
