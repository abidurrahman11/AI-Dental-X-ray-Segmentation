import streamlit as st
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ─── CONFIG ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (512, 512)

# Load your best model
@st.cache_resource
def load_model():
    model = smp.UnetPlusPlus(                    
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    # PATH to actual best checkpoint file
    checkpoint_path = "best_unetpp_from_scratch_smp_resnet50.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    except Exception as e:
        st.error(f"Failed to load model: {e}\nMake sure the .pth file is in the same folder.")
        st.stop()
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Preprocessing
preprocess = A.Compose([
    A.Resize(*IMG_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
    A.Normalize(mean=0.0, std=1.0),
    ToTensorV2(),
])

# ─── APP UI ────────────────────────────────────────────────────────────────
st.title("🦷 Automatic Teeth Segmentation in Panoramic Dental X-rays (OPG)")
st.markdown("Upload a panoramic radiograph to see AI-predicted teeth regions.")

uploaded_file = st.file_uploader("Upload OPG image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Preprocess
        augmented = preprocess(image=image_np)
        input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits)
            mask = (prob > 0.5).float()

        # Resized original for display & overlay
        resized_orig = cv2.resize(image_np, IMG_SIZE[::-1])
        pred_mask_np = mask[0, 0].cpu().numpy()

        # Overlay
        overlay = resized_orig.astype(float) / 255.0
        red_mask = np.zeros_like(overlay)
        red_mask[pred_mask_np > 0.5] = [1.0, 0.3, 0.3]  # light red
        blended = 0.6 * overlay + 0.4 * red_mask
        blended = np.clip(blended, 0, 1)

        # Display
        col1, col2 = st.columns(2)
        col1.image(resized_orig, caption="Processed Original (512x512)", use_container_width=True)
        col2.image(blended, caption="Predicted Teeth Overlay", use_container_width=True)

        st.success("Segmentation complete!")

    except Exception as e:
        st.error(f"Processing error: {str(e)}")