import streamlit as st
from PIL import Image
import numpy as np
from model import build_resunetplusplus
import cv2


@st.cache_resource
def load_model():
    import torch
    model1 = build_resunetplusplus()  # initialize your model architecture
    model1.load_state_dict(torch.load(r'D:\BMIProject\files_Adam\checkpoint.pth'))
    model1.eval()
    return model1


def preprocess_image(image):
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def predict_mask(image_tensor, model):
    import torch
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        return mask


def resize_mask_to_image(mask, image_size):
    import cv2
    return cv2.resize(mask, image_size, interpolation=cv2.INTER_NEAREST)


def draw_boundaries_on_image(original_image, mask):
    import cv2
    binary_mask = (mask > 0.5).astype(np.uint8)
    binary_mask_resized = cv2.resize(binary_mask, (original_image.width, original_image.height),
                                     interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_boundaries = np.array(original_image)
    for contour in contours:
        cv2.drawContours(image_with_boundaries, [contour], -1, (0, 0, 255), 2)
    return image_with_boundaries


# === MASKANI TO‘LDIRISH (CONTOUR FILLING) ===
def fill_contours(binary_mask, threshold=0.3):
    binary_mask = (binary_mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)
    return filled_mask.astype(np.float32)



# Streamlit UI
st.set_page_config(page_title="Semantic Segmentatsiya", layout="wide")

st.title("🛰️ Qishloq Xo'jaligi Yerlarini AI Orqali Aniqlash")
st.markdown("Tasvirni yuklang va segmentatsiya natijalarini ko‘ring.")

uploaded_file = st.file_uploader("📂 Tasvir yuklang", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner('🔄 Model yuklanmoqda...'):
        model = load_model()

    input_tensor = preprocess_image(image)

    with st.spinner('🔎 Segmentatsiya amalga oshirilmoqda...'):
        raw_mask = predict_mask(input_tensor, model)

    resized_mask = resize_mask_to_image(raw_mask, image.size)

    # === CHECKBOX: Fill contours ishlatish yoki yo'q ===
    apply_fill = st.checkbox("🔲 Maskani to'ldirishni (Fill Contours) ishlatish", value=True)

    threshold_value = 0.3  # Default

    if apply_fill:
        threshold_value = st.slider(
            "🔧 To'ldirish uchun Threshold qiymati:",
            min_value=0.0, max_value=1.0, value=0.3, step=0.01,
            help="Maskani to'ldirish uchun minimal ehtimoliy qiymatni tanlang (0.0-1.0)."
        )
        filled_mask = fill_contours(resized_mask, threshold=threshold_value)
    else:
        filled_mask = resized_mask

    # Rasm shakllantirish
    colored_mask = (filled_mask * 255).astype(np.uint8)

    colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2RGB)

    result_image = draw_boundaries_on_image(image, filled_mask)

    st.subheader("📊 Natijalar")

    # 3 ta ustun
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📌 Asl Tasvir")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🧠 Segmentatsiya Maskasi")
        st.image(colored_mask_rgb, use_container_width=True)

    with col3:
        st.markdown("### 🟥 Chegaralar bilan Tasvir")
        st.image(result_image, use_container_width=True)

    # Statistika qismi
    st.markdown("---")
    st.subheader("📈 Statistika")

    total_pixels = filled_mask.size
    land_pixels = np.sum(filled_mask > 0.5)
    non_land_pixels = total_pixels - land_pixels

    land_percentage = (land_pixels / total_pixels) * 100
    non_land_percentage = (non_land_pixels / total_pixels) * 100

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("🌱 Yer piksellari", f"{land_pixels} ta", f"{land_percentage:.2f}%")
    with col_stat2:
        st.metric("🏜️ Yer emas piksellari", f"{non_land_pixels} ta", f"{non_land_percentage:.2f}%")

    # Yuklab olish tugmalari
    st.markdown("---")
    st.subheader("⬇️ Natijalarni Yuklab Oling")

    from io import BytesIO

    buffer_mask = BytesIO()
    Image.fromarray(colored_mask_rgb).save(buffer_mask, format="PNG")
    st.download_button("📥 Segmentatsiya Maskasini Yuklab Olish", data=buffer_mask.getvalue(),
                       file_name="mask_filled.png", mime="image/png")

    buffer_result = BytesIO()
    Image.fromarray(result_image).save(buffer_result, format="PNG")
    st.download_button("📥 Chegaralar bilan Tasvirni Yuklab Olish", data=buffer_result.getvalue(),
                       file_name="boundary_filled.png", mime="image/png")
