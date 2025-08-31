import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model import build_resunetplusplus
import os


# === MODELNI YUKLASH ===
def load_model():
    model = build_resunetplusplus()
    state_dict = torch.load(r"D:\NigmatovSardor\AgricultureSemanticSegmentation\files_Adam2\checkpoint.pth", map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# === TASVIRNI PREPROCESS QILISH ===
def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0), image


# === MASKANI TOZALASH (MORPHOLOGICAL CLOSING) ===
def post_process_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = (mask > 0.4).astype(np.uint8) * 255
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return (closed_mask / 255.0).astype(np.float32)


# === MASKANI TO‘LDIRISH (CONTOUR FILLING) ===
def fill_contours(binary_mask):
    binary_mask = (binary_mask > 0.4).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)
    return filled_mask.astype(np.float32)


# === MASKA USTIDAN CHEGARALAR CHIZISH ===
def draw_boundaries_on_image(original_image, mask):
    binary_mask = (mask > 0.3).astype(np.uint8)
    binary_mask_resized = cv2.resize(binary_mask, (original_image.width, original_image.height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_boundaries = np.array(original_image)
    for contour in contours:
        cv2.drawContours(image_with_boundaries, [contour], -1, (0, 0, 255), 2)
    return image_with_boundaries


# === ASOSIY QISM ===
images_folder = r"D:\NigmatovSardor\AgricultureSemanticSegmentation\TestImages"
images_path = os.listdir(images_folder)

model = load_model()

save_dir = "Results"
os.makedirs(save_dir, exist_ok=True)

for i in range(len(images_path)):
    image_full_path = os.path.join(images_folder, images_path[i])
    input_image, original_image = preprocess(image_full_path)

    with torch.no_grad():
        output = model(input_image)
        predicted_mask = torch.sigmoid(output[0][0]).cpu().numpy()

    # === POST-PROCESSING QO‘LLASH ===
    predicted_mask = post_process_mask(predicted_mask)          # closing orqali tozalash
    # predicted_mask = fill_contours(predicted_mask)            # yoki: kontur asosida to‘ldirish

    image_with_boundaries = draw_boundaries_on_image(original_image, predicted_mask)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Kirish Tasvir')
    plt.imshow(original_image)

    plt.subplot(1, 3, 2)
    plt.title('Tozalangan Mask')
    plt.imshow(predicted_mask, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Chegaralari Chizilgan Tasvir')
    plt.imshow(image_with_boundaries)

    name_only = os.path.splitext(os.path.basename(images_path[i]))[0]
    plt.savefig(os.path.join(save_dir, f"{name_only}_results.png"))
    plt.close()
