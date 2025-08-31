import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from utils import calculate_metrics, seeding
from model import build_resunetplusplus
from metrics import DiceBCELoss
from train import DATASET, DataLoader, load_data


def load_model(checkpoint_path, model, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # Modelni yuklaydi
    model = model.to(device)  # Modelni GPU yoki CPU'ga o'tkazadi
    model.eval()  # Modelni baholash rejimiga o'tkazadi
    return model


def visualize_results(images, masks, preds, idx=0):
    image = images[idx].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    mask = np.squeeze(masks[idx], axis=0)  # (1, H, W) -> (H, W)
    pred = np.squeeze(preds[idx], axis=0)  # (1, H, W) -> (H, W)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Predicted Mask')
    plt.show()


def test_model(model, loader, loss_fn, device):
    model.eval()
    test_loss, test_jac, test_f1, test_recall, test_precision = 0, 0, 0, 0, 0

    images, masks, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_pred = torch.sigmoid(model(x))  # Model chiqishini sigmoiddan o'tkazish

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_bin = (y_pred > 0.5).float()  # Bashoratni binar maskaga aylantirish
            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []

            for yt, yp in zip(y, y_pred_bin):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            test_jac += np.mean(batch_jac)
            test_f1 += np.mean(batch_f1)
            test_recall += np.mean(batch_recall)
            test_precision += np.mean(batch_precision)

            images.extend(x.cpu().numpy())
            masks.extend(y.cpu().numpy())
            preds.extend(y_pred_bin.cpu().numpy())

    n = len(loader)
    print(f"Test Loss: {test_loss / n:.4f}")
    print(
        f"Jaccard: {test_jac / n:.4f}, F1: {test_f1 / n:.4f}, Recall: {test_recall / n:.4f}, Precision: {test_precision / n:.4f}")

    visualize_results(images, masks, preds)


if __name__ == "__main__":
    seeding(42)

    checkpoint_path = r"D:\NigmatovSardor\AgricultureSemanticSegmentation\files_Adam2\checkpoint.pth"
    dataset_path = r"D:\NigmatovSardor\AgricultureSemanticSegmentation\dataset"  # Papka yo'li

    batch_size = 16
    image_size = 256
    size = (image_size, image_size)

    # Ma'lumotlarni yuklash
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    test_dataset = DATASET(test_x, test_y, size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_resunetplusplus()
    model = load_model(checkpoint_path, model, device)

    loss_fn = DiceBCELoss()
    test_model(model, test_loader, loss_fn, device)
