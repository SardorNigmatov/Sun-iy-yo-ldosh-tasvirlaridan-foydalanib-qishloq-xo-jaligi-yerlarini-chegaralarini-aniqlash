import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from utils import calculate_metrics, seeding, create_dir, print_and_save, shuffling, epoch_time
from model import build_resunetplusplus
from metrics import DiceBCELoss
from tqdm import tqdm
import csv
import os

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


# Tasvirlar, maskalar va bashoratlarni saqlash uchun funksiya
def save_images(images, masks, preds, epoch, output_dir="results_images"):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(images)):
        image = images[i].cpu().numpy().transpose(1, 2, 0) * 255.0
        mask = masks[i].cpu().numpy().transpose(1, 2, 0) * 255.0
        pred = preds[i].cpu().numpy().transpose(1, 2, 0) * 255.0

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        pred = Image.fromarray(pred.astype(np.uint8))

        image.save(os.path.join(output_dir, f"epoch_{epoch}_image_{i}.png"))
        mask.save(os.path.join(output_dir, f"epoch_{epoch}_mask_{i}.png"))
        pred.save(os.path.join(output_dir, f"epoch_{epoch}_pred_{i}.png"))


# Ma'lumotlarni yuklash uchun funksiya
def load_data(path):
    images_path = sorted(glob(os.path.join(path, "images", "*.png")))
    masks_path = sorted(glob(os.path.join(path, "masks", "*.png")))

    # Ma'lumotlarni o'quv, validatsiya va sinov to'plamlariga ajratish
    train_x, temp_x, train_y, temp_y = train_test_split(images_path, masks_path, test_size=0.4, random_state=42)
    valid_x, test_x, valid_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]


# Ma'lumotlar to'plamini yaratish uchun sinf
class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        # Tasvir va maskalarni o'qish
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        return self.n_samples


# Modelni o'qitish uchun funksiya
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        y_pred = torch.sigmoid(y_pred)
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


# Modelni baholash uchun funksiya
def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred = torch.sigmoid(y_pred)
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss /= len(loader)
        epoch_jac /= len(loader)
        epoch_f1 /= len(loader)
        epoch_recall /= len(loader)
        epoch_precision /= len(loader)

        # Tasodifiy rasmni tanlash va uning natijalarini olish
        rand_index = random.randint(0, len(loader.dataset) - 1)
        sample_img, sample_mask = loader.dataset[rand_index]
        sample_img = torch.tensor(sample_img, dtype=torch.float32).unsqueeze(0).to(device)
        sample_pred = model(sample_img)
        sample_pred = torch.sigmoid(sample_pred).detach().cpu().squeeze().numpy()

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall,
                        epoch_precision], sample_img.cpu().squeeze().numpy(), sample_mask, sample_pred


if __name__ == "__main__":
    seeding(42)  # Tasodifiylikni boshqarish uchun urug'ni o'rnatish
    create_dir("files_Adam3")  # Fayllar saqlanadigan papkani yaratish

    csv_log_path = "files_Adam3/train_log.csv"

    # CSV faylni yaratish
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "TrainLoss", "TrainJaccard", "TrainF1", "TrainRecall", "TrainPrecision",
                             "ValLoss", "ValJaccard", "ValF1", "ValRecall", "ValPrecision", "Time(min:s)"])

    train_log_path = "files_Adam3/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log fayli mavjud")
    else:
        train_log = open(train_log_path, "w")
        train_log.write("\n")
        train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)  # Logga hozirgi sana va vaqtni yozish
    print("")

    # Hyperparametrlarni belgilash
    image_size = 256
    size = (image_size, image_size)
    batch_size = 16
    num_epochs = 150
    lr = 0.0001
    early_stopping_patience = 25
    checkpoint_path = "files_Adam3/checkpoint.pth"
    path = r"D:\NigmatovSardor\AgricultureSemanticSegmentation\train\train"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)  # Ma'lumotlarni aralashtirish

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    # Tasvirlarni ko'paytirish uchun transformatsiyalar
    transform = A.Compose({
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CoarseDropout(num_holes_range=(1,10), hole_height_range=(8,32),hole_width_range=(8,32),p=0.3),
    })

    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda')  # CUDA qurilmasiga o'tish
    model = build_resunetplusplus()  # Modelni yaratish
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    print(scheduler.get_last_lr())  # Oxirgi LR ni chiqarish uchun
    loss_fn = DiceBCELoss()  # Yo'qotish funktsiyasini tanlash
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # O'qitish
        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics, sample_img, sample_mask, sample_pred = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        # Eng yaxshi modelni saqlash
        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 yaxshilandi: {best_valid_metrics:2.4f} dan {valid_metrics[1]:2.4f} ga. Tekshirishni saqlamoqda: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        # Namuna rasm, maskalar va bashoratlarni saqlash
        sample_img = np.transpose(sample_img, (1, 2, 0))  # (H, W, C) formatiga o'tkazish
        if sample_mask.ndim == 3:
            sample_mask = np.squeeze(sample_mask, axis=0)  # (H, W) formatiga o'tkazish
        if sample_pred.ndim == 3:
            sample_pred = np.squeeze(sample_pred, axis=0)  # (H, W) formatiga o'tkazish

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(sample_img)
        ax[0].set_title('Original Image')
        ax[1].imshow(sample_mask, cmap='gray')
        ax[1].set_title('Ground Truth Mask')
        ax[2].imshow(sample_pred, cmap='gray')
        ax[2].set_title('Predicted Mask')
        plt.savefig(f'files_Adam2/epoch_{epoch + 1}_sample.png')
        plt.close(fig)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # CSV faylga yozish
        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, train_loss, train_metrics[0], train_metrics[1], train_metrics[2], train_metrics[3],
                 valid_loss, valid_metrics[0], valid_metrics[1], valid_metrics[2], valid_metrics[3],
                 f"{epoch_mins}m {epoch_secs}s"])

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        # Erta to'xtatish shartlarini tekshirish
        if early_stopping_count == early_stopping_patience:
            data_str = f"Erta to'xtatish: validatsiya yo'qotishi {early_stopping_patience} ta ketma-ketlikdan keyin yaxshilanmadi.\n"
            print_and_save(train_log_path, data_str)
            break
