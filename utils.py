import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from metrics import jac_score, recall, F2, dice_score, precision, hd_dist


#########################  Tasodifiylikni bir xil qilib seeding o'rnatish. #########################

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


######################### Yangi papka yaratish #########################

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


#########################  Datasetni aralashtirish #########################

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


#########################  Epoch vaqti (bosqichning boshlanish va tugash vaqti orasidagi vaqtni hisoblash). #########################

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


#########################  Natijalarni chop etish va saqlash #########################

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


#########################  Modellar uchun turli metrikalarni hisoblash #########################

def calculate_metrics(y_true, y_pred):
    ## Tensorlarni numpy formatiga o'tkazish
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    ## Thresholddan keyin taxminlarni (predictions) binar formatga o'tkazish
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)

    ## Hausdorff masofasi (HD) ni hisoblash
    if len(y_true.shape) == 3:
        score_hd = hd_dist(y_true[0], y_pred[0])
    elif len(y_true.shape) == 4:
        score_hd = hd_dist(y_true[0, 0], y_pred[0, 0])

    ## Binar formatni bir qatorli qilib joylashtirish (reshape)
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    ## Har xil metrikalarni hisoblash
    score_jaccard = jac_score(y_true, y_pred)  # Jaccard indeksi (IoU)
    score_f1 = dice_score(y_true, y_pred)  # Dice/F1 score
    score_recall = recall(y_true, y_pred)  # Recall
    score_precision = precision(y_true, y_pred)  # Precision
    score_fbeta = F2(y_true, y_pred)  # F2 score (ko'proq recall'ga urg'u beradi)
    score_acc = accuracy_score(y_true, y_pred)  # Aniqlik (Accuracy)

    ## Metrikalar ro'yxatini qaytarish
    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta, score_hd]
