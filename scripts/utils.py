import os
import pdb
import torch
import pickle
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, auc, roc_curve, balanced_accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from argparse import ArgumentParser
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.Tensor([alpha])
        self.size_average = size_average

    def forward(self, logits, label):
        # logits:[b,h,w] label:[b,h,w]
        pred = logits
        pred = pred.view(-1)  # b*h*w
        label = label.view(-1)

        if self.alpha:
            self.alpha = self.alpha.type_as(pred.data)
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)  # b*h*w

        pt = pred * label + (1 - pred) * (1 - label)
        diff = (1 - pt) ** self.gamma

        FL = -1 * alpha_t * diff * pt.log()

        if self.size_average:
            return FL.mean()
        else:
            return FL.sum()


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))

        return x


class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


def split_set(data_to_split, ratio=0.8):
    mask = np.random.rand(len(data_to_split)) < ratio
    return [data_to_split[mask], data_to_split[~mask]]


def compute_confusion_matrix(true_label, prediction_proba, decision_threshold=0.5):
    predict_label = torch.where(prediction_proba > decision_threshold, torch.ones_like(prediction_proba),
                                torch.zeros_like(prediction_proba))

    TP = torch.sum(torch.logical_and(predict_label == 1, true_label == 1)).cpu()
    TN = torch.sum(torch.logical_and(predict_label == 0, true_label == 0)).cpu()
    FP = torch.sum(torch.logical_and(predict_label == 1, true_label == 0)).cpu()
    FN = torch.sum(torch.logical_and(predict_label == 0, true_label == 1)).cpu()

    confusion_matrix = np.asarray([[TP, FP],
                                   [FN, TN]])
    return confusion_matrix


def compute_all_score(confusion_matrix, t=0.5):
    [[TP, FP], [FN, TN]] = confusion_matrix.astype(float)

    precision_positive = TP / (TP + FP) if (TP + FP) != 0 else np.nan
    precision_negative = TN / (TN + FN) if (TN + FN) != 0 else np.nan

    recall_positive = TP / (TP + FN) if (TP + FN) != 0 else np.nan
    recall_negative = TN / (TN + FP) if (TN + FP) != 0 else np.nan

    if (precision_positive + recall_positive) != 0:
        F1_score_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive)
    else:
        F1_score_positive = np.nan
    if (precision_negative + recall_negative) != 0:
        F1_score_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)
    else:
        F1_score_negative = np.nan

    accuracy = (TP + TN) / np.sum(confusion_matrix)
    balanced_acc = (recall_negative + recall_positive) / 2

    return [t, accuracy, balanced_acc, precision_positive, recall_positive, F1_score_positive, precision_negative,
            recall_negative, F1_score_negative]


def config_log(opt, output_dir):
    """
    Set configurations about logging to keep track of training progress.
    """
    log_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info('***** A new training has been started *****')
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    logging.info(opt)
    logging.info('Arg parser: ')
    logging.info('Saving checkpoint model to {:s}'.format(output_dir))