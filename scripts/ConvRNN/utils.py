import os
import pdb
import sys
import numpy as np
import torch
import logging
import collections

from torch import nn
from itertools import repeat
from torch.utils.data import Dataset, DataLoader

""" Copied from torch.nn.modules.utils """


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(Focal_loss, self).__init__()
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


class LandstatsData(Dataset):
    def __init__(self, dir_path, num_survey=3):
        # Read the file
        self.dir_path = dir_path
        self.file_ls = os.listdir(dir_path)
        self.num_survey = num_survey
        # Calculate len
        self.data_len = len(self.file_ls)

    def __getitem__(self, idx):
        # Get data with .npy file
        data_path = os.path.join(self.dir_path, self.file_ls[idx])
        data_tensor = torch.Tensor(np.load(data_path)[:self.num_survey, :, :, :])
        # read semantic label
        label = data_path.split('/')[-1].split('-')[0]
        RELI = int(data_path.split('/')[-1].split('-')[1].split('.')[0])

        # Transform to tensor
        return data_tensor, label, RELI

    def __len__(self):
        return self.data_len


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

    F1_score_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive) if (
                                                                                                                           precision_positive + recall_positive) != 0 else np.nan
    F1_score_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative) if (
                                                                                                                           precision_negative + recall_negative) != 0 else np.nan

    # pdb.set_trace()
    accuracy = (TP + TN) / np.sum(confusion_matrix)
    # pdb.set_trace()
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