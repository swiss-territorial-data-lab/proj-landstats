import os
import pdb
import sys
import torch
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
from module import Conv2dRNN
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from utils import config_log, LandstatsData

CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]
config_path = CURRENT_DIR.rsplit('/', 2)[0]
sys.path.append(config_path)

from metric.metric import metric

data_folder = '../data/'
np.random.seed(2022)


def parse_args():
    parser = ArgumentParser(description='convolutional RNN with pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='/home/shanci/porj-landstats/ckpt/ckpt-45.pth')
    parser.add_argument('--t', type=float, default=0.5)

    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--gamma', type=int, default=3)
    args = parser.parse_args()

    return args


def main(args):
    ckpt_dir = './ckpt/'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    config_log(args, ckpt_dir)

    # check CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using {device} device")

    # check model existence
    assert os.path.exists(args.model)

    batch_size = 1024

    model = Conv2dRNN(in_channels=2, out_channels=128, kernel_size=(3, 3), num_layers=20, batch_first=True).to(device)
    model.load_state_dict(torch.load(args.model), strict=True)

    # Data loading
    test_set = LandstatsData('./data/convRNN/test/', num_survey=3)
    testLoader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=True, num_workers=int(os.cpu_count() / 2), pin_memory=True, drop_last=False)

    with torch.no_grad():
        labels_test = []
        outputs_test = []
        ind_test = []
        # Calculating the loss and accuracy for the validation dataset
        for idx, (x_batch, labels, RELI) in enumerate(testLoader):
            x_batch = x_batch.to(device)
            outputs, _ = model(x_batch, None)
            outputs = outputs.cpu().detach().numpy()
            labels_test.extend(np.array(labels).astype(int))
            ind_test.extend(RELI.tolist())
            outputs_test.extend(outputs)
            # torch.cuda.empty_cache()

        df_change = pd.DataFrame({'changed': labels_test,
                                  'RELI': ind_test})
        df_change['proba_change'] = np.array(outputs_test)
        # df_change.index.name='RELI'
        # metric(mode='binary', PROBABILITY_THRESHOLD=args.t, print_log=True, change_pred=df_change)

    threshold = np.linspace(0, 1, 101)
    columns_score_name = ['Threshold', 'balanced_acc', 'true_pos_rate', 'true_neg_rate', 'miss_changes',
                          'miss_changed_ratio', 'miss_weighted_changes', 'miss_weighted_changed_ratio',
                          'automatized_points', 'automatized_capacity', 'raw_metric', 'weighted_metric']
    threshold_score = pd.concat([pd.DataFrame([metric(mode='binary', PROBABILITY_THRESHOLD=t, change_pred=df_change,
                                                      print_log=False)], columns=columns_score_name) for t in threshold]
                                , ignore_index=True)
    threshold_score.set_index('Threshold', inplace=True)

    print(threshold_score.iloc[threshold_score['weighted_metric'].argmax()])


if __name__ == '__main__':
    args = parse_args()
    main(args)
