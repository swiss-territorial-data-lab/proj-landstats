import os
import sys
import pdb
import torch
import pickle
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
from metric.metric import metric
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import FocalLoss, FCN, Dataset, split_set, config_log

sys.path.append("..")

data_folder = '../data/'
np.random.seed(2022)


def parse_args():
    parser = ArgumentParser(description='integration module with pytorch')
    # model and dataset
    parser.add_argument('--base', type=str, default="lr",
                        choices=['lr', 'fcn'],
                        help="Name for temporal or spatial module")
    parser.add_argument('--base_path', type=str, default=None,
                        help="Path for temporal or spatial module")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--OUT_DIR', type=str, default='./ckpt')
    parser.add_argument('--milestones', type=int, nargs="+", default=[60, 120],
                        help="milestone epoch for learning rate to decrease by 0.2")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--nbr_val', type=int, default=1,
                        help="Every # of epoch to validate model performance")
    parser.add_argument('--nbr_save', type=int, default=1,
                        help="Every # of epoch to save the model weights")

    args = parser.parse_args()

    return args


def main(args):
    ckpt_dir = '/integrate-{}-fcn/ckpt-{}-{}-{}-{}'.format(args.base, args.alpha, args.gamma, args.lr, args.batch_size)
    ckpt_dir = os.path.join(args.OUT_DIR, ckpt_dir)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    config_log(args, ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using {device} device")

    # Load temporal and spatial module features
    if args.base_path is None:
        if args.base == 'lr':
            pkl_filename = "./temp_results/temporal_spatial_proba_lr.pkl"

        elif args.base == 'fcn':
            pkl_filename = "./temp_results/temporal_spatial_proba_FCN.pkl"

    elif args.base_path:
        pkl_filename = args.base_path

    else:
        raise RuntimeError('No temporal and spatial feature input')

    with open(pkl_filename, 'rb') as file:
        df_proba = pickle.load(file)

    df_proba[['LU3']] = [[int(c[0].lstrip('LU'))] for c in df_proba[['LU3']].values]
    df_proba[['LC3']] = [[int(c[0].lstrip('LC'))] for c in df_proba[['LC3']].values]

    # Load image classification results
    pred_lc = pd.read_csv(os.path.join(data_folder, 'predictions_lc_area4.csv'), index_col=0)
    pred_lc.rename(columns={"prediction": "prediction_lc", "confidence": "confidence_lc"}, inplace=True)
    pred_lu = pd.read_csv(os.path.join(data_folder, 'predictions_lu_area4.csv'), index_col=0)
    pred_lu.rename(columns={"prediction": "prediction_lu", "confidence": "confidence_lu"}, inplace=True)
    pred = pd.concat([pred_lc, pred_lu], axis=1)

    data_merged = pred.merge(df_proba, on="RELI")

    train, test = split_set(data_merged)
    train, val = split_set(train)

    train_label = train.changed
    train_features = train.drop('changed', axis=1)
    logging.info('Length of the train dataset : {}'.format(len(train)))

    val_label = val.changed
    val_features = val.drop('changed', axis=1)
    logging.info('Length of the val dataset : {}'.format(len(val)))

    X_train = torch.Tensor(train_features.values).to(device)
    y_train = torch.Tensor(train_label.values).to(device)

    X_val = torch.Tensor(val_features.values).to(device)
    y_val = torch.Tensor(val_label.values).to(device)

    train_set = Dataset(X_train, y_train)

    # define loss function
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    epochs = args.epoch
    input_dim = len(train_features.columns)
    output_dim = 1
    learning_rate = args.lr
    batch_size = args.batch_size

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    model = FCN(input_dim, output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

    losses = []
    best_w_metric = 0
    best_epoch = 0

    for i in tqdm(range(int(epochs)), desc='Training Epochs'):
        for idx, (x_batch, y_batch) in enumerate(train_loader):
            labels = y_batch
            outputs = torch.squeeze(model(x_batch))
            # outputs = torch.squeeze(torch.where(proba>0.5,torch.ones_like(proba),torch.zeros_like(proba)))
            loss = criterion(outputs, labels)

            loss.backward()  # Computes the gradient of the given tensor w.r.t. graph leaves

            optimizer.step()  # Updates weights and biases with the optimizer (SGD)
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        scheduler.step()

        if i % args.nbr_val == 0:
            with torch.no_grad():
                # Calculating the loss and accuracy for the validation dataset
                outputs_val = torch.squeeze(model(X_val))
                df_change = pd.DataFrame(val_label)
                df_change['proba_change'] = outputs_val.cpu().detach().numpy()
                df_change.reset_index(inplace=True)
                metrics = metric(mode='multi', PROBABILITY_THRESHOLD=0.5, change_pred=df_change,
                                 lc_pred=pred_lc.reset_index(), lu_pred=pred_lu.reset_index(), print_log=False)
                weighted_metric = metrics[-1]
                balanced_acc_val = metrics[1]
                recall_p_val = metrics[2]

                if weighted_metric > best_w_metric:
                    best_w_metric = weighted_metric
                    best_epoch = i

                # Calculating the loss and accuracy for the train dataset
                lr = optimizer.param_groups[0]['lr']
                outputs_train = torch.squeeze(model(X_train))
                loss_train = criterion(outputs_train, y_train)
                loss_val = criterion(outputs_val, y_val)
                losses.append(loss_train.item())

                logging.info(
                    f"Epoch: {i}. \nValidation - Loss: {loss_val.item()} \t balanced_acc: {balanced_acc_val} \t "
                    f"postive recall: {recall_p_val} \t weighted_metric: {weighted_metric}")
                logging.info(
                    f"Train -  Loss: {loss_train.item()} \t learning rate: {lr}\t Best epoch: {best_epoch} \t Best "
                    f"metric: {best_w_metric}")

        if i % args.nbr_save == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ckpt-{}.pth'.format(i)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
