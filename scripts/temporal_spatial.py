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
from utils import compute_confusion_matrix, compute_all_score

sys.path.append("..")

data_folder = '../data/'
np.random.seed(2022)


def parse_args():
    parser = ArgumentParser(description='Logistgic regression with pytorch')
    # model and dataset
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--OUT_DIR', type=str, default='./ckpt')
    parser.add_argument('--milestones', type=int, nargs="+", default=[10, 15, 30],
                        help="milestone epoch for learning rate to decrease by 0.2")
    parser.add_argument('--loss', type=str, default="Focal_loss",
                        choices=['BCELoss', 'Focal_loss'],
                        help="choice loss for train or val in list")
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--nbr_val', type=int, default=1,
                        help="Every # of epoch to validate model performance")
    parser.add_argument('--nbr_save', type=int, default=1,
                        help="Every # of epoch to save the model weights")

    args = parser.parse_args()

    return args


def main(args):
    # config output dir
    ckpt_dir = './temporal_spatial/ckpt-{}-{}-{}-lr{}'.format(args.loss, args.alpha, args.gamma, args.lr)
    ckpt_dir = os.path.join(args.OUT_DIR, ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    config_log(args, ckpt_dir)

    # check CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using {device} device")

    # load data with desired columns
    # 8-neighbors 
    columns = ['LU4', 'LC4', 'LU3', 'LC3', 'LU2', 'LC2', 'LU1', 'LC1', 'nbr1_LU3', 'nbr1_LC3', 'nbr1_LU2', 'nbr1_LC2',
               'nbr1_LU1',
               'nbr1_LC1', 'nbr2_LU3', 'nbr2_LC3', 'nbr2_LU2', 'nbr2_LC2', 'nbr2_LU1', 'nbr2_LC1', 'nbr3_LU3',
               'nbr3_LC3',
               'nbr3_LU2', 'nbr3_LC2', 'nbr3_LU1', 'nbr3_LC1', 'nbr4_LU3', 'nbr4_LC3', 'nbr4_LU2', 'nbr4_LC2',
               'nbr4_LU1',
               'nbr4_LC1', 'nbr5_LU3', 'nbr5_LC3', 'nbr5_LU2', 'nbr5_LC2', 'nbr5_LU1', 'nbr5_LC1', 'nbr6_LU3',
               'nbr6_LC3',
               'nbr6_LU2', 'nbr6_LC2', 'nbr6_LU1', 'nbr6_LC1', 'nbr7_LU3', 'nbr7_LC3', 'nbr7_LU2', 'nbr7_LC2',
               'nbr7_LU1',
               'nbr7_LC1', 'nbr8_LU3', 'nbr8_LC3', 'nbr8_LU2', 'nbr8_LC2', 'nbr8_LU1', 'nbr8_LC1']

    # 4-neighbors 
    # columns = ['LU4', 'LC4', 'LU3', 'LC3', 'LU2', 'LC2', 'LU1', 'LC1', 'nbr1_LU3', 'nbr1_LC3', 'nbr1_LU2', 'nbr1_LC2', 'nbr1_LU1',
    #        'nbr1_LC1', 'nbr2_LU3', 'nbr2_LC3', 'nbr2_LU2', 'nbr2_LC2', 'nbr2_LU1', 'nbr2_LC1', 'nbr3_LU3', 'nbr3_LC3',
    #        'nbr3_LU2', 'nbr3_LC2', 'nbr3_LU1', 'nbr3_LC1', 'nbr4_LU3', 'nbr4_LC3', 'nbr4_LU2', 'nbr4_LC2', 'nbr4_LU1',
    #        'nbr4_LC1']

    # Time-deactivation
    # columns = ['LU4', 'LC4', 'LU3', 'LC3', 'nbr1_LU3', 'nbr1_LC3', 'nbr2_LU3', 'nbr2_LC3', 'nbr3_LU3', 'nbr3_LC3',
    #        'nbr4_LU3', 'nbr4_LC3', 'nbr5_LU3', 'nbr5_LC3', 'nbr6_LU3', 'nbr6_LC3', 'nbr7_LU3', 'nbr7_LC3',
    #        'nbr8_LU3', 'nbr8_LC3']

    # Space-deactivation
    # columns = ['LU4', 'LC4', 'LU3', 'LC3', 'LU2', 'LC2', 'LU1', 'LC1']

    # Data loading
    original_data = pd.read_csv(os.path.join(data_folder, 'trainset_with_neighbour.csv'), index_col=0)
    original_data = original_data[columns]
    original_data.dropna(inplace=True)
    logging.info('The length of the data without the rows with nan value is: {}'.format(len(original_data)))

    data_features = original_data.copy()
    data_features['changed'] = [0 if row['LU4'] == row['LU3'] and row['LC4'] == row['LC3'] else 1 for ind, row in
                                data_features[['LU4', 'LC4', 'LU3', 'LC3']].iterrows()]
    data_features.drop(['LC4', 'LU4'], axis=1, inplace=True)
    logging.info(
        'Total number of tiles that changed label in either Land Cover or Land Usage: %d' % sum(data_features.changed))

    train, test = split_set(data_features)
    train, val = split_set(train)

    # one-hot encoder to make categorical features as columns with 0-1
    # Make sure we use only the features available in the training set
    train_categorical = pd.get_dummies(train)
    val_categorical = pd.get_dummies(val)[train_categorical.columns]
    test_categorical = pd.get_dummies(test)[train_categorical.columns]

    train_label = train_categorical.changed
    train_features = train_categorical.drop('changed', axis=1)
    logging.info('Length of the train dataset : {}'.format(len(train_label)))

    val_label = val_categorical.changed
    val_features = val_categorical.drop('changed', axis=1)
    logging.info('Length of the validation dataset : {}'.format(len(val_label)))

    # Transform data to torch.Tensor and transmit to GPU
    X_train = torch.Tensor(train_features.values).cuda()
    y_train = torch.Tensor(train_label.values).cuda()

    X_val = torch.Tensor(val_features.values).cuda()
    y_val = torch.Tensor(val_label.values).cuda()

    train_set = Dataset(X_train, y_train)

    # define loss function, respectively
    # Default uses cross quotient loss function
    if args.loss == 'BCELoss':
        criterion = nn.BCELoss()
    elif args.loss == 'Focal_loss':
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    epochs = args.epoch
    input_dim = len(train_features.columns)
    output_dim = 1
    learning_rate = args.lr
    batch_size = args.batch_size

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    model = FCN(input_dim, output_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

    losses = []
    losses_val = []
    mat_ls = []
    recall_p_ls = []
    balanced_acc_ls = []

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
                weighted_metric = \
                    metric(mode='binary', PROBABILITY_THRESHOLD=0.5, print_log=False, change_pred=df_change)[-1]
                # outputs_val = torch.where(proba_val>0.5,torch.ones_like(proba_val),torch.zeros_like(proba_val))
                loss_val = criterion(outputs_val, y_val)
                mat_val = compute_confusion_matrix(y_val, outputs_val)
                scores_val = compute_all_score(mat_val)
                balanced_acc_val = scores_val[2]
                recall_p_val = scores_val[4]
                losses_val.append(loss_val.item())
                mat_ls.append(mat_val)
                recall_p_ls.append(recall_p_val)
                balanced_acc_ls.append(balanced_acc_val)

                # Calculating the loss and accuracy for the train dataset
                lr = optimizer.param_groups[0]['lr']
                outputs_train = torch.squeeze(model(X_train))
                loss_train = criterion(outputs_train, y_train)
                losses.append(loss_train.item())

                logging.info(
                    f"Epoch: {i}. \nValidation - Loss: {loss_val.item()} \t balanced_acc: {balanced_acc_val} \t "
                    f"postive recall: {recall_p_val} \t weighted_metric: {weighted_metric}")
                logging.info(f"Train -  Loss: {loss_train.item()} \t learning rate: {lr}\n")

        if i % args.nbr_save == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ckpt-{}.pth'.format(i)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
