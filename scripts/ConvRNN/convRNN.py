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
from torch.optim.lr_scheduler import MultiStepLR
from utils import config_log, compute_all_score, compute_confusion_matrix, Focal_loss, LandstatsData

CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0] 
config_path = CURRENT_DIR.rsplit('/', 2)[0]  
sys.path.append(config_path)

from metric.metric import metric

data_folder = '../data/'
np.random.seed(2022)


def parse_args():
    parser = ArgumentParser(description='convolutional RNN with pytorch')
    # model and dataset
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--loss', type=str, default="Focal_loss",
                        choices=['BCELoss', 'Focal_loss'],
                        help="choice loss for train or val in list")
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
    

    epochs = args.epoch
    learning_rate = args.lr
    batch_size = 1024

    model = Conv2dRNN(in_channels=2, out_channels=128, kernel_size=(3,3), num_layers=20, batch_first=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = MultiStepLR(optimizer, milestones=[10, 15, 30], gamma=0.5)

    # Data loading
    train_set = LandstatsData('./data/convRNN/train/', num_survey=3)
    val_set = LandstatsData('./data/convRNN/val/', num_survey=3)
    trainLoader = DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=int(os.cpu_count()/2), pin_memory=True, drop_last=False)
    valLoader = DataLoader(val_set, batch_size=batch_size,
                                    shuffle=True, num_workers=int(os.cpu_count()/2), pin_memory=True, drop_last=False)
    
    # define loss function, respectively
    # Default uses cross quotient loss function
    if args.loss == 'BCELoss':
        criterion = nn.BCELoss()
    elif args.loss == 'Focal_loss':
        criterion = Focal_loss(alpha=args.alpha, gamma=args.gamma)
    

    for i in tqdm(range(int(epochs)),desc='Training Epochs'):
        losses_train = []
        for idx, (x_batch, labels, RELI) in enumerate(trainLoader):
            x_batch = x_batch.to(device)
            outputs, _ = model(x_batch, None)
            # outputs = torch.squeeze(torch.where(proba>0.5,torch.ones_like(proba),torch.zeros_like(proba)))

            # calculate training loss
            labels = torch.Tensor(np.array(labels).astype(int)).to(device)
            loss = criterion(outputs, labels)
            losses_train.append(loss.cpu().detach().numpy())
            loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 

            optimizer.step() # Updates weights and biases with the optimizer (SGD)
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # scheduler.step()

        # validation
        if i % 1 == 0:
            with torch.no_grad():
                labels_val = []
                outputs_val = []
                ind_val = []
                # Calculating the loss and accuracy for the validation dataset
                for idx, (x_batch, labels, RELI) in enumerate(valLoader):
                    x_batch = x_batch.to(device)
                    outputs, _ = model(x_batch, None)
                    outputs = outputs.cpu().detach().numpy()
                    labels_val.extend(np.array(labels).astype(int))
                    ind_val.extend(RELI.tolist())
                    outputs_val.extend(outputs)
                    # torch.cuda.empty_cache()

                df_change = pd.DataFrame({'changed': labels_val, 
                                            'RELI': ind_val})
                df_change['proba_change'] = np.array(outputs_val)
                # df_change.index.name='RELI'

                threshold = np.linspace(0, 1, 101)
                columns_score_name = ['Threshold', 'balanced_acc', 'true_pos_rate', 'true_neg_rate', 'miss_changes',
                                      'miss_changed_ratio', 'miss_weighted_changes', 'miss_weighted_changed_ratio',
                                      'automatized_points', 'automatized_capacity', 'raw_metric', 'weighted_metric']
                threshold_score = pd.concat(
                    [pd.DataFrame([metric(mode='binary', PROBABILITY_THRESHOLD=t, change_pred=df_change,
                                          print_log=False)], columns=columns_score_name) for t in threshold]
                    , ignore_index=True)
                threshold_score.set_index('Threshold', inplace=True)

                t_best = threshold_score['weighted_metric'].argmax()

                weighted_metric = threshold_score.iloc[t_best]['weighted_metric']
                balanced_acc_val = threshold_score.iloc[t_best]['balanced_acc']
                recall_p_val = threshold_score.iloc[t_best]['true_pos_rate']

                # outputs_val = torch.where(proba_val>0.5,torch.ones_like(proba_val),torch.zeros_like(proba_val))
                loss_val = criterion(torch.Tensor(outputs_val), torch.Tensor(labels_val))

                # Calculating the loss and accuracy for the train dataset
                lr = optimizer.param_groups[0]['lr']
                loss_train = np.mean(losses_train)

                logging.info(f"Epoch: {i}. \nValidation - Loss: {loss_val.item()} \t balanced_acc: {balanced_acc_val}"
                             f" \t postive recall: {recall_p_val} \t weighted_metric: {weighted_metric}")
                logging.info(f"Train -  Loss: {loss_train.item()} \t learning rate: {lr}\n")
        

        if i % 1 == 0 and i != 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ckpt-{}.pth'.format(i)))


if __name__ == '__main__':
    args = parse_args()
    main(args)