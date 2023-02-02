import os 
import pdb
import math
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

import sys
sys.path.append("..")
from metric.metric import metric

data_folder = '../data/'
np.random.seed(2022)


def parse_args():
    parser = ArgumentParser(description='Logistgic regression with pytorch')
    # model and dataset
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--loss', type=str, default="Focal_loss",
                        choices=['BCELoss', 'Focal_loss'],
                        help="choice loss for train or val in list")
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=int, default=2)
    
    args = parser.parse_args()

    return args


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(Focal_loss, self). __init__()
        self.gamma = gamma
        self.alpha = torch.Tensor([alpha])
        self.size_average = size_average

    def forward(self, logits, label):
        # logits:[b,h,w] label:[b,h,w]
        pred = logits
        pred = pred.view(-1) # b*h*w
        label = label.view(-1)

        if self.alpha:
            self.alpha = self.alpha.type_as(pred.data) 
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label) # b*h*w
            
        pt = pred * label + (1 - pred) * (1-label)
        diff = (1-pt) ** self.gamma

        FL = -1 * alpha_t * diff * pt.log()
        
        if self.size_average: 
            return FL.mean()
        else: 
            return FL.sum()


# class FCN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FCN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 2048)
#         self.fc2 = nn.Linear(2048, 2048)
#         self.fc3 = nn.Linear(2048, 1024)
#         self.fc4 = nn.Linear(1024, 512)
#         self.fc5 = nn.Linear(512, 256)
#         self.fc6 = nn.Linear(256, output_dim)
#         self.dropout = nn.Dropout(p=0.1)
#         self.batchnorm1 = nn.BatchNorm1d(2048)
#         self.batchnorm2 = nn.BatchNorm1d(2048)
#         self.batchnorm3 = nn.BatchNorm1d(1024)
#         self.batchnorm4 = nn.BatchNorm1d(512)
#         self.batchnorm5 = nn.BatchNorm1d(256)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.batchnorm1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.batchnorm2(x)
#         x = torch.relu(self.fc3(x))
#         x = self.batchnorm3(x)
#         x = torch.relu(self.fc4(x))
#         x = self.batchnorm4(x)
#         x = torch.relu(self.fc5(x))
#         x = self.batchnorm5(x)
#         # x = self.dropout(x)
#         x = torch.sigmoid(self.fc6(x))   
        
#         return x


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        # self.fc3 = nn.Linear(2048, 2048)
        # self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, output_dim)
        # self.dropout = nn.Dropout(p=0.1)
        # self.batchnorm1 = nn.BatchNorm1d(2048)
        # self.batchnorm2 = nn.BatchNorm1d(2048)
        # self.batchnorm3 = nn.BatchNorm1d(2048)
        # self.batchnorm4 = nn.BatchNorm1d(2048)
        # self.batchnorm5 = nn.BatchNorm1d(1024)
        # self.batchnorm6 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.batchnorm1(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.batchnorm2(x)
        # x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        # x = self.batchnorm3(x)
        # x = torch.relu(self.fc4(x))
        # x = self.batchnorm4(x)
        x = torch.relu(self.fc5(x))
        # x = self.batchnorm5(x)
        x = torch.relu(self.fc6(x))
        # x = self.batchnorm6(x)
        x = torch.sigmoid(self.fc7(x))   
        
        return x

    
class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.length = self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.length


def split_set(data_to_split, ratio=0.8):
    mask = np.random.rand(len(data_to_split)) < ratio
    return [data_to_split[mask], data_to_split[~mask]]


def compute_confusion_matrix(true_label, prediction_proba, decision_threshold=0.5): 
    
    predict_label = torch.where(prediction_proba>decision_threshold,torch.ones_like(prediction_proba),torch.zeros_like(prediction_proba))
    
    TP = torch.sum(torch.logical_and(predict_label==1, true_label==1)).cpu()
    TN = torch.sum(torch.logical_and(predict_label==0, true_label==0)).cpu()
    FP = torch.sum(torch.logical_and(predict_label==1, true_label==0)).cpu()
    FN = torch.sum(torch.logical_and(predict_label==0, true_label==1)).cpu()
    
    confusion_matrix = np.asarray([[TP, FP],
                                    [FN, TN]])
    return confusion_matrix


def compute_all_score(confusion_matrix, t=0.5):
    
    [[TP, FP],[FN, TN]] = confusion_matrix.astype(float)
    
    precision_positive = TP/(TP+FP) if (TP+FP) !=0 else np.nan
    precision_negative = TN/(TN+FN) if (TN+FN) !=0 else np.nan
    
    recall_positive = TP/(TP+FN) if (TP+FN) !=0 else np.nan
    recall_negative = TN/(TN+FP) if (TN+FP) !=0 else np.nan

    F1_score_positive = 2 *(precision_positive*recall_positive)/(precision_positive+recall_positive) if (precision_positive+recall_positive) !=0 else np.nan
    F1_score_negative = 2 *(precision_negative*recall_negative)/(precision_negative+recall_negative) if (precision_negative+recall_negative) !=0 else np.nan
    
    # pdb.set_trace()
    accuracy =  (TP+TN)/np.sum(confusion_matrix)
    # pdb.set_trace()
    balanced_acc = (recall_negative + recall_positive) / 2

    return [t, accuracy, balanced_acc, precision_positive, recall_positive, F1_score_positive, precision_negative, recall_negative, F1_score_negative]


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



def main(args):

    ckpt_dir = './temporal_spatial/ckpt-{}-{}-{}-lr{}'.format(args.loss, args.alpha, args.gamma, args.lr)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    config_log(args, ckpt_dir)

    # check CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using {device} device")

    # load data with desired columns
    # 8-neighbors 
    columns = ['LU4', 'LC4', 'LU3', 'LC3', 'LU2', 'LC2', 'LU1', 'LC1', 'nbr1_LU3', 'nbr1_LC3', 'nbr1_LU2', 'nbr1_LC2', 'nbr1_LU1',
            'nbr1_LC1', 'nbr2_LU3', 'nbr2_LC3', 'nbr2_LU2', 'nbr2_LC2', 'nbr2_LU1', 'nbr2_LC1', 'nbr3_LU3', 'nbr3_LC3',
            'nbr3_LU2', 'nbr3_LC2', 'nbr3_LU1', 'nbr3_LC1', 'nbr4_LU3', 'nbr4_LC3', 'nbr4_LU2', 'nbr4_LC2', 'nbr4_LU1',
            'nbr4_LC1', 'nbr5_LU3', 'nbr5_LC3', 'nbr5_LU2', 'nbr5_LC2', 'nbr5_LU1', 'nbr5_LC1', 'nbr6_LU3', 'nbr6_LC3',
            'nbr6_LU2', 'nbr6_LC2', 'nbr6_LU1', 'nbr6_LC1', 'nbr7_LU3', 'nbr7_LC3', 'nbr7_LU2', 'nbr7_LC2', 'nbr7_LU1',
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
    data_features['changed'] = [0 if row['LU4'] == row['LU3'] and row['LC4'] == row['LC3'] else 1 for ind, row in data_features[['LU4', 'LC4', 'LU3', 'LC3']].iterrows()]
    data_features.drop(['LC4', 'LU4'], axis=1, inplace=True)
    logging.info('Total number of tiles that changed label in either Land Cover or Land Usage: %d' % sum(data_features.changed))

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

    test_label=test_categorical.changed
    test_features = test_categorical.drop('changed', axis=1)
    logging.info('Length of the test dataset : {}'.format(len(test_label)))

    X_train = torch.Tensor(train_features.values).cuda()
    y_train = torch.Tensor(train_label.values).cuda()

    X_val = torch.Tensor(val_features.values).cuda()
    y_val = torch.Tensor(val_label.values).cuda()

    X_test = torch.Tensor(test_features.values).cuda()
    y_test = torch.Tensor(test_label.values).cuda()

    trainset = dataset(X_train, y_train)
    
    # define loss function, respectively
    # Default uses cross quotient loss function
    if args.loss == 'BCELoss':
        criterion = nn.BCELoss()
    elif args.loss == 'Focal_loss':
        criterion = Focal_loss(alpha=args.alpha, gamma=args.gamma)

    epochs = args.epoch
    input_dim = len(train_features.columns)
    output_dim = 1 
    learning_rate = args.lr
    batch_size = 1024

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    model = FCN(input_dim, output_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15, 30], gamma=0.5)


    losses = []
    losses_val = []
    mat_ls = []
    recall_p_ls = []
    balanced_acc_ls = []

    for i in tqdm(range(int(epochs)),desc='Training Epochs'):
        for idx, (x_batch, y_batch) in enumerate(trainloader):
            labels = y_batch
            outputs = torch.squeeze(model(x_batch))
            # outputs = torch.squeeze(torch.where(proba>0.5,torch.ones_like(proba),torch.zeros_like(proba)))
            loss = criterion(outputs, labels)

            loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 

            optimizer.step() # Updates weights and biases with the optimizer (SGD)
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        scheduler.step()
            
        if i % 1 == 0:
            with torch.no_grad():
                # Calculating the loss and accuracy for the validation dataset
                outputs_val = torch.squeeze(model(X_val))
                df_change = pd.DataFrame(val_label)
                df_change['proba_change'] = outputs_val.cpu().detach().numpy()
                df_change.reset_index(inplace=True)
                weighted_metric = metric(mode='binary', PROBABILITY_THRESHOLD=0.5, print_log=False, change_pred=df_change)[-1]
                # outputs_val = torch.where(proba_val>0.5,torch.ones_like(proba_val),torch.zeros_like(proba_val))
                loss_val = criterion(outputs_val, y_val)
                # pdb.set_trace()
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
                # outputs_train = torch.where(proba_train>0.5,torch.ones_like(proba_train),torch.zeros_like(proba_train))
                loss_train = criterion(outputs_train, y_train)
                losses.append(loss_train.item())

                logging.info(f"Epoch: {i}. \nValidation - Loss: {loss_val.item()} \t balanced_acc: {balanced_acc_val} \t postive recall: {recall_p_val} \t weighted_metric: {weighted_metric}")
                logging.info(f"Train -  Loss: {loss_train.item()} \t learning rate: {lr}\n")
        

        if i % 1 == 0 and i != 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ckpt-{}.pth'.format(i)))


if __name__ == '__main__':
    args = parse_args()
    main(args)