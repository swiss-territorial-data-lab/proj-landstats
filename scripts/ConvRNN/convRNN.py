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
                weighted_metric = metric(mode='binary', PROBABILITY_THRESHOLD=0.5, print_log=False, change_pred=df_change)[-1]
                # outputs_val = torch.where(proba_val>0.5,torch.ones_like(proba_val),torch.zeros_like(proba_val))
                loss_val = criterion(torch.Tensor(outputs_val), torch.Tensor(labels_val))
                # pdb.set_trace()
                mat_val = compute_confusion_matrix(torch.Tensor(labels_val), torch.Tensor(outputs_val))
                scores_val = compute_all_score(mat_val)
                balanced_acc_val = scores_val[2]
                recall_p_val = scores_val[4]

                # Calculating the loss and accuracy for the train dataset
                lr = optimizer.param_groups[0]['lr']
                loss_train = np.mean(losses_train)

                logging.info(f"Epoch: {i}. \nValidation - Loss: {loss_val.item()} \t balanced_acc: {balanced_acc_val} \t postive recall: {recall_p_val} \t weighted_metric: {weighted_metric}")
                logging.info(f"Train -  Loss: {loss_train.item()} \t learning rate: {lr}\n")
        

        if i % 1 == 0 and i != 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'ckpt-{}.pth'.format(i)))


if __name__ == '__main__':
    args = parse_args()
    main(args)