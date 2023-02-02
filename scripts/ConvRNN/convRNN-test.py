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

    logging.info('***** A new test has been started *****')
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    logging.info(opt)
    logging.info('Arg parser: ')


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

    model = Conv2dRNN(in_channels=2, out_channels=128, kernel_size=(3,3), num_layers=20, batch_first=True).to(device)
    model.load_state_dict(torch.load(args.model), strict=True)

    # Data loading
    test_set = LandstatsData('./data/convRNN/test/', num_survey=3)
    testLoader = DataLoader(test_set, batch_size=batch_size,
                                shuffle=True, num_workers=int(os.cpu_count()/2), pin_memory=True, drop_last=False)


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
    columns_score_name = ['Threshold', 'balanced_acc', 'true_pos_rate', 'true_neg_rate', 'miss_changes', 'miss_changed_ratio', 'miss_weighted_changes', 'miss_weighted_changed_ratio', 
            'automatized_points', 'automatized_capacity', 'raw_metric', 'weighted_metric']
    threshold_score = pd.concat([pd.DataFrame([metric(mode='binary', PROBABILITY_THRESHOLD=t, change_pred=df_change, print_log=False)], columns=columns_score_name) for t in threshold],
                            ignore_index=True)
    threshold_score.set_index('Threshold', inplace=True)

    print(threshold_score.iloc[threshold_score['weighted_metric'].argmax()])


if __name__ == '__main__':
    args = parse_args()
    main(args)