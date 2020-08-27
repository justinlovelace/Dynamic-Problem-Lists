import json
import logging
import os
import shutil
import numpy as np
import sklearn
import pandas as pd
from copy import copy
import torch
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
from inspect import signature


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class OutputAUC():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.y_true_counts = {}

    def update(self, y_true, y_pred):
        # for i in range(len(y_true)):
        #     if i not in self.y_true_counts:
        #         self.y_true_counts[i] = y_true[i]
        #     self.y_true_counts[i] += y_true[i]

        for row in y_true:
            self.y_true.append(row)

        for row in y_pred:
            self.y_pred.append(row)

    def get(self):
        return self.y_pred, self.y_true

    def __call__(self):
        # print(self.y_pred)
        # print(self.y_pred[0].shape)
        # print(self.y_true)
        # print(self.y_true[0].shape)
        y_true_arr = np.asarray(self.y_true, dtype=np.long)
        bad_cols = np.where(~y_true_arr.any(axis=0))[0]
        y_true_arr = np.delete(y_true_arr, bad_cols, axis=1)

        y_pred_arr = np.asarray(self.y_pred)
        y_pred_arr = np.delete(y_pred_arr, bad_cols, axis=1)

        # print(y_true_arr.shape)
        # print(y_pred_arr.shape)
        return sklearn.metrics.roc_auc_score(y_true_arr, y_pred_arr,
                                             average='micro')

class MetricsICD():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.y_true_counts = {}

    def update(self, y_true, y_pred):
        # for i in range(len(y_true)):
        #     if i not in self.y_true_counts:
        #         self.y_true_counts[i] = y_true[i]
        #     self.y_true_counts[i] += y_true[i]

        for row in y_true:
            self.y_true.append(row)

        for row in y_pred:
            self.y_pred.append(row)

    def get(self):
        return self.y_pred, self.y_true

    def macro_auc(self):
        y_true_arr = np.asarray(self.y_true, dtype=np.long)
        bad_cols = np.where(~y_true_arr.any(axis=0))[0]
        y_true_arr = np.delete(y_true_arr, bad_cols, axis=1)

        y_pred_arr = np.asarray(self.y_pred)
        y_pred_arr = np.delete(y_pred_arr, bad_cols, axis=1)

        return sklearn.metrics.roc_auc_score(y_true_arr, y_pred_arr,
                                             average='macro')

    def __call__(self):
        # print(len(self.y_pred))
        # print(len(self.y_true))
        y_true_arr = np.asarray(self.y_true, dtype=np.long)
        bad_cols = np.where(~y_true_arr.any(axis=0))[0]
        y_true_arr = np.delete(y_true_arr, bad_cols, axis=1)

        y_pred_arr = np.asarray(self.y_pred)
        y_pred_arr = np.delete(y_pred_arr, bad_cols, axis=1)

        return sklearn.metrics.roc_auc_score(y_true_arr, y_pred_arr,
                                             average='micro')


class TestMetrics():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def plot_pr(self):
        print('Creating PR plot...')
        y_true_arr = np.asarray(self.y_true, dtype=np.long)

        y_pred_arr = np.asarray(self.y_pred)
        print('Calculating metrics...')
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true_arr, y_pred_arr)
        print('Filling plot...')
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
            sklearn.metrics.auc(recall, precision)))
        plt.show()


    def update(self, y_true, y_pred):
        for row in y_true:
            self.y_true.append(row)

        for row in y_pred:
            self.y_pred.append(row)

    def get(self):
        return self.y_pred, self.y_true

    def __call__(self):
        y_true_arr = np.asarray(self.y_true, dtype=np.long)

        y_pred_arr = np.asarray(self.y_pred)

        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true_arr, y_pred_arr)

        return sklearn.metrics.roc_auc_score(y_true_arr, y_pred_arr,
                                             average=None), sklearn.metrics.auc(recall, precision)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    newdict = {key: d[key] for key in ['AUCROC']}
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        newdict = {k: v.tolist() for k, v in newdict.items()}
        json.dump(newdict, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, icd=False):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    if icd:
        filepath = os.path.join(checkpoint, 'icd_last.pth.tar')
    else:
        filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        if icd:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'icd_best.pth.tar'))
        else:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None, parallel=True):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        print("File doesn't exist {}".format(checkpoint))
        sys.exit()
    checkpoint = torch.load(checkpoint)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    state_dict = checkpoint['state_dict']
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def drop_column(df, column_list):
    cols_to_drop = []
    for col in column_list:
        if col in df.columns:
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop)
    return df
