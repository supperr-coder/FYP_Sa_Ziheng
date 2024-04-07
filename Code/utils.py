from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold

import pandas as pd

def load_and_partition_data(data: pd.DataFrame, seq_length: int = 29) -> tuple[np.ndarray, int]:

    sequences = []
    for i in range(len(data)-seq_length+1):
        sequences.append(data[i:(i+seq_length)])

    return np.array(sequences)


def split_sequence(
    sequence: np.ndarray, ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    src_end = int(sequence.shape[1] * ratio)
    # [bs, src_seq_len, num_features]
    src = sequence[:, :src_end]
    # [bs, tgt_seq_len, num_features]
    tgt = sequence[:, src_end - 1 : -1]
    # [bs, tgt_seq_len, num_features]
    tgt_y = sequence[:, src_end:]

    return src.to(torch.float32), tgt.to(torch.float32), tgt_y.to(torch.float32)

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def sliding_windows_mutli_features(data, seq_length):
    x = []
    y = []

    for i in range((data.shape[0])-seq_length):
        _x = data[i:(i+seq_length),:] ## 2 columns for features  
        _y = data[i+seq_length,0] ## column 0 contains the labbel
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y).reshape(-1,1)

def sliding_windows_mutli_features_2(data, seq_length):
    x = []
    y = []

    for i in range((data.shape[0])-seq_length):
        _x = data[i:(i+seq_length),:] ## 2 columns for features  
        _y = data[i+seq_length,:] ## column 0 contains the labbel
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def generate_cv_folds(train_data, folds):
    # YOUR CODE HERE
    cv = KFold(n_splits=folds, shuffle=True, random_state=1345)
        
    train_, val_ = [], []

    
    for train_idx, val_idx in cv.split(train_data):
        train_set = Subset(train_data,train_idx)
        val_set = Subset(train_data,val_idx)
        
        train_.append(train_set)
        val_.append(val_set)

    return train_, val_

class EarlyStopper():
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, model, path_name):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model, path_name)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float)
        self.y = torch.tensor(np.array(y), dtype=torch.float)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]