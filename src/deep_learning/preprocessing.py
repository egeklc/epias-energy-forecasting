
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def scale_df(df, train:bool, scaler=None, scale_cols=None,
                x_scaler=None, y_scaler=None, 
                x_scale_cols=None, y_scale_cols=None):
    """
    Scale selected columns of a dataframe using single or separate scalers for features and target.
    """
    df_scaled = df.copy()
    
    single_scale = scaler is not None and scale_cols is not None
    x_y_scale = all( c is not None for c in  [x_scaler, y_scaler, x_scale_cols, y_scale_cols])
    
    
    if single_scale:
        if train == True:
            df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
        else:
            df_scaled[scale_cols] = scaler.transform(df[scale_cols])
            
    if x_y_scale:
        if train:
            df_scaled[x_scale_cols] = x_scaler.fit_transform(df[x_scale_cols])
            df_scaled[y_scale_cols]  = y_scaler.fit_transform(df[y_scale_cols])
        else:
            df_scaled[x_scale_cols] = x_scaler.transform(df[x_scale_cols])
            df_scaled[y_scale_cols]  = y_scaler.transform(df[y_scale_cols])
    return df_scaled

class TimeSeriesDataset(Dataset):
    """
    PyTorch dataset for generating sequences for time-series forecasting.
    """
    def __init__(self, df, feature_cols, target_col, window_size, horizon):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.float32)
        self.window_size = window_size
        self.horizon = horizon
        self.len_df = len(df)
        
    def __len__(self):
        return self.len_df - self.window_size - self.horizon + 1
    
    
    def __getitem__(self, idx):
        X = self.X[idx: idx + self.window_size]
        y = self.y[idx + self.window_size: idx + self.window_size + self.horizon]
        return X, y