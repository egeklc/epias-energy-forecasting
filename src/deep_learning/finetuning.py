import torch
import torch.nn as nn
import optuna

from src.deep_learning.lstm_model import LSTMModel
from src.deep_learning.training import train_epoch, evaluate_model

def objective(trial, train_loader, val_loader, input_size, horizon, device, num_epochs):
    
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1,3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lin_features = trial.suggest_int("lin_features", 16, 192, step=16)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    model = LSTMModel(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    horizon=horizon,
                    lin_features=lin_features,
                    dropout=dropout
                    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3, factor=0.5)
    
    criterion = nn.HuberLoss()
    
    patience = 7
    counter = 0
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        train_epoch(model=model, device=device, dataloader=train_loader, criterion=criterion, optimizer=optimizer)
        val_loss, _, _ = evaluate_model(model=model, device=device, dataloader=val_loader, criterion=criterion)
        lr_scheduler.step(val_loss)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            break
        
    return best_val_loss