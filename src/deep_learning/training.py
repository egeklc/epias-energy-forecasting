import torch
import torch.nn as nn

def train_epoch(model, device, dataloader, criterion, optimizer):
    
    model.train()
    running_loss = 0
    
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
    

def evaluate_model(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            running_loss += loss.item() * X.size(0)
            
            all_preds.append(y_pred.cpu())
            all_targets.append(y.cpu())
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, all_preds, all_targets


def train_model(model, device, train_loader, val_loader,
                criterion, optimizer, lr_scheduler,
                num_epochs, save_path,
                min_delta=None, patience=None):
    
    train_total_loss = []
    val_total_loss = []
    
    counter = 0
    best_val_loss = float("inf")
    
    early_stopping = min_delta is not None and patience is not None
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model=model, 
                                device=device,
                                dataloader=train_loader,
                                criterion=criterion,
                                optimizer=optimizer)
        
        val_loss, _, _ = evaluate_model(model=model, 
                                device=device,
                                dataloader=val_loader,
                                criterion=criterion)
        
        lr_scheduler.step(val_loss)
        
        train_total_loss.append(train_loss)
        val_total_loss.append(val_loss)
        
        print(f"Epoch [{epoch+1:03d}/{num_epochs:03d}]\n"
                f"Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}"
                f" | Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        if early_stopping:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"--> Validation loss improved. Saving model...")
            else:
                counter += 1
                print(f"--> No improvement. EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Training stopped.")
                return train_total_loss, val_total_loss
    
    if not early_stopping:
        torch.save(model.state_dict(), save_path)
        print(f"Training Complete. Model saved to {save_path}")
    return train_total_loss, val_total_loss
