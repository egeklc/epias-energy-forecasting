import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_forecast_and_plot(model, 
                    device, 
                    df, scaled_df,
                    feature_cols, scaler,
                    day, window_size=168, horizon=24):
    """
    Generates a single direct multi-step forecast for specified horizon using a trained model and visualizes the results.

    This function builds input window from the scaled dataframe using the given window size.
    Runs the trained PyTorch model for predicting the next horizon time steps.
    Inverse transforms predictions to the original scale.
    Computes MAE and RMSE scores and plots the predicted vs. actual values.

    """
    start_idx = (horizon * day) + window_size
    input_start = start_idx - window_size
    input_end = start_idx
    
    if start_idx + horizon > len(df):
        raise ValueError("Forecast exceeds dataset length.")
    
    test_window = scaled_df[feature_cols].iloc[input_start:input_end]
    X = torch.tensor(test_window.values, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = df["consumption"].iloc[start_idx : start_idx + horizon].values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    dates = df["date"].iloc[start_idx:start_idx + horizon]
    
    
    plt.figure(figsize=(10, 5))
    plt.grid(True, alpha=0.5, linestyle="--")
    plt.plot(dates, y_true, label="Real", alpha=1)
    plt.plot(dates, y_pred, label="Prediction", alpha=0.8)
    plt.title(f"24-Step Forecast | MAE={mae:.2f} | RMSE={rmse:.2f}")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.legend()
    plt.show()
    
    return {"MAE": mae, "RMSE": rmse}