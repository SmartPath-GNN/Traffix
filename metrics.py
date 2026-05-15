import torch
import math


def calculate_mae(y_true, y_pred):
    """
    MAE: Ortalama mutlak hata.
    """

    return torch.mean(torch.abs(y_true - y_pred)).item()


def calculate_rmse(y_true, y_pred):
    """
    RMSE: Kök ortalama kare hata.
    Büyük hataları daha fazla cezalandırır.
    """

    mse = torch.mean((y_true - y_pred) ** 2)
    return math.sqrt(mse.item())


def evaluate_naive_baseline(dataloader, target_col_indices):
    """
    Naive baseline:
    Son zaman adımındaki hız ve araç yoğunluğunu,
    bir sonraki zaman adımı için aynen tahmin eder.
    """

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            last_time_step = x_batch[:, -1, :, :]
            baseline_pred = last_time_step[:, :, target_col_indices]

            all_predictions.append(baseline_pred.cpu())
            all_targets.append(y_batch.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    baseline_mae = calculate_mae(all_targets, all_predictions)
    baseline_rmse = calculate_rmse(all_targets, all_predictions)

    return baseline_mae, baseline_rmse