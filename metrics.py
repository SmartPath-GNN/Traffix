import torch
import math


def masked_mse_loss(y_pred, y_true, mask_value=0.0):
    """
    Masked MSE Loss

    Sadece gerçek veri olan hücrelerde hata hesaplar.
    y_true değeri mask_value'dan büyükse o hücre gerçek veri kabul edilir.

    Amaç:
    Eksik veri nedeniyle 0 doldurulan hücrelerin loss hesabını bozmasını engellemek.
    """

    mask = y_true > mask_value

    if mask.sum() == 0:
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)

    loss = (y_pred - y_true) ** 2

    masked_loss = loss[mask]

    return masked_loss.mean()


def calculate_masked_mae(y_true, y_pred, mask_value=0.0):
    """
    Maskeli MAE hesaplar.
    Sadece y_true > mask_value olan hücreleri dikkate alır.
    """

    mask = y_true > mask_value

    if mask.sum() == 0:
        return 0.0

    error = torch.abs(y_true - y_pred)

    return error[mask].mean().item()


def calculate_masked_rmse(y_true, y_pred, mask_value=0.0):
    """
    Maskeli RMSE hesaplar.
    Sadece y_true > mask_value olan hücreleri dikkate alır.
    """

    mask = y_true > mask_value

    if mask.sum() == 0:
        return 0.0

    mse = torch.mean((y_true[mask] - y_pred[mask]) ** 2)

    return math.sqrt(mse.item())


def calculate_masked_r2(y_true, y_pred, mask_value=0.0):
    """
    Maskeli R2 hesaplar.
    Sadece y_true > mask_value olan hücreleri dikkate alır.
    """

    mask = y_true > mask_value

    if mask.sum() == 0:
        return 0.0

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    ss_res = torch.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = torch.sum((y_true_masked - torch.mean(y_true_masked)) ** 2)

    if ss_tot == 0:
        return 0.0

    r2 = 1 - (ss_res / ss_tot)

    return r2.item()

def evaluate_naive_baseline(dataloader, target_col_indices):
    """
    Naive baseline:
    Son zaman adımındaki değerleri gelecek tahmini olarak kullanır.
    Maskeli metriklerle değerlendirilir.
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

    baseline_mae = calculate_masked_mae(all_targets, all_predictions)
    baseline_rmse = calculate_masked_rmse(all_targets, all_predictions)
    baseline_r2 = calculate_masked_r2(all_targets, all_predictions)

    return baseline_mae, baseline_rmse, baseline_r2