import torch
import math
import gc

def masked_mse_loss(y_pred, y_true, mask_value=0.0):
    """Sadece gerçek veri barındıran hücrelerde kayıp hesaplayan maskeli MSE."""
    mask = y_true > mask_value
    if mask.sum() == 0:
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)
    loss = (y_pred - y_true) ** 2
    return loss[mask].mean()

def calculate_masked_mae(y_true, y_pred, mask_value=0.0):
    mask = y_true > mask_value
    if mask.sum() == 0: return 0.0
    return torch.abs(y_true - y_pred)[mask].mean().item()

def calculate_masked_rmse(y_true, y_pred, mask_value=0.0):
    mask = y_true > mask_value
    if mask.sum() == 0: return 0.0
    return math.sqrt(torch.mean((y_true[mask] - y_pred[mask]) ** 2).item())

def calculate_masked_r2(y_true, y_pred, mask_value=0.0):
    mask = y_true > mask_value
    if mask.sum() == 0: return 0.0
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    ss_res = torch.sum((y_true_m - y_pred_m) ** 2)
    ss_tot = torch.sum((y_true_m - torch.mean(y_true_m)) ** 2)
    if ss_tot == 0: return 0.0
    return (1 - (ss_res / ss_tot)).item()

def evaluate_naive_baseline(dataloader, target_col_indices, horizon=6):
    """Son zaman adımındaki bilinen gerçek verileri gelecek ufka kopyalayarak test eder."""
    total_mae = 0
    total_rmse = 0
    total_r2 = 0

    with torch.no_grad():
        for x_rec, _, _, y_batch in dataloader:
            # x_rec son adımı -> [Batch, Düğüm, Özellikler]
            last_time_step = x_rec[:, -1, :, :]
            baseline_pred_single = last_time_step[:, :, target_col_indices]

            # Boyutu [Batch, Horizon, Düğüm, Özellik] formuna broadcast etmek
            baseline_pred = baseline_pred_single.unsqueeze(1).repeat(1, horizon, 1, 1)

            # 🌟 OOM ÇÖZÜMÜ: Listeye eklemek yerine batch ortalamalarını topluyoruz
            total_mae += calculate_masked_mae(y_batch, baseline_pred)
            total_rmse += calculate_masked_rmse(y_batch, baseline_pred)
            total_r2 += calculate_masked_r2(y_batch, baseline_pred)
            
            del x_rec, y_batch, baseline_pred
            
    num_batches = len(dataloader)
    return total_mae / num_batches, total_rmse / num_batches, total_r2 / num_batches