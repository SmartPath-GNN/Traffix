import torch
import gc
from metrics import (
    calculate_masked_mae,
    calculate_masked_rmse,
    calculate_masked_r2
)

def train_one_epoch(model, dataloader, edge_index, edge_weight, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch_no, (x_rec, x_day, x_wek, y_batch) in enumerate(dataloader, start=1):
        x_rec = x_rec.float().to(device)
        x_day = x_day.float().to(device)
        x_wek = x_wek.float().to(device)
        y_batch = y_batch.float().to(device)

        optimizer.zero_grad()

        # Modeli çok kollu girdilerle besliyoruz
        predictions = model(x_rec, x_day, x_wek, edge_index, edge_weight)

        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_no % 20 == 0:
            print(f"      Batch {batch_no} | Loss: {loss.item():.6f}")
            
        # 🌟 RAM VE VRAM TEMİZLİĞİ: Geriye kalan devasa tensörleri hemen yok et
        del x_rec, x_day, x_wek, y_batch, predictions, loss
        if batch_no % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, edge_index, edge_weight, loss_fn, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    total_r2 = 0

    with torch.no_grad():
        for x_rec, x_day, x_wek, y_batch in dataloader:
            x_rec = x_rec.float().to(device)
            x_day = x_day.float().to(device)
            x_wek = x_wek.float().to(device)
            y_batch = y_batch.float().to(device)

            predictions = model(x_rec, x_day, x_wek, edge_index, edge_weight)
            loss = loss_fn(predictions, y_batch)

            total_loss += loss.item()
            
            # 🌟 OOM ÇÖZÜMÜ: Tüm tahminleri bellekte biriktirmek yerine batch bazlı hesaplıyoruz
            total_mae += calculate_masked_mae(y_batch, predictions)
            total_rmse += calculate_masked_rmse(y_batch, predictions)
            total_r2 += calculate_masked_r2(y_batch, predictions)
            
            # Belleği anında boşaltıyoruz
            del x_rec, x_day, x_wek, y_batch, predictions, loss

    num_batches = len(dataloader)
    return total_loss / num_batches, total_mae / num_batches, total_rmse / num_batches, total_r2 / num_batches