import torch
from metrics import (
    calculate_masked_mae,
    calculate_masked_rmse,
    calculate_masked_r2
)


def train_one_epoch(model, dataloader, edge_index, edge_weight, optimizer, loss_fn, device):
    model.train()

    total_loss = 0

    for batch_no, (x_batch, y_batch) in enumerate(dataloader, start=1):
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().to(device)

        optimizer.zero_grad()

        predictions = model(x_batch, edge_index, edge_weight)

        loss = loss_fn(predictions, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_no % 20 == 0:
            print(f"      Batch {batch_no} | Loss: {loss.item():.6f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, edge_index, edge_weight, loss_fn, device):
    model.eval()

    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            predictions = model(x_batch, edge_index, edge_weight)

            loss = loss_fn(predictions, y_batch)

            total_loss += loss.item()

            all_predictions.append(predictions.cpu())
            all_targets.append(y_batch.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mae = calculate_masked_mae(all_targets, all_predictions)
    rmse = calculate_masked_rmse(all_targets, all_predictions)
    r2 = calculate_masked_r2(all_targets, all_predictions)

    return total_loss / len(dataloader), mae, rmse, r2