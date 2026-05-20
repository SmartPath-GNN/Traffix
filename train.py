import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrafficSlidingWindowDataset
from model import GNNLSTM
from engine import train_one_epoch, evaluate
from metrics import evaluate_naive_baseline, masked_mse_loss
from utils import get_device, EarlyStopping


def main():
    device = get_device()
    print(f"-> Kullanılan cihaz: {device}")

    print("\n1. Veriler yükleniyor...")
    dataset_dict = torch.load("gnn_lstm_dataset_6ay.pt", map_location="cpu")
    x_raw = dataset_dict["x"].float()
    edge_index = dataset_dict["edge_index"].long().to(device)
    edge_weight = dataset_dict["edge_weight"].float().to(device)
    print(f"   -> x_raw shape: {x_raw.shape}")
    print(f"   -> edge_index shape: {edge_index.shape}")
    print(f"   -> edge_weight shape: {edge_weight.shape}")

    target_indices = [2, 3]

    print("\n2. Lazy Sliding Window Dataset hazırlanıyor...")
    WINDOW_SIZE = 12
    HORIZON = 12
    total_time = x_raw.shape[0]
    sample_count = total_time - WINDOW_SIZE
    print(f"   -> Toplam zaman adımı: {total_time}")
    print(f"   -> Window size: {WINDOW_SIZE}")
    print(f"   -> Tahmin ufku / Horizon: {HORIZON}")
    print(f"   -> Toplam örnek sayısı: {sample_count}")

    print("\n3. Veri seti kronolojik olarak bölünüyor...")
    print("   -> %80 eğitim, %10 doğrulama, %10 test")
    train_end = int(sample_count * 0.8)
    val_end = int(sample_count * 0.9)

    train_dataset = TrafficSlidingWindowDataset(
        data_x=x_raw,
        window_size=WINDOW_SIZE,
        target_col_indices=target_indices,
        start_index=0,
        end_index=train_end,
        horizon=HORIZON
    )
    val_dataset = TrafficSlidingWindowDataset(
        data_x=x_raw,
        window_size=WINDOW_SIZE,
        target_col_indices=target_indices,
        start_index=train_end,
        end_index=val_end,
        horizon=HORIZON
    )
    test_dataset = TrafficSlidingWindowDataset(
        data_x=x_raw,
        window_size=WINDOW_SIZE,
        target_col_indices=target_indices,
        start_index=val_end,
        end_index=sample_count,
        horizon=HORIZON
    )
    print(f"   -> Eğitim örnek sayısı: {len(train_dataset)}")
    print(f"   -> Doğrulama örnek sayısı: {len(val_dataset)}")
    print(f"   -> Test örnek sayısı: {len(test_dataset)}")

    BATCH_SIZE = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n4. Model başlatılıyor...")
    model = GNNLSTM(
        input_features=x_raw.shape[-1],
        gnn_hidden=16,
        lstm_hidden=32,
        output_features=len(target_indices)
    ).to(device)
    print(model)

    loss_fn = masked_mse_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # scheduler: optimizer'dan hemen sonra tanimlanir
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        #verbose=True
    )

    early_stopping = EarlyStopping(
        patience=10,
        delta=1e-6,
        path="best_gnn_lstm_model.pt"
    )

    print("\n5. Egitim basliyor...")
    EPOCHS = 50

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch:03d} basladi...")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            edge_index=edge_index,
            edge_weight=edge_weight,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )

        val_loss, val_mae, val_rmse, val_r2 = evaluate(
            model=model,
            dataloader=val_loader,
            edge_index=edge_index,
            edge_weight=edge_weight,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"Val RMSE: {val_rmse:.6f} | "
            f"Val R2: {val_r2:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ikisi de bir kez, dogru sirada
        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("\nEarly Stopping tetiklendi. Egitim durduruldu.")
            break

    print("\n6. Test verisi ile final degerlendirme yapiliyor...")
    model.load_state_dict(torch.load("best_gnn_lstm_model.pt", map_location=device))

    test_loss, test_mae, test_rmse, test_r2 = evaluate(
        model=model,
        dataloader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        loss_fn=loss_fn,
        device=device
    )

    print("\nFinal Test Sonuclari:")
    print(f"-> Test MSE Loss: {test_loss:.6f}")
    print(f"-> Test MAE     : {test_mae:.6f}")
    print(f"-> Test RMSE    : {test_rmse:.6f}")
    print(f"-> Test R2      : {test_r2:.6f}")

    print("\n7. Naive Baseline ile karsilastirma yapiliyor...")
    baseline_mae, baseline_rmse, baseline_r2 = evaluate_naive_baseline(
        dataloader=test_loader,
        target_col_indices=target_indices
    )

    print("\nNaive Baseline Test Sonuclari:")
    print(f"-> Baseline MAE : {baseline_mae:.6f}")
    print(f"-> Baseline RMSE: {baseline_rmse:.6f}")
    print(f"-> Baseline R2  : {baseline_r2:.6f}")

    print("\nModel ve Baseline Karsilastirmasi:")
    print(f"-> Model MAE    : {test_mae:.6f}")
    print(f"-> Baseline MAE : {baseline_mae:.6f}")
    print(f"-> Model RMSE   : {test_rmse:.6f}")
    print(f"-> Baseline RMSE: {baseline_rmse:.6f}")
    print(f"-> Baseline R2   : {baseline_r2:.6f}")

    print("\nIslem tamamlandi.")
    print("En iyi model dosyasi: best_gnn_lstm_model.pt")


if __name__ == "__main__":
    main()