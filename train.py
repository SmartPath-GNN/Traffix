import torch
from torch.utils.data import DataLoader
from dataset import TrafficSlidingWindowDataset
from model import MultiStreamGNNLSTM
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

    target_indices = [2, 3] # Hız ve Araç sayısı

    print("\n2. Çok Kollu Yapı ve Tahmin Ufku Hazırlanıyor...")
    WINDOW_SIZE = 3 # Kısa dönem pencere genişliği (Recent)
    HORIZON = 6     # Tahmin ufku dizi uzunluğu
    total_time = x_raw.shape[0]
    
    # 🌟 KRİTİK DÜZELTME: Güvenli zaman sınırlarının belirlenmesi
    # En az 1 haftalık (168 saat) geçmiş + kısa pencerenin sol payı (1) = 169. saatten başlamalıyız
    t_min = 168 + (WINDOW_SIZE // 2) 
    
    # Hedef ufuk boyunca veri sızıntısı olmaması için son 6 saati (HORIZON) bırakmalıyız
    t_max = total_time - HORIZON
    
    # Sadece bu güvenli alan içerisindeki verileri böleceğiz
    valid_sample_count = t_max - t_min

    print(f"\n3. Veri seti kronolojik olarak bölünüyor (Güvenli Bölge: {t_min} - {t_max})...")
    train_end = t_min + int(valid_sample_count * 0.8)
    val_end = t_min + int(valid_sample_count * 0.9)

    train_dataset = TrafficSlidingWindowDataset(
        data_x=x_raw, window_size=WINDOW_SIZE, target_col_indices=target_indices,
        start_index=t_min, end_index=train_end, horizon=HORIZON
    )
    val_dataset = TrafficSlidingWindowDataset(
        data_x=x_raw, window_size=WINDOW_SIZE, target_col_indices=target_indices,
        start_index=train_end, end_index=val_end, horizon=HORIZON
    )
    test_dataset = TrafficSlidingWindowDataset(
        data_x=x_raw, window_size=WINDOW_SIZE, target_col_indices=target_indices,
        start_index=val_end, end_index=t_max, horizon=HORIZON
    )

    BATCH_SIZE = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n4. Çok Kollu Model başlatılıyor...")
    model = MultiStreamGNNLSTM(
        input_features=x_raw.shape[-1],
        gnn_hidden=16,
        lstm_hidden=32,
        output_features=len(target_indices),
        horizon=HORIZON
    ).to(device)

    loss_fn = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=10, path="best_gnn_lstm_model.pt")

    print("\n5. Eğitim başlıyor...")
    EPOCHS = 50
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch:03d} basladi...")
        train_loss = train_one_epoch(model, train_loader, edge_index, edge_weight, optimizer, loss_fn, device)
        val_loss, val_mae, val_rmse, val_r2 = evaluate(model, val_loader, edge_index, edge_weight, loss_fn, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop: break

    print("\n6. Final Test Değerlendirmesi...")
    model.load_state_dict(torch.load("best_gnn_lstm_model.pt", map_location=device))
    test_loss, test_mae, test_rmse, test_r2 = evaluate(model, test_loader, edge_index, edge_weight, loss_fn, device)
    print(f"\nFinal Test Sonuçları:\n-> MSE Loss: {test_loss:.6f}\n-> MAE: {test_mae:.6f}\n-> RMSE: {test_rmse:.6f}\n-> R2: {test_r2:.6f}")

    print("\n7. Naive Baseline ile Karşılaştırma...")
    b_mae, b_rmse, b_r2 = evaluate_naive_baseline(test_loader, target_indices, horizon=HORIZON)
    print(f"\nNaive Baseline Sonuçları:\n-> MAE: {b_mae:.6f}\n-> RMSE: {b_rmse:.6f}\n-> R2: {b_r2:.6f}")

if __name__ == "__main__":
    main()