import torch
import joblib
import warnings
from model import MultiStreamGNNLSTM
from utils import get_device
from tersine_eslestirme import tersine_eslestirme_yap, kronik_darbogazlari_bul

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    device = get_device()
    print(f"-> Cihaz: {device}")

    print("\n1. Model ve Veriler Yükleniyor...")
    dataset_dict = torch.load("gnn_lstm_dataset_6ay.pt", map_location="cpu")
    x_raw = dataset_dict["x"].float().to(device)
    edge_index = dataset_dict["edge_index"].long().to(device)
    edge_weight = dataset_dict["edge_weight"].float().to(device)
    scaler = joblib.load('traffic_scaler.pkl')

    HORIZON = 6
    model = MultiStreamGNNLSTM(
        input_features=x_raw.shape[-1], gnn_hidden=16, lstm_hidden=32, output_features=2, horizon=HORIZON
    ).to(device)
    model.load_state_dict(torch.load("best_gnn_lstm_model.pt", map_location=device))
    model.eval()

    print("\n2. Çok Kollu Güncel Zaman Pencereleri Dilimleniyor...")
    t = x_raw.shape[0]
    window_size = 3

    # Girdileri batch boyutuna (1, Zaman, Düğüm, Özellik) yükseltiyoruz
    x_recent = x_raw[t - window_size : t].unsqueeze(0)
    x_daily  = x_raw[t - 24 - (window_size // 2) : t - 24 + (window_size // 2) + (window_size % 2)].unsqueeze(0)
    x_weekly = x_raw[t - 168 - (window_size // 2) : t - 168 + (window_size // 2) + (window_size % 2)].unsqueeze(0)

    print("\n3. Gelecek Dizi İçin Tek Seferde Tahmin Üretiliyor...")
    with torch.no_grad():
        tüm_ufuk_tahmini = model(x_recent, x_daily, x_weekly, edge_index, edge_weight)
        
    # Örnek analitik çıktı olarak serinin 1. saatini haritalandıralım
    tahmin_tensoru = tüm_ufuk_tahmini[0, 0, :, :] 

    print("\n4. Görselleştirme ve Coğrafi Analiz Başlıyor...")
    harita_gdf = tersine_eslestirme_yap(tahmin_tensoru, scaler)
    darbogazlar_gdf = kronik_darbogazlari_bul(harita_gdf, kume_sayisi=3)
    
    print("\n🎉 TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")