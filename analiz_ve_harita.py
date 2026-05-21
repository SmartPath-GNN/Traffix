import torch
import joblib
import warnings
from model import GNNLSTM
from utils import get_device
from tersine_eslestirme import tersine_eslestirme_yap, kronik_darbogazlari_bul
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ANA ÇALIŞTIRMA BLOĞU
if __name__ == "__main__":
    device = get_device()
    print(f"-> Cihaz: {device}")

    print("\n1. Model ve Veriler Yükleniyor...")
    # Veriyi ve graf yapısını yükle
    dataset_dict = torch.load("gnn_lstm_dataset_6ay.pt", map_location="cpu")
    x_raw = dataset_dict["x"].float().to(device)
    edge_index = dataset_dict["edge_index"].long().to(device)
    edge_weight = dataset_dict["edge_weight"].float().to(device)
    
    # Kaydettiğimiz scaler'ı yükle
    scaler = joblib.load('traffic_scaler.pkl')

    # Modeli ayağa kaldır (train.py'deki mimarinin birebir aynısı olmalı)
    model = GNNLSTM(
        input_features=x_raw.shape[-1],
        gnn_hidden=16,
        lstm_hidden=32,
        output_features=2 # Hız ve Araç sayısı
    ).to(device)
    
    # Eğitilmiş ağırlıkları (öğrenilmiş bilgileri) modele yükle
    model.load_state_dict(torch.load("best_gnn_lstm_model.pt", map_location=device))
    model.eval() # Değerlendirme (tahmin) moduna al

    print("\n2. Gelecek Zaman İçin Tahmin Üretiliyor...")
    # Örnek olarak elimizdeki en son 12 saatlik (WINDOW_SIZE=12) veriyi alıp geleceği tahmin edelim
    window_size = 12
    son_pencere = x_raw[-window_size:, :, :] # En güncel veri
    
    # Modelin beklediği batch boyutuna (1, time_steps, nodes, features) getir
    son_pencere = son_pencere.unsqueeze(0) 

    with torch.no_grad(): # Gradient hesaplamaya gerek yok
        # prediction boyutu: [1, num_nodes, output_features]
        gelecek_tahmini = model(son_pencere, edge_index, edge_weight)
        
    # Batch boyutunu atıp sadece kavşak verilerini (num_nodes, output_features) alalım
    tahmin_tensoru = gelecek_tahmini[0] 

    print("\n3. Görselleştirme ve Analiz Hattı Başlıyor...")
    # Sizin yazdığınız 1. Fonksiyonu çağırıyoruz
    harita_gdf = tersine_eslestirme_yap(tahmin_tensoru, scaler)
    
    # Sizin yazdığınız 2. Fonksiyonu çağırıyoruz
    darbogazlar_gdf = kronik_darbogazlari_bul(harita_gdf, kume_sayisi=3)
    
    print("\n🎉 TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")