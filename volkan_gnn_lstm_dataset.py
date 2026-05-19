import osmnx as ox
import pandas as pd
import time
import torch      # PyTorch tensörleri için
import numpy as np # Hızlı matris (vektörizasyon) işlemleri için
import warnings

# Pandas'ın ortalama alırken verebileceği gereksiz uyarıları gizler
warnings.filterwarnings('ignore')

def veri_setini_hazirla():
    print("1. Harita (Graf) yükleniyor... (Cache üzerinden saniyeler sürecek)")
    baslangic = time.time()
    ilceler = ['Beşiktaş, Istanbul, Turkey', 'Şişli, Istanbul, Turkey', 'Kağıthane, Istanbul, Turkey']
    G = ox.graph_from_place(ilceler, network_type='drive')
    
    unique_nodes = list(G.nodes())
    num_nodes = len(unique_nodes)
    print(f"   -> Toplam Düğüm (Kavşak) Sayısı: {num_nodes}")
    
    print("2. GNN ID Sözlüğü oluşturuluyor (map_matching.py ile BİREBİR AYNI sıra)...")
    node_to_gnn_id = {osmid: i for i, osmid in enumerate(unique_nodes)}
    
    print("3. Tüm Yollar (Kenarlar/Edges) ve AĞIRLIKLARI haritadan çıkarılıyor...")
    kaynak_dugumler = []
    hedef_dugumler = []
    kenar_agirliklari = [] 
    
    for u, v, key, data in G.edges(keys=True, data=True):
        kaynak_dugumler.append(node_to_gnn_id[u])
        hedef_dugumler.append(node_to_gnn_id[v])
        
        # OSMnx uzunluğu 'length' anahtarı ile verir. Eksikse varsayılan 10.0 metre ata.
        uzunluk = data.get('length', 10.0) 
        kenar_agirliklari.append(uzunluk)
            
    print("4. Edge Index DataFrame'e dönüştürülüp CSV olarak kaydediliyor (Yedekleme)...")
    edge_df = pd.DataFrame({'source': kaynak_dugumler, 'target': hedef_dugumler, 'weight_length': kenar_agirliklari})
    output_file_csv = 'edge_index.csv'
    edge_df.to_csv(output_file_csv, index=False)
    
    # ================= GNN+LSTM HAZIRLIK BÖLÜMÜ =================

    print("5. Graf Topolojisi PyTorch Tensörlerine Çevriliyor...")
    edge_index_tensor = torch.tensor([kaynak_dugumler, hedef_dugumler], dtype=torch.long)
    edge_weight_tensor = torch.tensor(kenar_agirliklari, dtype=torch.float32)

    print("6. Özellik (X) Matrisi CSV'den yükleniyor...")
    df = pd.read_csv('model_girdisi_son.csv')
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    
    
    print("   -> Zaman filtresi uygulanıyor (Sadece 2024 Temmuz'dan sonrası alınıyor)...")
    df = df[df['DATE_TIME'] >= '2024-07-01'] 
    
    print("   ->Veri Şişmesini Önlemek İçin Zaman Gruplanıyor...")
    # Zamanı saatlik yuvarlıyoruz (Örn: 08:14 ve 08:45 -> 08:00 olur)
    df['DATE_TIME'] = df['DATE_TIME'].dt.floor('1h')
    
    # Aynı saat içindeki aynı kavşağa ait verilerin ortalamasını alarak tek bir satıra indirgiyoruz
    df = df.groupby(['DATE_TIME', 'gnn_node_id']).mean().reset_index()
    
    feature_cols = ['HOUR', 'DAY_OF_WEEK', 'NORM_SPEED', 'NORM_VEHICLES']
    num_features = len(feature_cols)
    
    unique_times = df['DATE_TIME'].sort_values().unique()
    num_time_steps = len(unique_times)
    
    print(f"   -> Gruplama Sonrası Benzersiz Saat Dilimi: {num_time_steps}")
    
    print("7. 3D LSTM Küpü (Spatio-Temporal Matris) oluşturuluyor...")
    # Modelin beklediği [Zaman, Düğüm, Özellik] boyutlarında içi SIFIR olan bir matris
    X_array = np.zeros((num_time_steps, num_nodes, num_features), dtype=np.float32)
    
    time_to_idx = {time_val: idx for idx, time_val in enumerate(unique_times)}
    
    # Vektörizasyon: Hangi verinin küpte hangi XYZ koordinatına gideceğini buluyoruz
    t_indices = df['DATE_TIME'].map(time_to_idx).values 
    n_indices = df['gnn_node_id'].values                
    features = df[feature_cols].values                  
    
    # Tüm veriyi küpün içine tek seferde yerleştiriyoruz
    X_array[t_indices, n_indices, :] = features
    
    # Numpy matrisini nihai PyTorch Float Tensörüne çeviriyoruz
    x_tensor = torch.tensor(X_array, dtype=torch.float32)
    
    print("8. Tüm Veriler Modele Beslenmeye Hazır '.pt' Dosyası Olarak Kaydediliyor...")
    dataset_path = 'gnn_lstm_dataset_6ay.pt'
    torch.save({
        'x': x_tensor, 
        'edge_index': edge_index_tensor,
        'edge_weight': edge_weight_tensor,
        'num_nodes': num_nodes,
        'num_features': num_features
    }, dataset_path)
    
    sure = time.time() - baslangic
    print(f"\n✅ KUSURSUZ İŞLEM TAMAMLANDI! Toplam Süre: {sure:.2f} saniye.")
    print(f"📊 ÖZET BİLGİLER:")
    print(f"   -> Toplam Yol (Kenar) Sayısı: {len(kaynak_dugumler)}")
    print(f"   -> X Matrisi Boyutu: {x_tensor.shape} [Zaman Adımı, Düğüm, Özellik]")
    print(f"   -> Çıktı Dosyası: {dataset_path} (PyTorch Ekibi Sadece Bu Dosyayı Kullanacak)")

if __name__ == "__main__":
    veri_setini_hazirla()