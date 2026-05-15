import osmnx as ox
import pandas as pd
import time
import torch      # PyTorch tensörleri için
import numpy as np # Hızlı matris (vektörizasyon) işlemleri için

def veri_setini_hazirla():
    print("1. Harita (Graf) yükleniyor... (Cache üzerinden saniyeler sürecek)")
    baslangic = time.time()
    ilceler = ['Beşiktaş, Istanbul, Turkey', 'Şişli, Istanbul, Turkey', 'Kağıthane, Istanbul, Turkey']
    G = ox.graph_from_place(ilceler, network_type='drive')
    
    # KONTROL NOKTASI
    unique_nodes = list(G.nodes())
    num_nodes = len(unique_nodes)
    print(f"   -> Toplam Düğüm (Kavşak) Sayısı: {num_nodes}")
    
    print("2. GNN ID Sözlüğü oluşturuluyor (map_matching.py ile BİREBİR AYNI sıra)...")
    # CSV kullanmadan, direkt harita üzerinden sırayla numaralandırıyoruz (0'dan düğüm sayısına kadar)
    node_to_gnn_id = {osmid: i for i, osmid in enumerate(unique_nodes)}
    
    print("3. Tüm Yollar (Kenarlar/Edges) ve AĞIRLIKLARI haritadan çıkarılıyor...")
    kaynak_dugumler = []
    hedef_dugumler = []
    kenar_agirliklari = [] # YENİ: Yolların uzunluğunu tutacağımız liste
    
    # data=True diyerek yolların özelliklerini de (uzunluk vb.) çekiyoruz
    for u, v, key, data in G.edges(keys=True, data=True):
        kaynak_dugumler.append(node_to_gnn_id[u])
        hedef_dugumler.append(node_to_gnn_id[v])
        
        # YENİ: OSMnx uzunluğu 'length' anahtarı ile verir. Eğer veri bozuk/eksikse varsayılan 10.0 metre ata.
        # Bu ağırlık, GNN'in uzun yollardaki trafiği kısa yollardan ayırt etmesini sağlayacak.
        uzunluk = data.get('length', 10.0) 
        kenar_agirliklari.append(uzunluk)
            
    print("4. Edge Index DataFrame'e dönüştürülüp CSV olarak kaydediliyor (Yedekleme)...")
    # Orijinal kodundaki CSV kaydetme adımını bozmuyoruz, debug (hata ayıklama) için faydalıdır
    edge_df = pd.DataFrame({'source': kaynak_dugumler, 'target': hedef_dugumler, 'weight_length': kenar_agirliklari})
    output_file_csv = 'edge_index.csv'
    edge_df.to_csv(output_file_csv, index=False)
    
    # ================= YENİ GNN+LSTM HAZIRLIK BÖLÜMÜ =================

    print("5. Graf Topolojisi PyTorch Tensörlerine Çevriliyor...")
    # [2, num_edges] formatında, int tipinde GNN kenar matrisi
    edge_index_tensor = torch.tensor([kaynak_dugumler, hedef_dugumler], dtype=torch.long)
    # [num_edges] formatında, float tipinde kenar ağırlıkları
    edge_weight_tensor = torch.tensor(kenar_agirliklari, dtype=torch.float32)

    print("6. Özellik (X) Matrisi CSV'den yükleniyor ve 3D LSTM Küpüne dönüştürülüyor...")
    # Hazırladığın CSV dosyasını okuyoruz
    df = pd.read_csv('model_girdisi_son.csv')
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME']) # Zamanı sıralayabilmek için datetime yapıyoruz
    
    # Modele vereceğimiz sinyaller (Koordinatları ezberlememesi için çıkardık)
    feature_cols = ['HOUR', 'DAY_OF_WEEK', 'NORM_SPEED', 'NORM_VEHICLES']
    num_features = len(feature_cols)
    
    # Zaman adımlarını kronolojik sıraya diziyoruz
    unique_times = df['DATE_TIME'].sort_values().unique()
    num_time_steps = len(unique_times)
    
    # Modelin beklediği [Zaman, Düğüm, Özellik] boyutlarında içi tamamen SIFIR olan bir matris oluşturuyoruz.
    # Sıfır yapıyoruz ki; eğer bir saatte bir kavşağın verisi yoksa sistem çökmesin, o saati 0 kabul etsin.
    X_array = np.zeros((num_time_steps, num_nodes, num_features), dtype=np.float32)
    
    # Zamanları 0, 1, 2 gibi indekslere çeviren sözlük
    time_to_idx = {time_val: idx for idx, time_val in enumerate(unique_times)}
    
    # Vektörizasyon (Hızlandırma) İşlemi: Hangi verinin küpte hangi XYZ koordinatına gideceğini buluyoruz
    t_indices = df['DATE_TIME'].map(time_to_idx).values 
    n_indices = df['gnn_node_id'].values                
    features = df[feature_cols].values                  
    
    # Döngü kullanmadan, saniyeler içinde tüm veriyi küpün içine tek seferde yerleştiriyoruz
    X_array[t_indices, n_indices, :] = features
    
    # Numpy matrisini nihai PyTorch Float Tensörüne çeviriyoruz
    x_tensor = torch.tensor(X_array, dtype=torch.float32)
    
    print("7. Tüm Veriler Modele Beslenmeye Hazır '.pt' Dosyası Olarak Kaydediliyor...")
    # 4. Adımda (Eğitim) sadece bu dosyayı torch.load() ile çağıracağız.
    dataset_path = 'gnn_lstm_dataset.pt'
    torch.save({
        'x': x_tensor, 
        'edge_index': edge_index_tensor,
        'edge_weight': edge_weight_tensor,
        'num_nodes': num_nodes,
        'num_features': num_features
    }, dataset_path)
    
    sure = time.time() - baslangic
    print(f"\n✅ İŞLEM TAMAMLANDI! Toplam Süre: {sure:.2f} saniye.")
    print(f"📊 ÖZET BİLGİLER:")
    print(f"   -> Toplam Yol (Kenar) Sayısı: {len(kaynak_dugumler)}")
    print(f"   -> X Matrisi Boyutu: {x_tensor.shape} [Zaman, Düğüm, Özellik]")
    print(f"   -> Edge Index Boyutu: {edge_index_tensor.shape} [2, Kenarlar]")
    print(f"   -> Edge Weight Boyutu: {edge_weight_tensor.shape} [Kenarlar]")
    print(f"   -> Çıktı Dosyaları: {output_file_csv} (Yedek) ve {dataset_path} (Nihai Model Girdisi)")

if __name__ == "__main__":
    veri_setini_hazirla()

