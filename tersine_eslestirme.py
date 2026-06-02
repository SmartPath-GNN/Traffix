import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans

def tersine_eslestirme_yap(model_cikti_tensor, scaler, gecerli_dugum_maskesi):
    print("1. Harita (Graf) yükleniyor...")
    ilceler = ['Beşiktaş, Istanbul, Turkey', 'Şişli, Istanbul, Turkey', 'Kağıthane, Istanbul, Turkey']
    G = ox.graph_from_place(ilceler, network_type='drive')
    
    # Graf verisini doğrudan GeoPandas DataFrame'lerine çeviriyoruz
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    print("   -> GeoPandas ile yolların (LineString) geometrileri çıkarıldı.")

    print("2. ID'leri Geri Çevirme (Reverse Mapping) sözlüğü oluşturuluyor...")
    unique_nodes = list(G.nodes())
    gnn_to_node_id = {i: osmid for i, osmid in enumerate(unique_nodes)}

    print("3. Model tahminleri DataFrame'e dönüştürülüyor ve Maskeleme uygulanıyor...")
    tahmin_dizisi = model_cikti_tensor.cpu().detach().numpy()
    gercek_hiz_degerleri = scaler.inverse_transform(tahmin_dizisi)[:, 0] 

    # 🌟 KRİTİK MÜDAHALE: Verisi hiç olmayan (maske=False) sokakların tahminini NaN (Boş) yapıyoruz.
    gercek_hiz_degerleri[~gecerli_dugum_maskesi] = np.nan

    df_tahminler = pd.DataFrame({
        'gnn_node_id': range(len(gercek_hiz_degerleri)), 
        'tahmini_hiz_kmh': gercek_hiz_degerleri
    })
    
    df_tahminler['osmnx_node_id'] = df_tahminler['gnn_node_id'].map(gnn_to_node_id)

    print("4. Geometri Ekleme Başlıyor...")
    gdf_edges = gdf_edges.reset_index()
    gdf_sonuc = gdf_edges.merge(df_tahminler, left_on='u', right_on='osmnx_node_id', how='left')
    
    gdf_sonuc = gdf_sonuc[['u', 'v', 'name', 'length', 'geometry', 'tahmini_hiz_kmh']]
    
    print("✅ Tersine eşleştirme tamamlandı!")
    return gdf_sonuc

def kronik_darbogazlari_bul(gdf_sonuc, kume_sayisi=3):
    print("\n🔍 Kronik darboğazlar K-Means ile tespit ediliyor...")
    
    # K-Means için sadece gerçek verisi olan (NaN olmayan) yolları alıyoruz
    gdf_temiz = gdf_sonuc.dropna(subset=['tahmini_hiz_kmh']).copy()
    
    X = gdf_temiz[['tahmini_hiz_kmh']].values
    
    kmeans = KMeans(n_clusters=kume_sayisi, random_state=42, n_init=10)
    gdf_temiz['trafik_durumu_kumesi'] = kmeans.fit_predict(X)
    
    # Kümeleri hız ortalamalarına göre sıralayalım (0 = En yoğun/yavaş)
    merkezler = kmeans.cluster_centers_.flatten()
    sirali_indeksler = np.argsort(merkezler)
    etiket_haritasi = {eski: yeni for yeni, eski in enumerate(sirali_indeksler)}
    gdf_temiz['trafik_durumu_kumesi'] = gdf_temiz['trafik_durumu_kumesi'].map(etiket_haritasi)
    
    # 🌟 KRİTİK MÜDAHALE 2: Temiz verideki küme sonuçlarını, orijinal tam haritaya monte ediyoruz.
    gdf_tam_harita = gdf_sonuc.copy()
    gdf_tam_harita['trafik_durumu_kumesi'] = np.nan
    gdf_tam_harita.loc[gdf_temiz.index, 'trafik_durumu_kumesi'] = gdf_temiz['trafik_durumu_kumesi']
    
    kronik_darbogazlar = gdf_temiz[gdf_temiz['trafik_durumu_kumesi'] == 0]
    print(f"🚨 Toplam {len(kronik_darbogazlar)} adet 'Kronik Darboğaz' sokağı/yolu tespit edildi.")
    
    return gdf_tam_harita