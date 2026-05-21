import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans

def tersine_eslestirme_yap(model_cikti_tensor, scaler):
    print("1. Harita (Graf) yükleniyor...")
    ilceler = ['Beşiktaş, Istanbul, Turkey', 'Şişli, Istanbul, Turkey', 'Kağıthane, Istanbul, Turkey']
    G = ox.graph_from_place(ilceler, network_type='drive')
    
    # Graf verisini doğrudan GeoPandas DataFrame'lerine (Noktalar ve Çizgiler) çeviriyoruz
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    print("   -> GeoPandas ile yolların (LineString) geometrileri çıkarıldı.")

    print("2. ID'leri Geri Çevirme (Reverse Mapping) sözlüğü oluşturuluyor...")
    unique_nodes = list(G.nodes())
    
    # Senin komsuluk_matrisi.py'deki eşleştirmenin tam tersi: GNN ID -> OSM ID
    gnn_to_node_id = {i: osmid for i, osmid in enumerate(unique_nodes)}

    print("3. Model tahminleri DataFrame'e dönüştürülüyor...")
    # Örnek: model_cikti_tensor boyutu (num_nodes, num_features) olsun.
    # Modelin normalize edilmiş (0-1) çıktı ürettiğini varsayıyoruz, scaler ile gerçek km/s değerine geri çevirelim.
    # Not: normalizasyon.py'de kaydettiğin scaler'ı buraya yüklemen gerekir.
    
    tahmin_dizisi = model_cikti_tensor.cpu().detach().numpy()
    
    # Sadece hız tahmini yapıldığını varsayarak ters dönüşüm (inverse_transform)
    # Eğer çoklu sütun varsa (Hız ve Araç sayısı) scaler.inverse_transform(tahmin_dizisi) yapmalısın.
    gercek_hiz_degerleri = scaler.inverse_transform(tahmin_dizisi)[:, 0] # Sadece Hız sütununu al
    
    # GNN ID'leri ile gerçek OSM ID'leri eşleştirip DataFrame yapalım
    # YENİ KOD (Düzeltilmiş kısım)
    df_tahminler = pd.DataFrame({
        'gnn_node_id': range(len(gercek_hiz_degerleri)), # Haritaya göre değil, tahmin sayısına göre indeksle
        'tahmini_hiz_kmh': gercek_hiz_degerleri
    })
    
    # GNN ID'yi OSM ID'ye çevir (Adım 5 - Madde 1)
    df_tahminler['osmnx_node_id'] = df_tahminler['gnn_node_id'].map(gnn_to_node_id)
    
    print("4. Geometri Ekleme (Adım 5 - Madde 2)...")
    # Çizgi (Edge) geometrilerinde 'u' başlangıç düğümüdür (source node)
    # Tahminlerimizi bu başlangıç düğümüne göre haritadaki çizgilere (LineString) bağlıyoruz
    gdf_edges = gdf_edges.reset_index()
    gdf_sonuc = gdf_edges.merge(df_tahminler, left_on='u', right_on='osmnx_node_id', how='left')
    
    # Sadece ihtiyacımız olan sütunları bırakıp temizleyelim
    gdf_sonuc = gdf_sonuc[['u', 'v', 'name', 'length', 'geometry', 'tahmini_hiz_kmh']]
    
    print("5. Sonuçlar kaydediliyor...")
    # Çıktıyı GeoJSON olarak kaydedebilirsin (Bunu Kepler.gl veya QGIS gibi harita programlarında doğrudan açabilirsin)
    gdf_sonuc.to_file('trafik_tahmin_haritasi.geojson', driver='GeoJSON')
    print("✅ Tersine eşleştirme tamamlandı! 'trafik_tahmin_haritasi.geojson' dosyası oluşturuldu.")
    
    return gdf_sonuc

# KÜMELEME
def kronik_darbogazlari_bul(gdf_sonuc, kume_sayisi=3):
    print("\n🔍 Kronik darboğazlar K-Means ile tespit ediliyor...")
    
    # Boş verileri (NaN) temizle
    gdf_temiz = gdf_sonuc.dropna(subset=['tahmini_hiz_kmh']).copy()
    
    # Sadece hız verisini kullanarak K-Means uyguluyoruz
    # Amacımız trafiğin çok yavaş aktığı kümeleri bulmak
    X = gdf_temiz[['tahmini_hiz_kmh']].values
    
    kmeans = KMeans(n_clusters=kume_sayisi, random_state=42, n_init=10)
    gdf_temiz['trafik_durumu_kumesi'] = kmeans.fit_predict(X)
    
    # Kümeleri hız ortalamalarına göre sıralayalım (0 = En yoğun/yavaş, 1 = Normal, 2 = Akıcı)
    merkezler = kmeans.cluster_centers_.flatten()
    sirali_indeksler = np.argsort(merkezler)
    etiket_haritasi = {eski: yeni for yeni, eski in enumerate(sirali_indeksler)}
    gdf_temiz['trafik_durumu_kumesi'] = gdf_temiz['trafik_durumu_kumesi'].map(etiket_haritasi)
    
    # En yavaş küme (0 numaralı küme) kronik darboğazlardır
    kronik_darbogazlar = gdf_temiz[gdf_temiz['trafik_durumu_kumesi'] == 0]
    print(f"🚨 Toplam {len(kronik_darbogazlar)} adet 'Kronik Darboğaz' sokağı/yolu tespit edildi.")
    
    # İstersen darboğazları ayrı bir dosya olarak da kaydedebilirsin
    kronik_darbogazlar.to_file('kronik_darbogazlar.geojson', driver='GeoJSON')
    
    return gdf_temiz