import pandas as pd
import osmnx as ox
import time
# pip install scikit-learn kurulumunu yapmayı unutma, scikit-learn arkaplanda nearest_nodes fonksiyonunda kullanılıyor.
# --- OSMNX SUNUCU AYARLARI --- 
ox.settings.timeout = 1800
ox.settings.memory = 1073741824
ox.settings.max_query_area_size = 2500000000

input_file = 'filtrelenmis_veri.csv'
output_file = 'gnn_hazir_veri.csv'

# Beşiktaş-Şişli-Kağıthane
LAT_MIN, LAT_MAX = 41.035, 41.100
LON_MIN, LON_MAX = 28.960, 29.045

def map_matching_chunking():
    baslangic_zamani = time.time()
    
    print("1. Harita (Graf) yükleniyor... (İlçe bazlı isimle çekiliyor)")
    # Bounding box yerine doğrudan ilçelerin resmi isimlerini kullanıyoruz
    ilceler = [
        'Beşiktaş, Istanbul, Turkey', 
        'Şişli, Istanbul, Turkey', 
        'Kağıthane, Istanbul, Turkey'
    ]
    
    # graph_from_bbox yerine graph_from_place kullanıyoruz
    G = ox.graph_from_place(ilceler, network_type='drive')
    
    print(f"   -> Düğüm Sayısı: {len(G.nodes)}")
    
    print("2. GNN ID Sözlüğü Hazırlanıyor...")
    unique_nodes = list(G.nodes())
    node_to_gnn_id = {osmid: i for i, osmid in enumerate(unique_nodes)}

    print("3. Map Matching İşlemi Başlıyor (Disk-to-Disk Stream)...")
    chunk_size = 50000 # i7 işlemci ve DDR5 RAM için ideal boyut
    chunk_count = 0
    toplam_satir = 3889765 # Senin belirttiğin satır sayısı
    islenen_toplam = 0

    # Veriyi CSV'den chunk halinde doğrudan okuyoruz
    for chunk_df in pd.read_csv(input_file, chunksize=chunk_size):
        chunk_count += 1
        islem_basla = time.time()
        
        # En yakın düğümleri bulma (scikit-learn arkaplanda çalışır)
        chunk_df['osmnx_node_id'] = ox.distance.nearest_nodes(
            G, 
            X=chunk_df['LONGITUDE'].values, 
            Y=chunk_df['LATITUDE'].values
        )
        
        # GNN ID'sine çevirme
        chunk_df['gnn_node_id'] = chunk_df['osmnx_node_id'].map(node_to_gnn_id)
        
        # Anında Diske Yazma
        if chunk_count == 1:
            chunk_df.to_csv(output_file, index=False, mode='w') # İlk chunk, dosyayı yarat
        else:
            chunk_df.to_csv(output_file, index=False, mode='a', header=False) # Diğerleri altına ekle
            
        islenen_toplam += len(chunk_df)
        gecen_sure = time.time() - islem_basla
        yuzde = (islenen_toplam / toplam_satir) * 100
        
        print(f"📦 Paket {chunk_count:03d} tamamlandı | İşlenen: {islenen_toplam} / {toplam_satir} (%{yuzde:.1f}) | Paket Süresi: {gecen_sure:.1f} sn")

    toplam_sure = (time.time() - baslangic_zamani) / 60
    print(f"✅ BÜYÜK İŞLEM TAMAMLANDI! Toplam Süre: {toplam_sure:.1f} dakika.")
    print(f"Çıktı dosyası: {output_file}")

if __name__ == "__main__":
    map_matching_chunking()