import osmnx as ox
import pandas as pd
import time

def komsuluk_matrisini_cikar():
    print("1. Harita (Graf) yükleniyor... (Cache üzerinden saniyeler sürecek)")
    baslangic = time.time()
    ilceler = ['Beşiktaş, Istanbul, Turkey', 'Şişli, Istanbul, Turkey', 'Kağıthane, Istanbul, Turkey']
    G = ox.graph_from_place(ilceler, network_type='drive')
    
    # KONTROL NOKTASI
    unique_nodes = list(G.nodes())
    print(f"   -> Toplam Düğüm Sayısı: {len(unique_nodes)}")
    
    print("2. GNN ID Sözlüğü oluşturuluyor (map_matching.py ile BİREBİR AYNI sıra)...")
    # CSV kullanmadan, direkt harita üzerinden sırayla numaralandırıyoruz (0'dan 9416'ya)
    node_to_gnn_id = {osmid: i for i, osmid in enumerate(unique_nodes)}
    
    print("3. Tüm Yollar (Kenarlar/Edges) haritadan çıkarılıyor...")
    kaynak_dugumler = []
    hedef_dugumler = []
    
    # Haritadaki tüm yolları eksiksiz alıyoruz
    for u, v, key in G.edges(keys=True):
        kaynak_dugumler.append(node_to_gnn_id[u])
        hedef_dugumler.append(node_to_gnn_id[v])
            
    print("4. Edge Index DataFrame'e dönüştürülüp kaydediliyor...")
    edge_df = pd.DataFrame({'source': kaynak_dugumler, 'target': hedef_dugumler})
    
    output_file = 'edge_index.csv'
    edge_df.to_csv(output_file, index=False)
    
    sure = time.time() - baslangic
    print(f"✅ İŞLEM TAMAMLANDI! Toplam Kenar (Yol) Sayısı: {len(edge_df)}")
    print(f"Süre: {sure:.2f} saniye. Dosya: {output_file}")

if __name__ == "__main__":
    komsuluk_matrisini_cikar()