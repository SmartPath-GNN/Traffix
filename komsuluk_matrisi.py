import osmnx as ox
import pandas as pd
import time

def komsuluk_matrisini_cikar():
    print("1. Harita (Graf) yükleniyor...")
    baslangic = time.time()
    
    ilceler = [
        'Beşiktaş, Istanbul, Turkey', 
        'Şişli, Istanbul, Turkey', 
        'Kağıthane, Istanbul, Turkey'
    ]
    G = ox.graph_from_place(ilceler, network_type='drive')
    
    print("2. GNN ID Sözlüğü tekrar oluşturuluyor...")
    # Node'ları 0, 1, 2... formatına eşliyoruz
    unique_nodes = list(G.nodes())
    node_to_gnn_id = {osmid: i for i, osmid in enumerate(unique_nodes)}
    
    print("3. Yollar (Kenarlar/Edges) çıkarılıyor...")
    # PyTorch Geometric formatı için (source, target) listeleri hazırlıyoruz
    kaynak_dugumler = []
    hedef_dugumler = []
    
    # G.edges() haritadaki birbirine bağlanan tüm kavşakları verir
    for u, v, key in G.edges(keys=True):
        if u in node_to_gnn_id and v in node_to_gnn_id:
            kaynak_dugumler.append(node_to_gnn_id[u])
            hedef_dugumler.append(node_to_gnn_id[v])
            
    print("4. Edge Index DataFrame'e dönüştürülüp kaydediliyor...")
    edge_df = pd.DataFrame({
        'source': kaynak_dugumler,
        'target': hedef_dugumler
    })
    
    output_file = 'edge_index.csv'
    edge_df.to_csv(output_file, index=False)
    
    sure = time.time() - baslangic
    print(f"✅ İŞLEM TAMAMLANDI! Toplam Bağlantı (Kenar) Sayısı: {len(edge_df)}")
    print(f"Süre: {sure:.2f} saniye. Dosya: {output_file}")
    
    print("\nÖrnek Komşuluk Formatı (İlk 5 Satır):")
    print(edge_df.head())

if __name__ == "__main__":
    komsuluk_matrisini_cikar()