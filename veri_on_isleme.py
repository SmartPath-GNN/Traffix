import pandas as pd

# --- AYARLAR ---
input_file = 'istanbulTraffic2020-2024.csv' # Kaggle'dan indirdiğin dosya adı
output_file = 'filtrelenmis_veri.csv'

# Beşiktaş, Şişli, Kağıthane bölgesini kapsayan koordinat sınırları (Bounding Box)
LAT_MIN, LAT_MAX = 41.035, 41.100
LON_MIN, LON_MAX = 28.960, 29.045

def veri_temizle():
    print("🚀 İşlem başlatıldı, veri yükleniyor... (7GB dosya chunk halinde işleniyor)")
    
    silinecekler = ['GEOHASH', 'MINIMUM_SPEED', 'MAXIMUM_SPEED']
    chunks = []
    chunk_count = 0
    
    # 1. Veriyi Chunk halinde Oku ve İşle
    for chunk in pd.read_csv(input_file, chunksize=1000000):
        chunk_count += 1
        print(f"📦 Chunk {chunk_count} işleniyor ({chunk.shape[0]} satır)...")
        
        # 2. Gereksiz Sütunları Ayıkla
        chunk = chunk.drop(columns=[col for col in silinecekler if col in chunk.columns])
        
        # 3. Bölge Filtreleme (Enlem ve Boylam bazlı)
        chunk = chunk[
            (chunk['LATITUDE'] >= LAT_MIN) & 
            (chunk['LATITUDE'] <= LAT_MAX) & 
            (chunk['LONGITUDE'] >= LON_MIN) & 
            (chunk['LONGITUDE'] <= LON_MAX)
        ]
        
        # 4. Boş Verileri (NaN) Temizle
        chunk = chunk.dropna()
        
        # Işlenmis chunk'ı saklayacak liste
        if len(chunk) > 0:
            chunks.append(chunk)
    
    # 5. Tüm Chunks'ları Birleştir
    df = pd.concat(chunks, ignore_index=True)
    print(f"📊 Toplam işlenen satır sayısı: {len(df)}")
    
    # 6. Yeni Veri Setini Kaydet
    df.to_csv(output_file, index=False)
    print(f"✅ Filtrelenmiş veri seti '{output_file}' adıyla başarıyla oluşturuldu!")

if __name__ == "__main__":
    veri_temizle()