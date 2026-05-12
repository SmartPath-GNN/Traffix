import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

input_file = 'gnn_hazir_veri.csv'
output_file = 'model_girdisi_son.csv'

def oznitelik_muhendisligi():
    print("1. GNN hazır verisi yükleniyor...")
    baslangic = time.time()
    
    # Veriyi RAM'e alıyoruz (DDR5 için 3.8M satır saniyeler sürer)
    df = pd.read_csv(input_file)
    print(f"   -> Toplam Satır: {len(df)}")

    print("2. Zaman (Zaman Serisi / LSTM) öznitelikleri çıkarılıyor...")
    # Metin halindeki tarihi Pandas DateTime formatına çeviriyoruz
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    
    # Modeller için Saati (0-23) ve Haftanın Gününü (0-6) ayırıyoruz
    df['HOUR'] = df['DATE_TIME'].dt.hour
    df['DAY_OF_WEEK'] = df['DATE_TIME'].dt.dayofweek

    print("3. Min-Max Normalizasyonu Yapılıyor (0-1 arasına sıkıştırma)...")
    scaler = MinMaxScaler()
    
    # Sadece sayısal değerleri sıkıştırıyoruz
    sutunlar_normalize = ['AVERAGE_SPEED', 'NUMBER_OF_VEHICLES']
    df[['NORM_SPEED', 'NORM_VEHICLES']] = scaler.fit_transform(df[sutunlar_normalize])
    
    print("4. Modelin ihtiyaç duymadığı geçici veriler temizleniyor...")
    # Yapay zekaya doğrudan girmeyecek olan orijinal metin ve raw id sütunlarını atıyoruz
    temizlenecekler = ['DATE_TIME', 'osmnx_node_id', 'AVERAGE_SPEED', 'NUMBER_OF_VEHICLES']
    df = df.drop(columns=[col for col in temizlenecekler if col in df.columns])

    print("5. Son veri seti kaydediliyor...")
    df.to_csv(output_file, index=False)
    
    sure = time.time() - baslangic
    print(f"✅ BÜYÜK VERİ MÜHENDİSLİĞİ TAMAMLANDI! Süre: {sure:.2f} saniye.")
    print(f"Üretilen Son Dosya: {output_file}")
    
    print("\nYapay Zekaya Girecek Nihai Veri (İlk 3 Satır):")
    print(df.head(3))

if __name__ == "__main__":
    oznitelik_muhendisligi()