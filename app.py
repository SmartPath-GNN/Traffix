import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import torch
import joblib
from model import GNNLSTM
from utils import get_device
from analiz_ve_harita import tersine_eslestirme_yap, kronik_darbogazlari_bul

# 1. Sayfa Ayarları
st.set_page_config(page_title="Traffix Akıllı Trafik Yönetimi", layout="wide")
st.title("🚦 Traffix: GNN-LSTM Tabanlı Trafik Tahmin Paneli")

# 2. Modeli ve Verileri Önbelleğe Alma (Uygulamanın yavaşlamaması için ÇOK ÖNEMLİ)
@st.cache_resource
def load_model_and_data():
    # 1. Cihazı ve Veriyi Yükle
    device = get_device()
    dataset_dict = torch.load("gnn_lstm_dataset_6ay.pt", map_location="cpu")
    x_raw = dataset_dict["x"].float().to(device)
    edge_index = dataset_dict["edge_index"].long().to(device)
    edge_weight = dataset_dict["edge_weight"].float().to(device)
    
    scaler = joblib.load('traffic_scaler.pkl')

    # 2. Modeli Ayağa Kaldır
    model = GNNLSTM(
        input_features=x_raw.shape[-1],
        gnn_hidden=16,
        lstm_hidden=32,
        output_features=2 
    ).to(device)
    model.load_state_dict(torch.load("best_gnn_lstm_model.pt", map_location=device))
    model.eval()

    # 3. OTO-REGRESİF ZAMAN DÖNGÜSÜ (1'den 6 Saate Kadar)
    window_size = 12
    mevcut_pencere = x_raw[-window_size:, :, :].unsqueeze(0).clone() 

    tahminler_sozlugu = {} # 6 farklı saatin haritasını burada tutacağız

    with torch.no_grad():
        for saat_ileri in range(1, 7): # 1, 2, 3, 4, 5, 6
            # A) Tahmin Üret
            gelecek_tahmini = model(mevcut_pencere, edge_index, edge_weight)
            tahmin_tensoru = gelecek_tahmini[0] 

            # B) Bu saat için harita verilerini üret
            harita_gdf = tersine_eslestirme_yap(tahmin_tensoru, scaler)
            darbogazlar_gdf = kronik_darbogazlari_bul(harita_gdf, kume_sayisi=3)
            
            # C) Sözlüğe kaydet (Örn: tahminler_sozlugu[3] = 3. saatin haritası)
            tahminler_sozlugu[saat_ileri] = {
                'harita': harita_gdf,
                'darbogazlar': darbogazlar_gdf
            }
            
            # D) --- PENCEREYİ BİR SAAT İLERİ KAYDIR ---
            # Mevcut saati ve günü alıp 1 saat ekliyoruz
            son_saat = mevcut_pencere[0, -1, 0, 0].item()
            son_gun = mevcut_pencere[0, -1, 0, 1].item()
            
            yeni_saat = (son_saat + 1) % 24
            yeni_gun = son_gun if yeni_saat != 0 else (son_gun + 1) % 7
            
            # Modelin beklediği yeni özellik matrisini oluşturuyoruz
            yeni_adim = torch.zeros((1, 1, mevcut_pencere.shape[2], 4), device=device)
            yeni_adim[0, 0, :, 0] = yeni_saat
            yeni_adim[0, 0, :, 1] = yeni_gun
            yeni_adim[0, 0, :, 2:] = gelecek_tahmini # Kendi ürettiği tahmini GİRDİ olarak veriyor!
            
            # Eski pencerenin en başındaki saati silip, yeni saati en sona ekliyoruz
            mevcut_pencere = torch.cat((mevcut_pencere[:, 1:, :, :], yeni_adim), dim=1)
            
    return tahminler_sozlugu

# SADECE 1 KERE ÇALIŞIP 6 SAATİ DE ÖNBELLEĞE ALIYOR
tahminler_sozlugu = load_model_and_data()

# 3. Sol Menü (Kullanıcı Kontrolleri)
st.sidebar.header("Kontrol Paneli")
secilen_zaman = st.sidebar.slider("Tahmin Ufku (Saat)", min_value=1, max_value=6, value=1)
gosterim_modu = st.sidebar.radio("Harita Modu", ["Isı Haritası (Genel Trafik)", "Sadece Darboğazlar (Uyarı)"])

# 🌟 DÜZELTME: Sürgüden gelen zamana göre İLGİLİ SAATİN verisini sözlükten çekiyoruz!
# (.copy() kullanmak çok önemlidir, aksi takdirde Streamlit hata verir)
harita_gdf = tahminler_sozlugu[secilen_zaman]['harita'].copy()
darbogazlar_gdf = tahminler_sozlugu[secilen_zaman]['darbogazlar'].copy()


# 4. Arayüz Bölünmesi (Harita ve Grafikler yan yana)
col1, col2 = st.columns([2, 1])

def hiz_rengi_belirle(hiz):
    if pd.isna(hiz):
        return [128, 128, 128, 100]  # Gri: Veri yok veya hesaplanamadı
    elif hiz < 25.0:
        return [227, 26, 28, 255]    # Koyu Kırmızı: Trafik Kilit (0-25 km/s)
    elif hiz < 40.0:
        return [253, 141, 60, 255]   # Turuncu: Yoğun (25-40 km/s)
    elif hiz < 55.0:
        return [254, 204, 92, 255]   # Sarı: Akıcı ama kalabalık (40-55 km/s)
    else:
        return [35, 139, 69, 255]    # Koyu Yeşil: Tamamen Akıcı (55+ km/s)

with col1:
    st.subheader(f"🗺️ İstanbul Tıkanıklık Yayılımı (+{secilen_zaman} Saat)")
    
    harita_gdf['hiz_gosterim'] = harita_gdf['tahmini_hiz_kmh'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Veri Yok")
    
    # 1. RENKLERİ UYGULA
    harita_gdf['cizgi_rengi'] = harita_gdf['tahmini_hiz_kmh'].apply(hiz_rengi_belirle)

    # 2. DARBOĞAZLARI İZOLE ET (Sadece 0 numaralı kümeyi alıyoruz)
    gercek_darbogazlar_gdf = darbogazlar_gdf[darbogazlar_gdf['trafik_durumu_kumesi'] == 0].copy()
    gercek_darbogazlar_gdf['hiz_gosterim'] = gercek_darbogazlar_gdf['tahmini_hiz_kmh'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Veri Yok")

    # 3. HARİTA KATMANLARI
    layer_traffic = pdk.Layer(
        "GeoJsonLayer",
        data=harita_gdf,
        get_line_color="cizgi_rengi",
        get_line_width=10,            
        pickable=True                 
    )
    
    # DÜZELTME: Scatterplot (nokta) yerine GeoJsonLayer (Çizgi) kullanıyoruz
    layer_bottleneck = pdk.Layer(
        "GeoJsonLayer",
        data=gercek_darbogazlar_gdf,
        get_line_color=[255, 0, 0, 255], # Parlak, opak kırmızı
        get_line_width=10,               # Trafik çizgisinden daha kalın (vurgulu)
        pickable=True
    )

    # 4. GÖSTERİM MANTIĞI
    if gosterim_modu == "Sadece Darboğazlar (Uyarı)":
        layers = [layer_bottleneck] # Sadece darboğazları kalın kırmızı göster
    else:
        layers = [layer_traffic] # DÜZELTME: Isı haritasında darboğazları üstüne ekleme, sadece normal trafiği göster!
    
    # 5. HARİTAYI ÇİZ
    st.pydeck_chart(pdk.Deck(
        map_style="carto-dark",
        initial_view_state=pdk.ViewState(latitude=41.06, longitude=29.0, zoom=12),
        layers=layers,
        # DÜZELTME: Tooltip artık orijinal sayıyı değil, formatlanmış metin sütununu okuyor
        tooltip={"text": "Sokak: {name}\nHız: {hiz_gosterim} km/s"} 
    ))
with col2:
    st.subheader("📈 Seçili Yol Segmenti İzleme")
    # Kullanıcının haritadan seçtiği veya listeden seçtiği bir sokağın zaman serisi
    secilen_sokak = st.selectbox("Sokak Seçin", ["Büyükdere Cd.", "Barbaros Blv.", "Piyale Paşa Blv."])
    
    st.info("💡 Sistem, denetimsiz K-Means algoritması ile bu güzergahta darboğaz riski tespit etmiştir.")