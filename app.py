import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import torch
import joblib
from model import MultiStreamGNNLSTM
from utils import get_device
from analiz_ve_harita import tersine_eslestirme_yap, kronik_darbogazlari_bul

# 1. Sayfa Ayarları
st.set_page_config(page_title="Traffix Akıllı Trafik İzleme Paneli", layout="wide")
st.title("🚦 Traffix: Trafik İzleme Paneli")

# 2. Statik Model ve Ham Verileri Önbelleğe Alma
@st.cache_resource
def load_static_model_and_data():
    device = get_device()
    dataset_dict = torch.load("gnn_lstm_dataset_6ay.pt", map_location="cpu")
    x_raw = dataset_dict["x"].float().to(device)
    edge_index = dataset_dict["edge_index"].long().to(device)
    edge_weight = dataset_dict["edge_weight"].float().to(device)
    scaler = joblib.load('traffic_scaler.pkl')

    HORIZON = 6
    # Eğitilen 16 ve 32 katman boyutlarına sahip model yapısı
    model = MultiStreamGNNLSTM(
        input_features=x_raw.shape[-1], gnn_hidden=16, lstm_hidden=32, output_features=2, horizon=HORIZON
    ).to(device)
    model.load_state_dict(torch.load("best_gnn_lstm_model.pt", map_location=device))
    model.eval()

    t = x_raw.shape[0]
    window_size = 3

    x_recent = x_raw[t - window_size : t].unsqueeze(0)
    x_daily  = x_raw[t - 24 - (window_size // 2) : t - 24 + (window_size // 2) + (window_size % 2)].unsqueeze(0)
    x_weekly = x_raw[t - 168 - (window_size // 2) : t - 168 + (window_size // 2) + (window_size % 2)].unsqueeze(0)

    # 🌟 DİNAMİK MASKE: Zaman ve özellik boyutlarında toplayarak hiç verisi olmayan düğümleri tespit etme
    gecerli_dugum_maskesi = (x_raw.abs().sum(dim=(0, 2)) > 1e-6).cpu().numpy()

    return model, x_recent, x_daily, x_weekly, edge_index, edge_weight, scaler, HORIZON, gecerli_dugum_maskesi

model, x_recent, x_daily, x_weekly, edge_index, edge_weight, scaler, HORIZON, gecerli_dugum_maskesi = load_static_model_and_data()

# 🌟 HIZ OPTİMİZASYONU: Cache mimarisi
@st.cache_resource
def get_cached_predictions(use_recent, use_daily, use_weekly, _model, _x_recent, _x_daily, _x_weekly, _edge_index, _edge_weight, _scaler, horizon, _maske):
    x_r = _x_recent if use_recent else torch.zeros_like(_x_recent)
    x_d = _x_daily if use_daily else torch.zeros_like(_x_daily)
    x_w = _x_weekly if use_weekly else torch.zeros_like(_x_weekly)
    
    with torch.no_grad():
        tum_ufuk_tahmini = _model(x_r, x_d, x_w, _edge_index, _edge_weight)
        
    tahminler_sozlugu = {}
    for saat_ileri in range(1, horizon + 1):
        tahmin_tensoru = tum_ufuk_tahmini[0, saat_ileri - 1, :, :]
        # Maskeyi eşleştirme fonksiyonuna gönderiyoruz
        harita_gdf = tersine_eslestirme_yap(tahmin_tensoru, _scaler, _maske)
        darbogazlar_gdf = kronik_darbogazlari_bul(harita_gdf, kume_sayisi=3)
        
        tahminler_sozlugu[saat_ileri] = {
            'harita': harita_gdf, 'darbogazlar': darbogazlar_gdf
        }
    return tahminler_sozlugu

# 3. Kullanıcı Kontrolleri (Sol Panel)
st.sidebar.header("🕹️ Kontrol Paneli")
secilen_zaman = st.sidebar.slider("Tahmin Ufku (Saat)", min_value=1, max_value=6, value=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Zamansal Bağlam Kolları")
st.sidebar.caption("Yapay zekanın hangi pencereleri hesaba katacağını seçin:")
use_recent = st.sidebar.checkbox("Kısa Dönem (Son 3 Saat şoku)", value=True)
use_daily = st.sidebar.checkbox("Günlük Döngü (Dün Aynı Saat)", value=True)
use_weekly = st.sidebar.checkbox("Haftalık Döngü (Geçen Hafta)", value=True)

gosterim_modu = st.sidebar.radio("Harita Modu", ["Isı Haritası (Genel Trafik)", "Sadece Darboğazlar (Uyarı)"])

tahminler_sozlugu = get_cached_predictions(
    use_recent, use_daily, use_weekly, model, x_recent, x_daily, x_weekly, edge_index, edge_weight, scaler, HORIZON, gecerli_dugum_maskesi
)

harita_gdf = tahminler_sozlugu[secilen_zaman]['harita']
darbogazlar_gdf = tahminler_sozlugu[secilen_zaman]['darbogazlar']

# 4. Görselleştirme Arayüzü
col1, col2 = st.columns([2, 1])

def hiz_rengi_belirle(hiz):
    if pd.isna(hiz) or hiz < 5.0: return [128, 128, 128, 50]   # Verisiz yollar: Silik Gri
    elif hiz < 25.0: return [227, 26, 28, 255]                 # Koyu Kırmızı
    elif hiz < 40.0: return [253, 141, 60, 255]                # Turuncu
    elif hiz < 55.0: return [254, 204, 92, 255]                # Sarı
    else: return [35, 139, 69, 255]                            # Koyu Yeşil

def darbogaz_rengi_kalinligi_belirle(satir):
    if satir['trafik_durumu_kumesi'] == 0:
        return [255, 0, 0, 255], 18                            # Kırmızı ve Kalın
    return [128, 128, 128, 50], 4                              # Diğer Yollar: Silik Gri ve İnce

with col1:
    st.subheader(f"🗺️ İstanbul Tıkanıklık Yayılımı (+{secilen_zaman} Saat)")
    
    harita_gdf['hiz_gosterim'] = harita_gdf['tahmini_hiz_kmh'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Veri Yok")
    harita_gdf['cizgi_rengi'] = harita_gdf['tahmini_hiz_kmh'].apply(hiz_rengi_belirle)

    if gosterim_modu == "Isı Haritası (Genel Trafik)":
        layer = pdk.Layer(
            "GeoJsonLayer", 
            data=harita_gdf, 
            get_line_color="cizgi_rengi", 
            get_line_width=10, 
            pickable=True
        )
    else:
        # Darboğaz modunda tüm yolları veriyoruz ama renk/kalınlık dinamik atanıyor
        darbogazlar_gdf[['cizgi_rengi', 'cizgi_kalinligi']] = darbogazlar_gdf.apply(darbogaz_rengi_kalinligi_belirle, axis=1, result_type='expand')
        darbogazlar_gdf['hiz_gosterim'] = darbogazlar_gdf['tahmini_hiz_kmh'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Veri Yok")
        
        layer = pdk.Layer(
            "GeoJsonLayer", 
            data=darbogazlar_gdf, 
            get_line_color="cizgi_rengi", 
            get_line_width="cizgi_kalinligi", 
            pickable=True
        )

    st.pydeck_chart(pdk.Deck(
        map_style="carto-dark",
        initial_view_state=pdk.ViewState(latitude=41.06, longitude=29.0, zoom=12),
        layers=[layer],
        tooltip={"text": "Sokak: {name}\nHız: {hiz_gosterim} km/s"}
    ))

with col2:
    st.subheader("📈 Gelişmiş Şehir Analitiği")
    
    gecerli_sokaklar = harita_gdf[harita_gdf['name'].notna()]
    sokak_isimleri = gecerli_sokaklar['name'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    sokak_listesi = sokak_isimleri.unique()
    
    varsayilan_index = list(sokak_listesi).index("Büyükdere Caddesi") if "Büyükdere Caddesi" in sokak_listesi else 0
    
    secilen_sokak = st.selectbox("İzlemek İstediğiniz Güzergahı Seçin:", options=sokak_listesi, index=varsayilan_index)
    
    harita_isim_temiz = harita_gdf.copy()
    harita_isim_temiz['name'] = harita_isim_temiz['name'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    
    sokak_verisi = harita_isim_temiz[harita_isim_temiz['name'] == secilen_sokak]
    ortalama_hiz = sokak_verisi['tahmini_hiz_kmh'].mean()
    hiz_metni = f"{ortalama_hiz:.1f} km/s" if pd.notna(ortalama_hiz) else "Veri Yok"
    
    kume_darbogaz_mi = (sokak_verisi['trafik_durumu_kumesi'] == 0).any() if 'trafik_durumu_kumesi' in sokak_verisi.columns else False
    
    # 🌟 YENİ: Renk (Hız) Kategorilerine Göre Detaylı Durum Belirleme
    durum_kategorisi = ""
    
    if pd.isna(ortalama_hiz) or ortalama_hiz < 5.0:
        durum_kategorisi = "VERI_YOK"
    elif ortalama_hiz < 25.0 or kume_darbogaz_mi:
        durum_kategorisi = "KIRMIZI"
    elif ortalama_hiz < 40.0:
        durum_kategorisi = "TURUNCU"
    elif ortalama_hiz < 55.0:
        durum_kategorisi = "SARI"
    else:
        durum_kategorisi = "YESIL"

    # Metrik Kartı Renk ve Yön Ayarları
    if durum_kategorisi == "VERI_YOK":
        delta_val = None
        delta_color = "off"
    elif durum_kategorisi == "KIRMIZI":
        delta_val = "Kilitli / Darboğaz"
        delta_color = "normal"  # Eksi (-) kullanıldığı için Streamlit kırmızı renderlar
    elif durum_kategorisi == "TURUNCU":
        delta_val = "Yoğun Akış"
        delta_color = "normal"  # Eksi (-) kırmızı
    elif durum_kategorisi == "SARI":
        delta_val = "Orta Seviye Akış"
        delta_color = "normal"     # Yönsüz, nötr gri renk
    else:
        delta_val = "Akıcı Akış"
        delta_color = "normal"  # Artı (+) yeşil

    # Metrik Kartı Gösterimi
    st.metric(
        label=f"{secilen_zaman} Saat Sonra {secilen_sokak} Durumu", 
        value=hiz_metni, 
        delta=delta_val,
        delta_color=delta_color
    )
    
    # 🌟 YENİ: Uyarı Kutuları Yönetimi (Renklere Özel)
    if durum_kategorisi == "VERI_YOK":
        st.warning(f"⚪ **{secilen_sokak}** güzergahı için sensör bilgisi bulunmamaktadır (veya hız 5 km/s altındadır).")
    elif durum_kategorisi == "KIRMIZI":
        st.error(f"🚨 KRİTİK (Koyu Kırmızı): {secilen_zaman} saat sonra **{secilen_sokak}** üzerinde trafik durma noktasına gelecektir. Alternatif rotaları değerlendirin! (Tahmini Hız: {ortalama_hiz:.1f} km/s)")
    elif durum_kategorisi == "TURUNCU":
        st.warning(f"⚠️ DİKKAT (Turuncu): **{secilen_sokak}** güzergahında belirgin bir yoğunluk bekleniyor. Seyahat süreniz uzayabilir. (Tahmini Hız: {ortalama_hiz:.1f} km/s)")
    elif durum_kategorisi == "SARI":
        st.info(f"ℹ️ BİLGİ (Mavi): **{secilen_sokak}** üzerinde akış devam ediyor ancak yer yer hafif yavaşlamalar görülebilir. (Tahmini Hız: {ortalama_hiz:.1f} km/s)")
    else: # YESIL
        st.success(f"✅ HARİKA (Yeşil): **{secilen_sokak}** güzergahının {secilen_zaman} saat sonra tamamen temiz ve akıcı olması öngörülmektedir.")

    