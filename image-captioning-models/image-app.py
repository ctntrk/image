import streamlit as st
from models.model_utils import generate_caption
import os
from PIL import Image
import io

# Hakkında Bölümü
with st.expander("📌 Hakkında"):
    st.markdown("""
    **Bu AI araç** açık kaynaklı image-to-text modelini temel alan Salesforce/blip-image-captioning-base ve Salesforce/blip-image-captioning-large modelleri kullanılarak oluşturulmuştur. Bu sistem girdi olarak resim alan sonrasında image-to-text modeli kullanarak resmin tarifini metin olarak çıktı üretmektedir. Bir resim yükleyin, gerekli parametreleri seçin ve  resmin tarif etmesini bekleyin.
    """)

# Kullanıcıların Dikkatine Bölümü
with st.expander("❗ Kullanıcıların Dikkatine"):
    st.warning("""
    - Modellerin yüklenmesi ve resim işleme süreleri cihaz performansına bağlı olarak değişebilir
    - Modelin çalışması ve çıktı üretilmesi uzun sürebilir
    - Lütfen işlem tamamlanana kadar sayfayı kapatmayınız
    - Yüksek çözünürlüklü resimler daha uzun süre işlem gerektirebilir
    """)

# Sidebar Kontrolleri
st.sidebar.header("Ayarlar")
model_type = st.sidebar.selectbox("Model Tipi", ["base", "large"], 
                                help="Büyük model daha doğru ancak daha yavaş çalışır")
max_length = st.sidebar.slider("Maksimum Uzunluk", 10, 100, 50,
                             help="Üretilen metnin maksimum kelime uzunluğunu belirler. Daha yüksek değerler daha uzun ancak daha yavaş çıktılar üretebilir")
num_beams = st.sidebar.slider("Beam Sayısı", 1, 7, 5,
                            help="Arama genişliğini kontrol eder. Yüksek değerler daha iyi sonuçlar ancak daha yavaş işlem demektir")
repetition_penalty = st.sidebar.slider("Tekrar Cezası", 1.0, 2.0, 1.5,
                                     help="Aynı kelimelerin tekrarını cezalandırır. Yüksek değerler tekrarları azaltır")
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7,
                              help="Rastgelelik seviyesi. Düşük değerler daha odaklı, yüksek değerler daha yaratıcı çıktılar üretir")

# Ana İçerik
st.header("Resimden Metin Üretici")
uploaded_file = st.file_uploader("Lütfen bir resim yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Görüntüyü işleme
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        
        # Geçici dosya oluşturma (gerekiyorsa)
        temp_path = None
        if not hasattr(generate_caption, 'supports_bytes'):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Alt başlık oluşturma
        with st.spinner("Resim analiz ediliyor, lütfen bekleyin..."):
            caption = generate_caption(
                temp_path if temp_path else image,
                model_type=model_type,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                temperature=temperature
            )
        
        # Sonuçları gösterme
        st.subheader("Oluşturulan Açıklama")
        st.image(image, use_column_width=True)
        st.success(caption)
        
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
    finally:
        # Geçici dosyayı temizleme
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path) 
