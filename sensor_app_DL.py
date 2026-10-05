import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Motion Sensor Classification App", layout="centered")

st.title("Motion Sensor Classification App")
st.write(
    "Bu uygulama, ivmeölçer ve jiroskop sensör verilerine dayanarak sürüş/hareket davranışlarını (örneğin normal veya agresif) Yapay Sinir Ağı (ANN) modeli ile sınıflandırır."
)

@st.cache_resource
def load_nn_model():
    return load_model("motion_ann_model_optimized.h5")

model = load_nn_model()

st.subheader("Sensor Degerlerini Giriniz:")
# Not: Gercek projede egitim veri setindeki tum sensor oznitelikleri (features) buraya eklenmelidir.
feature_1 = st.number_input("Sensor Ozelligi 1", value=0.0)
feature_2 = st.number_input("Sensor Ozelligi 2", value=0.0)
feature_3 = st.number_input("Sensor Ozelligi 3", value=0.0)

if st.button("Haraket Sinifini Tahmin Et", type="primary"):
    try:
        # Ornek bir girdi matrisi olusturma (Modelin egitildigi shape'e uygun olmalidir)
        input_data = np.array([[feature_1, feature_2, feature_3] + [0.0] * 15]) # Beklenen oznitelik sayisina gore tamamlanabilir
        
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        st.success(f"Tahmin Sonucu: Model bu hareketi **Class {predicted_class}** olarak sınıflandırdı.")
    except Exception as e:
        st.error(f"Tahmin sirasinda bir hata olustu: {e}")