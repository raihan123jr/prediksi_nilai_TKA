import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Nilai TKA")

model = joblib.load("model.joblib")

st.title("Prediksi Nilai TKA")
st.markdown("Aplikasi machine learning untuk memprediksi nilai TKA")

jam_belajar_per_hari = st.slider("Jam Belajar Per Hari", 1, 24, 8)
persen_kehadiran = st.slider("Persen Kehadiran", 0, 100, 50)
bimbel = st.pills("Bimbel", ["ya", "tidak"], default="tidak")

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame(
        [[jam_belajar_per_hari, persen_kehadiran, bimbel]],
        columns=["jam_belajar_per_hari", "persen_kehadiran", "bimbel"]
    )

    prediksi = model.predict(data_baru)[0]
    prediksi = prediksi.clip(0, 100)

    st.success(f"Model memprediksi **Nilai TKA = {prediksi:.0f}**")
    st.balloons()
