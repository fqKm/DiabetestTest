import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

rfc = load_model()

# Streamlit App
st.title("Aplikasi Diabetes")
st.write("Cek kemungkinan diabetes berdasarkan data kesehatan Anda.")

# Input Form
st.sidebar.header("Masukkan Data")
age = st.sidebar.number_input("Umur", min_value=0, step=1, value=0)
bs_fast = st.sidebar.number_input("Gula Darah Puasa (%)", min_value=0.0, step=0.1, value=0.0)
bs_pp = st.sidebar.number_input("Gula Darah Setelah Makan (%)", min_value=0.0, step=0.1, value=0.0)
plasma_r = st.sidebar.number_input("Glukosa Plasma Acak (mmol/L)", min_value=0.0, step=0.1, value=0.0)
plasma_f = st.sidebar.number_input("Glukosa Plasma Puasa (mmol/L)", min_value=0.0, step=0.1, value=0.0)
hba1c = st.sidebar.number_input("Hemoglobin A1c (mmol/mol)", min_value=0.0, step=0.1, value=0.0)

# Collect input data into a dictionary
input_data = {
    "Age": age,
    "BS Fast": bs_fast,
    "BS pp": bs_pp,
    "Plasma R": plasma_r,
    "Plasma F": plasma_f,
    "HbA1c": hba1c,
}

# Convert input data into a DataFrame
df_input = pd.DataFrame([input_data])

# Prediction Button
if st.sidebar.button("Submit"):
    try:
        # Ensure data types are correct for the model
        df_input = df_input.astype(float)

        # Predict the result
        prediction = rfc.predict(df_input)[0]
        probabilities = rfc.predict_proba(df_input)[0]
        probability_dictionary = {label: round(probability * 100, 2) for label, probability in zip(rfc.classes_, probabilities)}

        # Display the results
        st.header("Hasil Prediksi")
        if prediction in ["Type1", "Type2"]:
            st.error(f"Anda Terdiagnosa Diabetes: **{prediction}**")
        else:
            st.success("Anda Tidak Terdiagnosa Diabetes")

        st.subheader("Persentase Probabilitas:")
        for label, probability in probability_dictionary.items():
            st.write(f"- **{label}**: {probability}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.write("Masukkan data pada sidebar dan klik 'Submit' untuk melihat hasil prediksi.")
