import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, FEATURE_COLUMNS

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements to predict the Iris species.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python -m src.train` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1)
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1)

if st.button("Predict Species", type="primary"):
    sample = pd.DataFrame([{
        "SepalLengthCm": sepal_length,
        "SepalWidthCm": sepal_width,
        "PetalLengthCm": petal_length,
        "PetalWidthCm": petal_width,
    }])

    prediction = model.predict(sample)[0]
    st.success(f"Predicted species: **{prediction}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        probability_df = pd.DataFrame({
            "Species": bundle["classes"],
            "Probability": probs
        }).sort_values("Probability", ascending=False)
        st.bar_chart(probability_df.set_index("Species"))
