import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define final feature list
feature_names = [
    'bedrooms', 'bathrooms', 'sqft_living', 'waterfront', 'view',
    'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'lat',
    'sqft_living15', 'sqft_lot15'
]

# Streamlit UI
st.title("Expensive Home Classifier")
st.write("This app uses a trained Random Forest model to classify whether a house is considered expensive based on its features.")

# Tabs for manual input and CSV upload
tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])

# ----- Manual Input Tab -----
with tab1:
    st.header("Manual Entry")

    def user_input_features():
        data = {}
        data['bedrooms'] = st.slider("Bedrooms", 0, 10, 3)
        data['bathrooms'] = st.slider("Bathrooms", 0, 8, 2)
        data['sqft_living'] = st.slider("Sqft Living", 0, 15000, 2000)
        data['waterfront'] = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        data['view'] = st.slider("View (0 = none, 4 = great)", 0, 4, 0)
        data['grade'] = st.slider("Grade", 1, 13, 7)
        data['sqft_above'] = st.slider("Sqft Above", 0, 10000, 2000)
        data['sqft_basement'] = st.slider("Sqft Basement", 0, 5000, 500)
        data['yr_built'] = st.selectbox("Year Built", list(range(2000, 2016)))
        data['lat'] = st.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.5, step=0.001)
        data['sqft_living15'] = st.number_input("Sqft Living (15 nearest)", value=2000.0)
        data['sqft_lot15'] = st.number_input("Sqft Lot (15 nearest)", value=5000.0)
        return pd.DataFrame([data])

    manual_input_df = user_input_features()

    if st.button("Predict (Manual Input)"):
        scaled_input = scaler.transform(manual_input_df)
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0][1]

        st.success(f"Prediction: {'Expensive Home (1)' if prediction == 1 else 'Not Expensive (0)'}")
        st.info(f"Confidence: {proba:.2%}")

# ----- CSV Upload Tab -----
with tab2:
    st.header("Upload CSV for Bulk Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)

        if all(feature in input_df.columns for feature in feature_names):
            scaled_input = scaler.transform(input_df[feature_names])
            predictions = model.predict(scaled_input)
            probs = model.predict_proba(scaled_input)[:, 1]

            input_df["Prediction"] = predictions
            input_df["Confidence"] = probs

            st.write("Predictions:")
            st.dataframe(input_df)

            st.download_button("Download Results", input_df.to_csv(index=False), "predictions.csv")
        else:
            st.error("Your CSV is missing one or more required columns.")

    st.markdown("Download the CSV template from your instructor or project guide.")

# Footer
st.markdown("---")
st.caption("Model trained on KC House dataset (2000–2015 homes). Features include structure, view, and location (latitude).")
