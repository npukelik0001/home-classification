import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define three tabs now: Manual Input, CSV Upload, Data Insights
tab1, tab2, tab3 = st.tabs(["Manual Input", "CSV Upload", "Data Insights"])

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

# ----- Data Insights Tab -----
with tab3:
    st.header("Exploratory Data Analysis & Visualizations")

    df_cleaned = pd.read_csv("df_cleaned.csv")

    # Use default style
    plt.style.use('default')

    # Plot 1: Target Distribution Pie Chart
    fig1, ax1 = plt.subplots(figsize=(6,6))
    target_counts = df_cleaned['expensive_home'].value_counts()
    labels = ['Not Expensive (0)', 'Expensive (1)']
    ax1.pie(target_counts.values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    ax1.set_title('Distribution of Expensive vs Not Expensive Homes')
    st.pyplot(fig1)

    # Plot 2: Bar and Box Plots side by side
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(x='expensive_home', data=df_cleaned, palette='pastel', ax=axes[0])
    axes[0].set_title('Distribution of Expensive vs Non-Expensive Homes')
    axes[0].set_xlabel('Expensive Home (0 = No, 1 = Yes)')
    axes[0].set_ylabel('Count')

    sns.boxplot(data=df_cleaned, x='expensive_home', y='sqft_living', palette='pastel', ax=axes[1])
    axes[1].set_title('Living Space by Expensive Classification')
    axes[1].set_xlabel('Expensive Home (0 = No, 1 = Yes)')
    axes[1].set_ylabel('sqft_living')

    plt.tight_layout()
    st.pyplot(fig2)

    # Summary stats
    st.subheader("Expensive Home Statistics (by Class):")
    st.write(df_cleaned['expensive_home'].value_counts())
    st.write("Proportions:")
    st.write(df_cleaned['expensive_home'].value_counts(normalize=True).round(2))

    st.subheader("Visual Interpretations:")
    st.markdown("""
    1. Around 59% of the homes are classified as expensive, showing a slightly imbalanced but acceptable target distribution.  
    2. Expensive homes tend to have significantly larger living space, with higher median and wider spread in sqft_living.  
    3. The expensive category contains more extreme outliers in square footage, which may influence model performance.
    """)

    # Create sqft_bin column for binning sqft_living
    df_cleaned['sqft_bin'] = pd.cut(df_cleaned['sqft_living'],
                                    bins=[0, 1000, 1500, 2000, 2500, 3000, 4000, 6000, np.inf],
                                    labels=['<1000', '1000–1500', '1500–2000', '2000–2500',
                                            '2500–3000', '3000–4000', '4000–6000', '6000+'])

    # Calculate proportion of expensive homes per bin
    bin_summary = df_cleaned.groupby('sqft_bin')['expensive_home'].mean().reset_index()

    # Plot 3: Proportion of Expensive Homes by Square Footage
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.barplot(data=bin_summary, x='sqft_bin', y='expensive_home', palette='pastel', ax=ax3)
    ax3.set_title('Proportion of Expensive Homes by Square Footage')
    ax3.set_xlabel('Square Footage Bins')
    ax3.set_ylabel('Proportion of Homes Classified as Expensive')
    ax3.set_ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

    st.subheader("Visual Interpretations:")
    st.markdown("""
    1. Larger homes (especially above 4000 sq ft) are overwhelmingly classified as expensive.  
    2. There's a sharp increase in expensive classification starting around 2500–3000 sq ft.  
    3. Homes under 1500 sq ft are rarely considered expensive, typically under 20%.
    """)

    # Plot 4: Proportion of Expensive Homes by Grade
    fig4, ax4 = plt.subplots(figsize=(10,6))
    df_cleaned.groupby('grade')['expensive_home'].mean().plot(
        kind='bar', color='skyblue', edgecolor='black', ax=ax4
    )
    ax4.set_title('Proportion of Expensive Homes by Grade')
    ax4.set_xlabel('House Grade')
    ax4.set_ylabel('Proportion of Expensive Homes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig4)

    st.subheader("Visual Interpretations:")
    st.markdown("""
    1. Homes with grades 10 and above are almost always classified as expensive, with proportions close to 1.  
    2. Mid-tier grades like 8 and 9 show a mix of expensive and non-expensive homes, while grades below 7 are rarely classified as expensive.  
    3. The steep increase in proportion between grades 8 and 9 suggests that this threshold may be critical for predicting expensive homes.
    """)

# Footer
st.markdown("---")
st.caption("Model trained on KC House dataset (2000–2015 homes). Features include structure, view, and location (latitude).")
