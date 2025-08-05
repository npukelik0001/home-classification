
# Expensive Home Classifier

This project applies machine learning classification models to predict whether a home in King County, WA is considered expensive based on its physical features and location. Developed for CIS 9660 - Data Mining for Business Analytics and deployed using Streamlit.

---

## Overview

The goal of this project is to assist homeowners, real estate professionals, and housing analysts in quickly assessing whether a property is likely to be considered expensive. The classification is based on house structure, view, grade, and location features.

The target variable is binary:
- 1 = Expensive home
- 0 = Not expensive

---

## Dataset

- **Source**: [King County House Sales (Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Size**: ~21,000 residential home sales from 2000–2015
- **Key Features**:
  - Bedrooms, bathrooms
  - Square footage (above, basement, total, neighbors)
  - View, waterfront, grade
  - Latitude
  - Year built

---

## Data Preprocessing

- Removed irrelevant or missing values
- Converted continuous target variable (price) to binary class: expensive or not
- Standardized numeric features using `StandardScaler`
- Selected 12 final features for modeling

---

## Model Development

The following classification models were trained and evaluated using 5-fold cross-validation:

- Logistic Regression  
- Naive Bayes  
- Decision Tree  
- Random Forest (best performer)  
- Support Vector Machine  
- K-Nearest Neighbors  
- K-Means (unsupervised reference)

**Best Model**: Random Forest  
- Accuracy: 91.78%  
- Weighted F1 Score: 0.92  
- Final model saved as `random_forest_model.pkl`

---

## Streamlit App Features

- **Manual Input**: Users can enter home features through sliders and dropdowns
- **CSV Upload**: Upload a file with multiple homes to classify in bulk
- **Output**: Binary prediction and confidence score
- **Scalable**: Can be extended to include zip code, renovation status, etc.

---

## Repository Contents

- `app.py` — Streamlit web application  
- `HouseSales.ipynb` — Notebook with full model training and evaluation  
- `df_cleaned.csv` — Cleaned dataset used for modeling  
- `random_forest_model.pkl` — Trained Random Forest classifier  
- `scaler.pkl` — StandardScaler used during training  
- `requirements.txt` — Python package list  
- `Technical_Report.pdf` — Final 1-page report with visualizations and summary  

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Deployed Web App

Use the classifier live on Streamlit Cloud:  
**[Streamlit App URL](https://your-streamlit-app-url)**  
*(Replace this link with your actual deployment URL)*

---

## Author

**Nastassia Pukelik**  
Baruch College – CIS 9660  
Summer 2025
