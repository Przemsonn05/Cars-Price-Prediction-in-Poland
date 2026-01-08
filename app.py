import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);}
.card {
    padding:30px;border-radius:15px;background:rgba(255,255,255,0.08);
    text-align:center;transition:0.3s;margin-bottom:20px;
}
.card:hover {transform:translateY(-4px);background:rgba(255,255,255,0.18);}
h1,h2,h3,h4,p,label,.stMarkdown {color:white!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("models/XGBoost_model.joblib")

model = load_model()

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go(page):
    st.session_state.page = page
    st.rerun()

def home():
    st.markdown(
    "<h1 style='text-align: center;'>🚗 Car Price Prediction</h1>",
    unsafe_allow_html=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h3>🔮 Predict Car Price</h3><p>Estimate vehicle market value</p></div>", unsafe_allow_html=True)
        if st.button("Go to Prediction", use_container_width=True): go("predict")

    with col2:
        st.markdown("<div class='card'><h3>🧠 About the Model</h3><p>Understand ML logic</p></div>", unsafe_allow_html=True)
        if st.button("Learn more", use_container_width=True): go("info")

    st.divider()

    st.markdown("<h2 style='text-align:center;'>📊 About the Project</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:900px;margin:auto;text-align:center;font-size:18px;line-height:1.6;">

    <b>Car Price Prediction Project</b> focuses on forecasting used vehicle prices on the Polish automotive market using modern machine learning techniques.

    Multiple regression models were tested during development, including <b>Linear Regression</b>, <b>Random Forest</b> and <b>XGBoost</b>. 
    After extensive evaluation, XGBoost was selected as the final production model due to its superior accuracy and robustness.

    A comprehensive market analysis was performed together with advanced <b>data preprocessing, data cleaning, outlier handling and feature engineering</b> to ensure high model quality.

    It is important to note that not all vehicles follow strict market pricing patterns.  
    The model may be less accurate for rare vehicles, special editions and supercars, where prices are highly volatile and depend on individual configurations and collector demand.

    To validate the model and demonstrate its real-world usability, an interactive web application was created that allows users to predict car prices and understand how the model works.

    </div>
    """, unsafe_allow_html=True)

def predict():
    st.title("🔮 Predict Car Price")
    if st.button("← Back"): go("home")

    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Production Year", 1980, 2024, 2015)
        mileage = st.number_input("Mileage (km)", 0, 1_000_000, 120000)
        power = st.number_input("Power (HP)", 50, 800, 150)
        displacement = st.number_input("Displacement (cm³)", 800, 6000, 2000)
        vehicle_age = st.number_input("Vehicle Age (years)", 0, 50, 5)
        annual_mileage = st.number_input("Annual Mileage (km)", 0, 100_000, 12_000)

    with c2:
        brand = st.selectbox("Brand", ["BMW","Audi","Mercedes","Toyota","Volkswagen","Ford","Other"])
        model_name = st.text_input("Vehicle Model", "Unknown")
        fuel = st.selectbox("Fuel", ["Petrol","Diesel","Hybrid","Electric","LPG"])
        gearbox = st.selectbox("Transmission", ["Manual","Automatic"])
        body = st.selectbox("Body Type", ["Sedan","SUV","Hatchback","Combi","Coupe"])
        color = st.selectbox("Color", ["Black","White","Grey","Silver","Blue","Red","Other"])
        drive = st.selectbox("Drive", ["FWD","RWD","AWD","4x4"])
        first_owner = st.selectbox("First Owner?", ["Yes","No"])
        condition = st.selectbox("Condition", ["New","Used","Very Good","Excellent","Poor"])
        is_popular_color = st.selectbox("Popular Color?", ["Yes","No"])
        is_premium_car = st.selectbox("Premium Car?", ["Yes","No"])
        origin_country = st.text_input("Origin Country", "Poland")
        offer_location = st.text_input("Offer Location", "City")
        num_features = st.number_input("Number of Features", 0, 50, 5)
        features = st.text_input("Features (comma separated)", "")
        age_category = st.selectbox("Age category", ["New", "Young", "Middle aged", "Old"])

    if st.button("Predict Price 💰", use_container_width=True):
        try:
            vehicle_age = 2024 - year
            annual_mileage = mileage / max(vehicle_age, 1)
            power_per_liter = power / max(displacement / 1000, 0.1)

            data = {
                "Condition": condition,
                "Vehicle_brand": brand,
                "Vehicle_model": model_name,
                "Production_year": year,
                "Mileage_km": mileage,
                "Power_HP": power,
                "Displacement_cm3": displacement,
                "Fuel_type": fuel,
                "Drive": drive,
                "Transmission": gearbox,
                "Type": body,
                "Colour": color,
                "Origin_country": origin_country,
                "First_owner": first_owner,
                "Offer_location": offer_location,
                "Features": features if features else "None",
                "Num_features": num_features,
                "Vehicle_age": vehicle_age,
                "Annual_mileage": annual_mileage,
                "Age_category": age_category,
                "Is_popular_color": 1 if is_popular_color == "Yes" else 0,
                "Is_premium_car": 1 if is_premium_car == "Yes" else 0,
                "Power_per_liter": power_per_liter
            }

            df = pd.DataFrame([data])

            y_log = model.predict(df)[0]
            price = np.expm1(y_log)

            st.success(f"Estimated value: {price:,.0f} PLN")

        except Exception as e:
            st.error(str(e))

def info():
    st.title("🧠 How the Model Works")
    if st.button("← Back"): go("home")

    st.markdown("""
### Model Overview
This project uses an **XGBoost regression model** trained on the Polish car market to predict vehicle prices.  
The goal is to provide reliable price estimates for used cars based on specifications, condition, and market data, helping users, dealerships, and analysts understand car valuations.

---

### Why XGBoost?
XGBoost was chosen for its **high performance on tabular data** and ability to capture **complex, non-linear relationships**.  
Key strengths include:

- **Gradient Boosting framework**: Sequentially trains weak learners to minimize errors, improving predictive accuracy.  
- **Robust to missing values and outliers**: Automatically handles incomplete or noisy data.  
- **Regularization**: Controls model complexity to prevent overfitting.  
- **Fast and scalable**: Can efficiently train on large datasets, making it suitable for real-world applications.  
- **Widely adopted in industry**: XGBoost is a go-to choice for structured datasets due to its combination of speed and accuracy.

---

### Features Used
The model considers a comprehensive set of features covering car specifications, usage, and market context:

- **Vehicle information**: Vehicle_brand, Vehicle_model, Vehicle_age, Production_year  
- **Performance & usage**: Mileage_km, Annual_mileage, Power_HP, Power_per_liter, Displacement_cm3  
- **Specifications**: Fuel_type, Transmission, Drive, Type (body), Colour  
- **Market context**: Offer_location, Origin_country, Condition, First_owner  
- **Optional features**: Features (list of installed features), Num_features (count of features)  
- **Special indicators**: Is_popular_color, Is_premium_car, Age_category

> ⚠️ Some features may not always be available or applicable. The model is robust but cannot guarantee perfect predictions for highly unique or rare vehicles.

---

### Model Performance (Test Set)
| Metric | Value |
|--------|-------|
| R²     | 94.5% |
| RMSE   | 14,727 PLN |
| MAE    | 7,023 PLN |
| MAPE   | 15.96% |

**Interpretation:**
- **R² = 94.5%**: The model explains almost all the variance in car prices, showing a strong fit.  
- **RMSE & MAE**: Indicate typical prediction errors in PLN; the model provides reliable estimates for most vehicles.  
- **MAPE ~16%**: On average, predictions deviate by ~16% from actual prices, with larger errors likely for rare or premium cars.

---

### Key Insights & Notes
- Best performance is observed for **mass-market vehicles**; rare, collector, or luxury cars often have highly variable prices.  
- The model captures **non-linear dependencies** between features, such as how mileage, age, and engine power interact with brand and model.  
- Feature importance analysis can identify which factors **most influence car prices**, supporting both predictive and analytical tasks.  
- The model is **intended as a guidance tool**, not a legally binding valuation — results should be interpreted in context.

---

### Business & Application Relevance
This project demonstrates how machine learning can support:

- **Dealership pricing strategies**: Quickly estimate competitive car prices.  
- **Consumer guidance**: Assist buyers and sellers in understanding vehicle market value.  
- **Market analytics**: Identify trends in vehicle depreciation and factors influencing pricing.  

The web application provides an **interactive interface** to predict car prices and explore the factors driving model decisions, making advanced ML insights accessible to non-technical users.
""")


if st.session_state.page == "home": home()
elif st.session_state.page == "predict": predict()
elif st.session_state.page == "info": info()