import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);}
.card {
    padding:30px; border-radius:15px; background:rgba(255,255,255,0.08);
    text-align:center; transition:0.3s; margin-bottom:20px;
}
.card:hover {transform:translateY(-4px); background:rgba(255,255,255,0.18);}
h1,h2,h3,h4,p,label,.stMarkdown {color:white!important;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Przemsonn/poland-car-price-model",
        filename="final_car_price_model.joblib"
    )
    return joblib.load(model_path)

def get_brand_reliability_category(brand):
    brand_lower = str(brand).lower()
    luxury = ['ferrari', 'lamborghini', 'rolls-royce', 'bentley', 'aston martin', 'mclaren', 'maserati', 'porsche']
    american = ['ram', 'dodge', 'chevrolet', 'hummer', 'cadillac']
    vintage = ['syrena', 'nysa', 'warszawa', 'polonez', 'żuk', 'gaz', 'moskwicz', 'lada', 'wartburg', 'trabant', 'tata']
    premium_asian = ['infiniti', 'acura', 'baic', 'ssangyong']
    budget = ['dacia', 'fiat', 'daewoo', 'lancia']
    
    if brand_lower in luxury: return 'Luxury'
    if brand_lower in american: return 'American'
    if brand_lower in vintage: return 'Vintage'
    if brand_lower in premium_asian: return 'Premium_Asian'
    if brand_lower in budget: return 'Budget'
    return 'Standard'

def prepare_input_data(user_inputs):
    df = pd.DataFrame([user_inputs])
    
    current_year = 2026
    df['Vehicle_age'] = current_year - df['Production_year']
    
    if isinstance(df['Features'].iloc[0], str):
        feat_list = [f.strip() for f in df['Features'].iloc[0].split(',') if f.strip()]
        df['Features'] = [feat_list]
    
    df['Num_features'] = df['Features'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    df['Age_category'] = pd.cut(df['Vehicle_age'], bins=[-1, 3, 10, 20, 100], labels=['New', 'Standard', 'Old', 'Vintage'])
    df['Is_new_car'] = (df['Vehicle_age'] <= 1).astype(int)
    df['Is_old_car'] = (df['Vehicle_age'] > 20).astype(int)
    
    df['Mileage_per_year'] = df['Mileage_km'] / (df['Vehicle_age'] + 1)
    df['Usage_intensity'] = df['Mileage_km'] / (df['Vehicle_age'] + 1)
    df['HP_per_liter'] = df['Power_HP'] / (df['Displacement_cm3'] / 1000 + 0.1)
    
    df['Performance_category'] = pd.cut(df['Power_HP'], bins=[-1, 100, 200, 400, 2000], labels=['Economy', 'Standard', 'Sport', 'Supercar'])
    df['Brand_category'] = get_brand_reliability_category(df['Vehicle_brand'].iloc[0])
    
    df['Is_premium'] = df['Brand_category'].isin(['Luxury', 'Premium_Asian']).astype(int)
    df['Is_supercar'] = ((df['Power_HP'] > 500) | (df['Brand_category'] == 'Luxury')).astype(int)
    df['is_collector'] = ((df['Vehicle_age'] > 30) | (df['Brand_category'] == 'Luxury')).astype(int)
    
    df['Mileage_km_log'] = np.log1p(df['Mileage_km'])
    df['Power_HP_log'] = np.log1p(df['Power_HP'])
    df['Displacement_cm3_log'] = np.log1p(df['Displacement_cm3'])
    df['Vehicle_age_squared'] = df['Vehicle_age'] ** 2
    df['Power_HP_squared'] = df['Power_HP'] ** 2
    df['Mileage_km_squared'] = df['Mileage_km'] ** 2
    
    df['Age_Mileage_interaction'] = df['Vehicle_age'] * df['Mileage_km']
    df['Power_Age_interaction'] = df['Power_HP'] * df['Vehicle_age']
    df['Mileage_per_year_Age'] = df['Mileage_per_year'] * df['Vehicle_age']
    
    df['Brand_frequency'] = 100  
    df['Brand_popularity'] = 'Common' 
    
    expected_columns = [
        'Condition', 'Vehicle_brand', 'Vehicle_model', 'Mileage_km', 'Power_HP', 
        'Displacement_cm3', 'Fuel_type', 'Drive', 'Transmission', 'Type', 
        'Doors_number', 'Colour', 'Origin_country', 'First_owner', 
        'Offer_location', 'Features', 'Vehicle_age', 'Num_features', 
        'Age_category', 'Is_new_car', 'Is_old_car', 'Mileage_per_year', 
        'Usage_intensity', 'HP_per_liter', 'Performance_category', 'Is_premium', 
        'Is_supercar', 'is_collector', 'Mileage_km_log', 'Power_HP_log', 
        'Displacement_cm3_log', 'Vehicle_age_squared', 'Power_HP_squared', 
        'Mileage_km_squared', 'Age_Mileage_interaction', 
        'Power_Age_interaction', 'Mileage_per_year_Age', 'Brand_category', 
        'Brand_frequency', 'Brand_popularity'
    ]
    return df[expected_columns]

model_data = load_model()
pipeline = model_data['model_pipeline'] if isinstance(model_data, dict) else model_data

if 'page' not in st.session_state: st.session_state.page = 'home'
def go(page): 
    st.session_state.page = page
    st.rerun()

def home():
    st.markdown("<h1 style='text-align: center;'>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)
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

def predict_page():
    st.title("🔮 Predict Car Price")
    if st.button("← Back"): go("home")

    brand_list = ['Abarth', 'Acura', 'Aixam', 'Alfa Romeo', 'Alpine', 'Aston Martin',
       'Audi', 'Austin', 'Autobianchi', 'Baic', 'Bentley', 'BMW', 'Buick',
       'Cadillac', 'Chevrolet', 'Chrysler', 'Citroën', 'Cupra', 'Dacia',
       'Daewoo', 'Daihatsu', 'DFSK', 'DKW', 'Dodge', 'DS Automobiles',
       'FAW', 'Ferrari', 'Fiat', 'Ford', 'Gaz', 'GMC', 'Honda', 'Hummer',
       'Hyundai', 'Infiniti', 'Isuzu', 'Iveco', 'Jaguar', 'Jeep', 'Kia',
       'Lada', 'Lamborghini', 'Lancia', 'Land Rover', 'Lexus', 'Lincoln',
       'Lotus', 'Warszawa', 'Maserati', 'Maybach', 'Mazda', 'McLaren',
       'Mercedes-Benz', 'Mercury', 'MG', 'Microcar', 'MINI', 'Mitsubishi',
       'Moskwicz', 'Nissan', 'NSU', 'Nysa', 'Oldsmobile', 'Opel',
       'Toyota', 'Tata', 'Uaz', 'Żuk', 'Trabant', 'Suzuki', 'Inny',
       'Volvo', 'Subaru', 'Volkswagen', 'SsangYong', 'Saab', 'Plymouth',
       'Renault', 'Peugeot', 'Rolls-Royce', 'RAM', 'Triumph', 'Rover',
       'Wołga', 'Tarpan', 'Polonez', 'Pontiac', 'Porsche', 'Santana',
       'Saturn', 'Scion', 'Seat', 'Škoda', 'Smart', 'Syrena', 'Talbot',
       'Tavria', 'Tesla', 'Vanderhall', 'Vauxhall', 'Wartburg',
       'Zaporożec', 'Zastava'] 
    country_list = ["Germany","USA","Japan","France","Italy","United Kingdom",
        "South Korea","China","Sweden","Spain","Netherlands","Belgium","Canada","Australia","India","Russia","Mexico",
        "Brazil","Czech Republic","Poland","Turkey","Austria","Switzerland","Denmark","Norway","Finland","Portugal",
        "Greece","Thailand","Vietnam"]
    fuel_list = ['Gasoline', 'Gasoline + LPG', 'Diesel', 'Hybrid', 'Gasoline + CNG', 'Hydrogen', 'Electric']
    drive_list = ['Front wheels' ,'Rear wheels', '4x4 (attached automatically)',
       '4x4 (permanent)', '4x4 (attached manually)']
    body_list = ['small_cars', 'coupe', 'city_cars', 'convertible', 'compact',
       'SUV', 'sedan', 'station_wagon', 'minivan']
    color_list = ["Black","White","Grey","Silver","Blue","Red", 'Yellow', 'Green', "Other"]

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            brand_choice = st.selectbox("Brand", brand_list, None)
            production_year = st.slider("Production Year", 1920, 2026, 1920)
            power_HP = st.slider("Power (HP)", 10, 1500, 10)
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"], None)
            car_type = st.selectbox("Body Type", body_list, None)
            fuel_type = st.selectbox("Fuel Type", fuel_list, None)
            condition = st.selectbox("Condition", ["Used", "New"], None)
            offer_location = st.text_input("Location (city, street, adress, etc.)")
            
        with c2:
            vehicle_model = st.text_input("Model Name")
            mileage_km = st.slider("Mileage (km)", 0, 1000000, 0)
            displacement_cm3 = st.slider("Displacement (cm3)", 500, 8000, 500)
            drive = st.selectbox("Drive", drive_list, None)
            colour = st.selectbox("Colour", color_list, None)
            origin_country = st.selectbox("Origin Country", country_list, None)
            first_owner = st.selectbox("First Owner", ['Yes', 'No'], None)
            features = st.text_input("Features (comma separated, e.g. ABS, GPS)")

        submit = st.form_submit_button("Predict Price 💰", use_container_width=True)

    if submit:
        try:
            user_inputs = {
                "Condition": condition, 
                "Vehicle_brand": brand_choice, 
                "Vehicle_model": vehicle_model,
                "Production_year": production_year, 
                "Mileage_km": mileage_km, 
                "Power_HP": power_HP,
                "Displacement_cm3": displacement_cm3, 
                "Fuel_type": fuel_type, 
                "Drive": drive,
                "Transmission": transmission, 
                "Type": car_type, 
                "Doors_number": 5, 
                "Colour": colour, 
                "Origin_country": origin_country, 
                "First_owner": first_owner,
                "Offer_location": offer_location, 
                "Features": features
            }
            
            final_df = prepare_input_data(user_inputs)
            y_log = pipeline.predict(final_df)[0]
            price = np.expm1(y_log)
            
            st.success(f"### Estimated Market Value: {price:,.0f} PLN")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

def info():
        st.title("🧠 How the Model Works")
        if st.button("← Back"): go("home")

        st.markdown("""
### Model Overview
This application uses an **XGBoost regression model** trained on data from the Polish automotive market to estimate **used car prices**.  
The primary goal is to provide **reliable and data-driven price predictions** based on vehicle specifications, usage patterns, and market context.  
Such estimates can support **buyers, sellers, dealerships, and analysts** in better understanding vehicle valuations and market trends.

---

### Why XGBoost?
**XGBoost** was selected due to its strong performance on **structured tabular data** and its ability to model **complex, non-linear relationships** between variables.

Key advantages include:

- **Gradient Boosting framework** – sequentially builds decision trees that minimize prediction errors and improve overall accuracy.  
- **Robustness to imperfect data** – handles missing values and noisy observations effectively.  
- **Regularization mechanisms** – built-in L1/L2 penalties help control model complexity and reduce overfitting.  
- **High computational efficiency** – optimized implementation allows fast training even on large datasets.  
- **Industry-proven approach** – widely used in machine learning competitions and production systems for structured data problems.

---

### Features Used
The model incorporates a wide range of variables describing **vehicle characteristics, usage, and market context**:

- **Vehicle information:** Vehicle_brand, Vehicle_model, Vehicle_age, Production_year  
- **Performance & usage:** Mileage_km, Annual_mileage, Power_HP, Power_per_liter, Displacement_cm3  
- **Technical specifications:** Fuel_type, Transmission, Drive, Type (body style), Colour  
- **Market context:** Offer_location, Origin_country, Condition, First_owner  
- **Optional attributes:** Features (list of installed features)
- **Engineered indicators:** New created features such as:  HP_per_liter, is_collector, Is_premium, etc.

---

### Model Performance (Test Set)

| Metric | Value |
|------|------|
| **R²** | **92.5%** |
| **RMSE** | **22,918 PLN** |
| **MAE** | **8,062 PLN** |
| **MAPE** | **17.2%** |

**Interpretation**

- **R² = 92.5%** indicates that the model explains the vast majority of variability in vehicle prices.  
- **RMSE and MAE** represent the typical magnitude of prediction errors in PLN and show that the model produces **accurate estimates for most vehicles**.  
- **MAPE ≈ 17%** means predictions deviate on average by about **17% from actual prices**, which is reasonable given the large variability of the used car market.

---

### Key Insights
- The model performs best for **mass-market vehicles**, where the dataset contains many comparable examples.  
- **Luxury, collector, or rare vehicles** may produce larger prediction errors due to limited representation in the dataset.  
- The algorithm successfully captures **non-linear interactions** between variables such as mileage, age, engine power, and vehicle brand.  
- Feature importance analysis can reveal **which attributes most strongly influence car prices**, supporting both predictive and analytical use cases.

---

### Business & Practical Applications
This project illustrates how machine learning can support **real-world automotive market analysis**, including:

- **Dealership pricing strategies** – quickly estimating competitive vehicle prices.  
- **Consumer decision support** – helping buyers and sellers understand realistic market values.  
- **Market analytics** – identifying trends in vehicle depreciation and pricing factors.

The web application provides an **interactive interface** that allows users to estimate car prices and explore the factors influencing predictions, making advanced machine learning insights accessible to non-technical users.
""")

if st.session_state.page == "home": home()
elif st.session_state.page == "predict": predict_page()
elif st.session_state.page == "info": info()