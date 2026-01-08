# 🚗 Car Price Prediction in Poland

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-red?style=for-the-badge&logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)

## 📌 Project Overview

The objective of this project is to develop a robust Machine Learning model capable of forecasting used car prices in the Polish market. By analyzing features such as vehicle age, engine power, equipment specifications, and brand segmentation, the system provides accurate price estimations to aid buyers and sellers.

The project evolved through four distinct modeling stages: **Linear Regression**, **Random Forest**, **Base XGBoost**, and a **Final Tuned XGBoost** (optimized with Optuna and refined data scope).

After extensive data preprocessing, Exploratory Data Analysis (EDA), and feature engineering, the final model achieved production-grade performance, significantly outperforming baseline approaches.

---

## 🚀 Live Demo & Models

### 🖥️ Streamlit Dashboard
Explore the interactive application to predict car prices in real-time:
**[Launch App](YOUR_STREAMLIT_LINK_HERE)**

### 🤗 Hugging Face Model Registry
Due to file size constraints, the trained models are hosted on the Hugging Face Hub:
**[View Models on Hugging Face](https://huggingface.co/Przemsonn/poland-car-price-model)**

---

## 📊 Model Performance & Evolution

A strategic decision was made to refine the scope of the final model by filtering out vintage cars (pre-1980) and ultra-luxury supercars (top 1% price quantile). This shifted the focus to the **mass consumer market**, resulting in drastically improved accuracy.

| Model Version | Description | R² Score | RMSE (PLN) | MAE (PLN) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | Linear Regression (Baseline) | 89.78% | 26,547 | 9,597 | 19.30% |
| **Model 2** | Random Forest | 92.14% | 23,273 | 8,333 | 18.95% |
| **Model 3** | XGBoost (Base) | 92.24% | 23,124 | 7,655 | 15.90% |
| **Model 4** | **XGBoost (Tuned & Cleaned)** | **94.32%** | **14,150** | **6,838** | **15.84%** |

### Key Improvements
The final model demonstrates a **massive leap in reliability**:
* **RMSE Reduction:** The error margin dropped by over **12,000 PLN** compared to the baseline and **9,000 PLN** compared to the base XGBoost.
* **R² Increase:** An improvement of nearly **5 percentage points**, explaining over 94% of the price variance.

---

## 💡 Business Conclusions

From a business perspective, the transition from Model 1 to Model 4 represents a significant value add:

1.  **Market Focus Strategy:** By removing outliers (vintage/supercars), the model became specialized for 99% of the market. While it loses the ability to value a 1960 Ferrari, it becomes exceptionally good at valuing the Volkswagen Golfs and Toyota Corollas that make up the bulk of transactions.
2.  **Risk Reduction:** Reducing the RMSE from ~26k PLN to ~14k PLN means the "pricing risk" is cut nearly in half. For a car dealership, this translates to buying inventory at the right price and avoiding overpaying.
3.  **Profit Margin Protection:** With a MAPE (Mean Absolute Percentage Error) of ~15%, the model provides a tight pricing corridor. This allows sellers to set competitive prices without eroding their profit margins due to estimation errors.

---

## ⚙️ Technical Architecture

### 1. Advanced Feature Engineering
Domain knowledge was applied to extract high-value signals from raw data:
* **`Power_per_liter`**: A ratio indicating engine efficiency and performance density.
* **`Vehicle_age`**: Calculated dynamically relative to the advertisement publication date.
* **`Is_premium`**: A categorical flag for luxury brands based on market segmentation.
* **`Annual_mileage`**: Derived to detect excessive wear (high mileage) or "garage queens" (suspiciously low mileage).

### 2. External API Integration
The system integrates with the **National Bank of Poland (NBP) REST API** to fetch real-time EUR/PLN exchange rates. This ensures that dataset entries listed in EUR are normalized to PLN based on accurate market data, preventing currency fluctuation noise.

### 3. Hyperparameter Optimization (Optuna)
Instead of a brute-force GridSearch, the project utilizes **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler. This allowed for efficient navigation of the XGBoost hyperparameter space (learning rate, depth, regularization), converging on optimal settings with fewer computational resources.

---

## 🛠 Tech Stack

* **Language:** Python 3.9+
* **Core Libraries:** Pandas, NumPy, Scikit-Learn
* **Modeling:** XGBoost (Gradient Boosting)
* **Optimization:** Optuna
* **Visualization:** Matplotlib, Seaborn
* **Utilities:** Joblib (Serialization), Requests (API)

---

## 📥 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Car-Price-Prediction.git](https://github.com/YourUsername/Car-Price-Prediction.git)
    cd Car-Price-Prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the main script:**
    ```bash
    python main.py
    ```

4.  **Launch the Streamlit App:**
    ```bash
    streamlit run app.py
    ```