# Car Price Prediction Model - AI Coding Guidelines

## Project Overview
This is a machine learning project predicting used car prices in the Polish market using a Jupyter notebook (`notebooks/cars_price_prediction.ipynb`). The pipeline progresses through data preprocessing, exploratory data analysis (EDA), feature engineering, and dual model comparison (Ridge regression and Random Forest).

## Data Pipeline Architecture

### Data Flow
1. **Raw Data**: `data/Car_sale_ads.csv` - Polish car listings with ~30 features
2. **Preprocessing Phase**: Handle missing values, outliers, currency conversion (EUR→PLN at 4.3 rate)
3. **Feature Engineering**: Create derived features (vehicle age, annual mileage, feature counts, premium brand flags)
4. **Model Training**: Split into train/test (70/30), encode categorical variables, fit regression models

### Key Data Characteristics
- **Target**: `price_PLN` (prices in Polish Zloty, converted from EUR when needed)
- **Outlier Handling**: Strict filtering removes extreme values:
  - Price: 2,500 PLN - 3,000,000 PLN
  - Mileage: < 800,000 km
  - Power: 20-800 HP (excludes supercars/anomalies)
  - Displacement: 200-7,000 cm³
- **Categorical Encoding**: OneHotEncoder with `handle_unknown='ignore'` preserves unseen categories in test data

## Critical Developer Workflows

### Running the Notebook
1. Ensure dataset exists at `data/Car_sale_ads.csv`
2. All cells must execute sequentially (dependencies between preprocessing and modeling cells)
3. Generated visualizations save to `images/` directory as PNG files
4. Use log transformation (`np.log1p(y)` / `np.expm1()`) for predictions to normalize price distribution

### Feature Engineering Patterns
Key engineered features follow these conventions:
- **Numeric derived**: `Vehicle_age = Year_publication - Production_year`
- **Rate-based**: `Annual_mileage = Mileage_km / (Vehicle_age + 1e-6)` (epsilon prevents division by zero)
- **Binary flags**: `Is_premium_car`, `Is_popular_color` (boolean 0/1)
- **Categoricals**: `Age_category` maps continuous values to discrete bins (New/Young/Middle_Aged/Old)
- **Feature counts**: `Num_features = len(Features.split(','))` from list column

## Project-Specific Patterns & Conventions

### Preprocessing Steps (Non-Negotiable Order)
1. Drop high-NaN columns: `CO2_emissions`, `First_registration_date`, `Vehicle_version`, `Vehicle_generation`
2. Fill categorical NaN: `First_owner` and `Origin_country` → 'Unknown'
3. Fill numeric NaN with **model-grouped medians** (e.g., `Displacement_cm3` median per `Vehicle_model`)
4. Remove rows with NaN in `Transmission` (sparse but critical feature)
5. Drop features with weak price correlation: `Brand_popularity` (0.0007), `Doors_number` (-0.038)
6. Always drop original `Price` and `Currency` columns (replaced by `price_PLN`)

### Model Pipelines
All models use `sklearn.pipeline.Pipeline` with two preprocessors:
- **Regression (Ridge)**: `ColumnTransformer` with `StandardScaler` for numerics + `OneHotEncoder` for categoricals
- **Tree-based (Random Forest)**: Same transformer but numerics use `'passthrough'` (trees don't benefit from scaling)

Hyperparameter tuning uses **log-transformed targets** (`y_train_log = np.log1p(y_train)`) to normalize heteroscedastic residuals.

### Visualization Conventions
- All plots save to `images/` with descriptive filenames using underscores
- Use `plt.tight_layout()` and `dpi=300` for reproducibility
- Add grid with `alpha=0.3` for readability on busy plots
- Format numeric axes: `yaxis.set_major_formatter()` to display large numbers with spaces (e.g., "100 000" not "100000")

## Cross-Component Dependencies

### Feature Interactions
The model relies on these feature relationships:
- **Transmission ↔ Price**: Automatic cars average 120k PLN vs 40k PLN for manual (3x difference)
- **Brand ↔ Price**: German premium brands (Mercedes/BMW/Audi) command ~100k PLN vs 35-50k PLN for others
- **Vehicle_type ↔ Price**: Coupes/SUVs (~110-140k) vs city cars (~25k) - 5x variance
- **Condition ↔ Price**: New cars 160k PLN vs used 50k PLN (strongest single predictor)
- **Equipment (Num_features) ↔ Price**: Strong positive correlation—vehicles with 60+ features reach 200k+ PLN

### Missing Data Strategy
- **Mileage, Power, Displacement**: Fill with feature-grouped medians to preserve model-specific patterns
- **Drive**: Fill grouped-by-model using mode() with fallback to 'Unknown'
- **Transmission**: Drop entire rows (too critical to impute)
- **Date fields**: Extract year component only; drop publication year if constant (checked in preprocessing)

## Integration Points & External Dependencies

### Libraries & Versions
- **Data processing**: pandas (groupby transforms, fillna patterns), numpy (log transforms)
- **ML Pipeline**: scikit-learn (Pipeline, ColumnTransformer, GridSearchCV/RandomizedSearchCV)
- **Visualization**: matplotlib (subplots with tight_layout), seaborn (categorical plots)
- **Stats**: scipy.stats (Q-Q plots for residual diagnostics)

### Grid/Randomized Search Configuration
- **Ridge**: GridSearchCV with 6 alpha values × 4 solvers × 2 intercept options = 48 combinations
- **RandomForest**: RandomizedSearchCV with 30 iterations (avoids 432 combinations from grid)
- **Cross-validation**: KFold(5 splits) for Ridge, StratifiedShuffleSplit(3 splits) for RF
- **Scoring**: Multi-metric with `refit='neg_mse'` to prioritize RMSE

## Common Pitfalls & Warnings

1. **Currency Mismatch**: EUR prices must be converted (4.3 rate) before modeling; verify all prices in `price_PLN` column
2. **Outlier Sensitivity**: Ridge regression shows heteroscedasticity (higher errors on expensive cars); log transformation partially mitigates
3. **Feature Leakage**: `Year_publication` is constant across dataset—must drop before training
4. **Categorical Unknowns**: Only OneHotEncoder with `handle_unknown='ignore'` prevents errors on unseen test categories
5. **Tree Model Scaling**: Do NOT scale numerics for Random Forest (wastes preprocessing)
6. **Epsilon in Division**: Always use `+ 1e-6` when dividing by vehicle age to avoid division by zero

## Documentation & Generated Artifacts
- **Diagnostics**: `Heatmap_for_numeric_data_type` shows feature correlations; use to identify collinear pairs
- **Residuals**: Check `Actual_vs_Predicted` and `Q-Q_Plot_of_Residuals` for heteroscedasticity and normality
- **Learning Curves**: Flatten around 100k samples (dataset size), indicating ceiling on current model complexity
- **Feature Importance** (Random Forest): Use to rank predictors; Power_HP and Production_year typically top 2

## Recommended Next Steps for Model Improvement
1. Implement XGBoost/LightGBM for better handling of non-linearity and luxury vehicle outliers
2. Add price quantile regression to capture uncertainty bounds by vehicle segment
3. Investigate premium segment separately (separate model for cars > 100k PLN) to reduce MAPE from 19.5%
4. Create categorical interaction features (e.g., Brand × Type) to capture luxury SUV premiums
