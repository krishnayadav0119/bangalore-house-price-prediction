# Bangalore House Price Prediction

Regression project (Major Project, DSBA September 2026 Batch): predict house prices in Bengaluru from area, location and number of rooms.

## Problem
Given a property's size, bedrooms (BHK), bathrooms, balconies, location, area type and availability, predict its listing price in **lakhs of rupees**.

## Dataset
- Bangalore House Price Prediction Dataset (Kaggle, by Amey Thakur), file `Bengaluru_House_Data.csv`
- 13,320 rows, 9 columns: `area_type, availability, location, size, society, total_sqft, bath, balcony, price`
- Data is about 5 years old and contains listing prices, not final sale prices.

## Approach
1. **Explore:** shape, types, missing values (society 41%, balcony 4.6%), 529 duplicates, 1,305 locations.
2. **Clean:** removed duplicates, dropped `society`, dropped 17 rows with missing location/size, filled `bath` and `balcony` with medians, built numeric `bhk`, converted `total_sqft` (ranges averaged), dropped 46 unusable rows. Result: 12,728 rows.
3. **Outliers:** area per bedroom below 300 sqft, bathrooms above BHK + 2, price-per-sqft outliers within each location, BHK price-order outliers, homes above 10,000 sqft, homes above 10 BHK. Result: 7,960 rows.
4. **Feature engineering:** `price_per_sqft` (EDA and outlier detection only, never a model input, to avoid data leakage), locations with 10 or fewer listings grouped as "other" (1,193 to 170), `availability` simplified to two values, `size` dropped.
5. **EDA:** 14 plots plus a bonus, each with an interpretation.
6. **Modelling:** 80/20 split (`random_state=42`) before preprocessing, scikit-learn pipeline (StandardScaler + OneHotEncoder), 7 models, 5-fold CV, a log-price experiment, and RandomizedSearchCV tuning.
7. **Interpret:** actual vs predicted, residuals, permutation importance, business insights, prediction function.

## Results (test set; RMSE and MAE in lakhs)

| Model | Train R2 | Test R2 | RMSE | MAE | CV R2 |
|---|---|---|---|---|---|
| **Random Forest (tuned)** | 0.856 | **0.770** | **47.58** | 22.13 | **0.694** |
| Random Forest | 0.938 | 0.734 | 51.18 | 21.80 | 0.672 |
| Gradient Boosting | 0.812 | 0.734 | 51.15 | 23.85 | 0.671 |
| Ridge | 0.659 | 0.690 | 55.22 | 27.57 | 0.658 |
| Linear Regression | 0.659 | 0.689 | 55.29 | 27.66 | 0.658 |
| Lasso | 0.650 | 0.684 | 55.69 | 28.18 | 0.651 |
| Decision Tree | 0.983 | 0.387 | 77.62 | 28.00 | 0.542 |

Best settings: `n_estimators=200, max_depth=15, min_samples_split=2, min_samples_leaf=2, max_features=0.5`.

## Key findings
- Top 5 features: `total_sqft` (by far the strongest), `location`, `area_type`, `bath`, `bhk`.
- The same 1,200 sqft 2 BHK home is predicted at about 120.5 lakhs in Rajaji Nagar vs 63.2 lakhs in Electronic City.
- The Decision Tree overfits heavily (train 0.983, test 0.387).
- Log-transforming price did not improve results, so raw price is used.

## Limitations
- Old data, listing prices only, no floor/building age/amenities.
- About 37% of rows removed as outliers; best for typical homes.
- Luxury homes (above about 500 lakhs) are under-predicted.

## How to run
1. Open `notebook/Bangalore_House_Price_Prediction.ipynb` in Google Colab.
2. Upload `Bengaluru_House_Data.csv` when asked (or place it in `data/`).
3. Run **Runtime > Run all** (about 10 minutes, mostly the tuning step).

### Run the web app
```bash
pip install streamlit pandas scikit-learn joblib
streamlit run app.py
```
`app.py` needs `house_price_model.pkl` and `locations.json` in the same folder (see the app file for how to create them).

## Project structure
```
data/        Bengaluru_House_Data.csv
notebook/    Bangalore_House_Price_Prediction.ipynb
report/      Project report (PDF or Word)
app.py       Streamlit demo
house_price_model.pkl
locations.json
README.md
```

## Tech
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, joblib, Streamlit.
