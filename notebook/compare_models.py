import joblib
import pandas as pd

# Load models
rf_model = joblib.load("loan_risk_model_random_forest.pkl")
xgb_model = joblib.load("loan_risk_model_xgboost.pkl")

# Random Forest input
rf_input = pd.DataFrame([{
    "person_age":22,
    "person_gender":"female",
    "person_education":"Master",
    "person_income":71948,
    "person_emp_exp":0,
    "person_home_ownership":"RENT",
    "loan_amnt":35000,
    "loan_intent":"PERSONAL",
    "loan_int_rate":16.02,
    "loan_percent_income":0.49,
    "cb_person_cred_hist_length":3,
    "credit_score":561,
    "previous_loan_defaults_on_file":"No"
}])

# XGBoost input
xgb_input = pd.DataFrame([{
    "marital_status":"Single",
    "number_of_dependents":2,
    "vehicle_ownership":"Yes",
    "bank_account_age":5,
    "savings_balance":150000
}])

# Make predictions (bypassing the scikit-learn 1.6 compatibility issue)
rf_X_transformed = rf_model.named_steps["preprocessor"].transform(rf_input)
rf_prediction = rf_model.named_steps["classifier"].predict(rf_X_transformed)

xgb_X_transformed = xgb_model.named_steps["preprocessor"].transform(xgb_input)
xgb_prediction = xgb_model.named_steps["classifier"].predict(xgb_X_transformed)

print("Random Forest Prediction :", rf_prediction[0])
print("XGBoost Prediction       :", xgb_prediction[0])