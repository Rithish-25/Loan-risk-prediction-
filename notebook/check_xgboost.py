import joblib
import pandas as pd

# Load XGBoost model
model = joblib.load("loan_risk_model_xgboost.pkl")

# Sample input (5 features only)
sample_data = pd.DataFrame([{
    "marital_status": "Single",
    "number_of_dependents": 2,
    "vehicle_ownership": "Yes",
    "bank_account_age": 5,
    "savings_balance": 150000
}])

# Make predictions (bypassing the scikit-learn 1.6 compatibility issue)
X_transformed = model.named_steps["preprocessor"].transform(sample_data)
prediction = model.named_steps["classifier"].predict(X_transformed)
probability = model.named_steps["classifier"].predict_proba(X_transformed)

print("Prediction (0 = Safe, 1 = Default Risk):", prediction[0])
print(f"Risk Probability: {probability[0][1]:.2%}")