import joblib

model = joblib.load("xgb_ufc_model.pkl")

features = model.feature_names_in_
print(list(features))