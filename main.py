from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd


app = FastAPI(title='El-Matador Shadow Deployment')

model = joblib.load('xgb_ufc_model.pkl')
features = joblib.load('model_features.pkl')

class FightStats(BaseModel):
    stats:dict

@app.post('/predict')
def predict_fight(data: FightStats):

    input_df = pd.DataFrame([data.stats])
    input_df = input_df.reindex(columns=features , fill_value= 0)

    prediction = model.predict_proba(input_df)[0]

    return{
        'status':'winner',
        'blue_corner_win_probability':float(round(prediction[0]*100,2)),
        'red_corner_win_probability':float(round(prediction[1]*100,2))
    }

