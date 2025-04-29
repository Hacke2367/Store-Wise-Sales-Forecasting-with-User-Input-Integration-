from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np


class SalesInput(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    Month: int
    Year: int
    Holiday__0: int
    Holiday__1: int
    Holiday__2: int
    Holiday__3: int
    Dayofmonth: int
    IsWeekend: int
    Sales_Lag7: float
    Sales_RollingMean7: float


app = FastAPI()
model = joblib.load('STORE-WISE.pkl')

REQUIRED_FEATURES = [
    'Store',  # ✅ स्टोर ID फीचर जोड़ें
    'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Year',
    'Holiday__0', 'Holiday__1', 'Holiday__2', 'Holiday__3',
    'Dayofmonth', 'IsWeekend', 'Sales_Lag7', 'Sales_RollingMean7'
]


@app.post("/predict")
async def predict(input_data: SalesInput):
    try:
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])[REQUIRED_FEATURES]

        prediction = model.predict(input_df)
        prediction_actual = np.expm1(prediction)

        return {"prediction": prediction_actual.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

















