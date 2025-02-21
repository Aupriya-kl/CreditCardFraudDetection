import pickle
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Load the classifier
with open("CreditCardClassifier.pkl", "rb") as file:
    CreditCardClassifier = pickle.load(file)

# Define the data model
class Credit(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Initialize the app
app = FastAPI()

@app.get("/")
def root():
    return {"Credit card detection"}

@app.post("/predict")
def predict(data: Credit):
    data = data.dict()
    # Convert dictionary values to a NumPy array
    features = np.array([[
        data['Time'], data['V1'], data['V2'], data['V3'], data['V4'], data['V5'], data['V6'], 
        data['V7'], data['V8'], data['V9'], data['V10'], data['V11'], data['V12'], data['V13'], 
        data['V14'], data['V15'], data['V16'], data['V17'], data['V18'], data['V19'], data['V20'], 
        data['V21'], data['V22'], data['V23'], data['V24'], data['V25'], data['V26'], data['V27'], 
        data['V28'], data['Amount']
    ]])

    prediction = CreditCardClassifier.predict(features)
    
    if prediction[0] == 1:
        prediction_text = "not valid"
    else:
        prediction_text = "valid"
    
    return {
        "prediction": prediction_text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
