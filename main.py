from fastapi import FastAPI
import pickle
import pandas as pd
from src.data_model import Water

# Initialize FastAPI app
app = FastAPI(
    title="Water Potability Prediction",
    description="Predicting Water Potability"
)

# Load the trained model
with open("/Users/furkanozer/ml_pipeline/src/model.pkl", "rb") as f:
    model = pickle.load(f)

# Root endpoint
@app.get("/")
def index():
    return "Welcome to Water Potability Prediction FastAPI"

@app.post("/predict")
def model_predict(water: Water):
    # Convert the input Water object to a pandas DataFrame
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })

    # Use the trained model to make predictions
    predicted_value = model.predict(sample)

    # Return the prediction result
    if predicted_value == 1:
        return {"message": "Water is Consumable"}
    else:
        return {"message": "Water is not Consumable"}
  