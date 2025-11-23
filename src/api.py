from fastapi import FastAPI
from src.inference import predict_anomaly


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Vehicle Anomaly Detection API Running"}

@app.get("/predict")
def predict(rpm: float, speed: float, coolant_temp: float, engine_load: float, vibration: float, battery_voltage: float):
    result = predict_anomaly(rpm, speed, coolant_temp, engine_load, vibration, battery_voltage)
    return {"status": result}
