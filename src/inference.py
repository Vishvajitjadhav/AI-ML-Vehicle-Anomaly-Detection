import numpy as np
import joblib

def load_artifacts():
    model = joblib.load("models/anomaly_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def predict_anomaly(rpm, speed, coolant_temp, engine_load, vibration, battery_voltage):
    model, scaler = load_artifacts()

    # Prepare input as 2D array
    input_data = np.array([[rpm, speed, coolant_temp, engine_load, vibration, battery_voltage]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        return "NORMAL — Vehicle is operating safely"
    else:
        return "ANOMALY DETECTED — Possible fault in engine/sensors"

if __name__ == "__main__":
    # Example sensor input (you can change these values)
    result = predict_anomaly(
        rpm=4200,
        speed=120,
        coolant_temp=118,
        engine_load=85,
        vibration=0.15,
        battery_voltage=9.1

        # NORMAL — Vehicle is operating safely
        # rpm=1800,
        # speed=70,
        # coolant_temp=90,
        # engine_load=40,
        # vibration=0.03,
        # battery_voltage=12.4

    )
    print(result)
