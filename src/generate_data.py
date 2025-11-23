import numpy as np
import pandas as pd

np.random.seed(42)   # for reproducibility

# Number of samples
n_samples = 3000

# Generate normal sensor readings
rpm_normal = np.random.normal(2000, 300, n_samples)
speed_normal = np.random.normal(60, 10, n_samples)
coolant_normal = np.random.normal(90, 5, n_samples)
engine_load_normal = np.random.normal(40, 10, n_samples)
vibration_normal = np.random.normal(0.02, 0.01, n_samples)
battery_voltage_normal = np.random.normal(12.5, 0.3, n_samples)

# Generate anomalies
n_anomaly = 300
rpm_anomaly = np.random.normal(4000, 500, n_anomaly)
speed_anomaly = np.random.normal(120, 20, n_anomaly)
coolant_anomaly = np.random.normal(120, 10, n_anomaly)
engine_load_anomaly = np.random.normal(90, 5, n_anomaly)
vibration_anomaly = np.random.normal(0.2, 0.05, n_anomaly)
battery_voltage_anomaly = np.random.normal(9, 0.5, n_anomaly)

# Combine
data = {
    "rpm": np.concatenate([rpm_normal, rpm_anomaly]),
    "speed": np.concatenate([speed_normal, speed_anomaly]),
    "coolant_temp": np.concatenate([coolant_normal, coolant_anomaly]),
    "engine_load": np.concatenate([engine_load_normal, engine_load_anomaly]),
    "vibration": np.concatenate([vibration_normal, vibration_anomaly]),
    "battery_voltage": np.concatenate([battery_voltage_normal, battery_voltage_anomaly]),
    "label": [0]*n_samples + [1]*n_anomaly
}

df = pd.DataFrame(data)
df.to_csv("vehicle_sensor_data.csv", index=False)

df.head()
