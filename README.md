🚗 Vehicle Sensor Anomaly Detection (AI/ML + FastAPI + Docker)

A complete end-to-end Machine Learning workflow for detecting abnormal vehicle sensor behavior.
Designed to simulate automotive predictive maintenance, matching real ARAI-style AI/ML projects.

This system uses:

Python (ML Model)

Scikit-learn

FastAPI for inference

Docker for deployment

WSL2 backend (for Windows users)

✨ Features

✔ Synthetic automotive sensor dataset
✔ Data preprocessing (cleaning, outliers, scaling)
✔ ML model (Random Forest)
✔ Model evaluation (Accuracy, Recall, F1)
✔ Inference pipeline
✔ FastAPI microservice
✔ Docker deployment-ready
✔ Realistic automotive use-case: anomaly detection

📁 Project Structure
AI-ML-Vehicle-Anomaly-Detection/
│
├── data/
│   └── vehicle_sensor_data.csv
│
├── models/
│   ├── anomaly_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── inference.py
│   └── api.py
│
├── Dockerfile
├── requirements.txt
└── README.md

⚙️ Setup Instructions
1️⃣ Create Virtual Environment
python -m venv venv


Activate (Windows):

venv\Scripts\activate

2️⃣ Install Required Libraries
pip install -r requirements.txt

3️⃣ Generate Dataset
python src/generate_data.py


Dataset saved to:

data/vehicle_sensor_data.csv

4️⃣ Preprocess the Data
python src/preprocess.py

5️⃣ Train the ML Model
python src/train_model.py


Saves:

models/anomaly_model.pkl
models/scaler.pkl

6️⃣ Run Inference (Without Docker)
python src/inference.py

🧪 API (FastAPI)
Start API (inside Docker):
docker run -p 8000:8000 vehicle-ml-api

Root endpoint:
GET http://localhost:8000/


Response:

{"message": "Vehicle Anomaly Detection API Running"}

Prediction endpoint:
GET http://localhost:8000/predict?rpm=4500&speed=120&coolant_temp=118&engine_load=85&vibration=0.15&battery_voltage=9.1


Response:

{
  "status": "ANOMALY DETECTED — Possible fault in engine/sensors"
}

📊 Model Performance

Accuracy: ~95%

Precision: ~90%

Recall: ~96% (very important for safety)

F1 Score: ~93%

High recall ensures fewer missed faults, which is critical in automotive safety systems.

🐳 Docker Deployment
Build Docker Image:
docker build -t vehicle-ml-api .

Run Container:
docker run -p 8000:8000 vehicle-ml-api


API available at:

http://localhost:8000

🚘 Automotive Context (Interview Explanation)

This project simulates real vehicle telemetry:

RPM

Speed

Coolant temperature

Engine load

Vibration

Battery voltage

ML model classifies:

0 → Normal operation

1 → Anomaly detected (possible engine/sensor fault)

This is similar to predictive maintenance workflows used in automotive R&D labs like ARAI.

🧠 Interview-Ready Explanation

“I built a complete AI/ML workflow for automotive anomaly detection.
It includes data preprocessing, model training, validation, inference pipeline, and Dockerized deployment.
The microservice uses FastAPI and can be deployed on Linux or cloud/on-prem servers, matching ARAI’s ML workflow architecture.”

🛠️ Technologies Used

Python

NumPy, Pandas

Scikit-Learn

FastAPI

Uvicorn

Docker

WSL2

Joblib

✅ Status

✔ Fully working
✔ Docker-ready
✔ Interview-ready
✔ Deployable ML microservice

🖼️ Screenshots (Execution Proof)
1. Preprocessing Output

This shows dataset loading, cleaning, splitting, and preprocessing.

![Preprocess Output](/mnt/data/Screenshot 2025-11-23 000103.png)

2. Model Training Output

This shows model metrics (Accuracy, Recall, F1 Score).

![Training Output](/mnt/data/Screenshot 2025-11-23 000540.png)

3. Docker Build + API Startup

This confirms that the FastAPI service is running successfully inside Docker.

![Docker Run API](/mnt/data/Screenshot 2025-11-23 114715.png)

4. Root Endpoint Running (localhost:8000)

This shows that the API is running and accessible.

![API Root](/mnt/data/Screenshot 2025-11-23 114820.png)

5. Prediction Endpoint Working

This shows a real anomaly detection output via API.

![Prediction Output](/mnt/data/Screenshot 2025-11-23 114919.png)