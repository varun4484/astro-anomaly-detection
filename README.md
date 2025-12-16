# 🌌 A Deep Learning Approach for Detecting Unusual Celestial Phenomena in Astronomical Datasets

---

## ✨ Overview

This project develops a **deep learning pipeline** to detect **anomalies in astronomical images**, specifically distinguishing between celestial objects labeled as:

- 🔴 **E** — Anomalies  
- 🟢 **SB** — Non-anomalies  

The pipeline uses a **ResNet50 model**, trained on a large dataset of preprocessed astronomical images, and exposes a **FastAPI-based REST API** for real-time anomaly detection. Prediction results are stored in a **PostgreSQL database**.

---

## 📂 Dataset

### 📌 Source
The dataset consists of **preprocessed astronomical images** in `.jpg` format, resized to **227 × 227 pixels**.

### 🗂 Structure

**Training Set**  
`data/images_E_S_SB_227x227_a_03_train_processed/`  
- `E/` — Anomaly class  
- `SB/` — Non-anomaly class  
- 📊 **Total images:** 90,524  

**Test Set**  
`data/images_E_S_SB_227x227_a_03_test_processed/`  
- `E/` — Anomaly class  
- `SB/` — Non-anomaly class  
- 📊 **Total images:** 10,057  

---

## ⚙️ Requirements

- 🐍 Python 3.12  
- 🗄 PostgreSQL (for storing predictions)  
- 📦 Dependencies listed in `requirements.txt`

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd anomaly-detection-astro
python -m venv venv
.\venv\Scripts\activate.bat   # On Windows
pip install -r requirements.txt
psql -U postgres -c "CREATE DATABASE anomaly_detection;"
```
python setup_db.py
psycopg2.connect(
    dbname="anomaly_detection",
    user="postgres",
    password="yourpassword",  # Replace with your password
    host="localhost"
)
```
python train_model.py
models/resnet50_model.h5
uvicorn main:app --reload
```
http://127.0.0.1:8000
```
curl.exe -X POST -F "file=@C:/path/to/image.jpg" http://127.0.0.1:8000/detect
Example Response
{
  "result": "Anomaly Detected",
  "score": 1.0
}
psql -U postgres -d anomaly_detection
SELECT * FROM predictions;
| id | image_name | score | result           | timestamp  |
| -- | ---------- | ----- | ---------------- | ---------- |
| 1  | 311.jpg    | 1.0   | Anomaly Detected | 2025-06-08 |
```
📁 Project Files

train_model.py — Trains the ResNet50 model

main.py — FastAPI application

database.py — PostgreSQL database interactions

setup_db.py — Database schema setup

requirements.txt — Project dependencies

models/ — Stores trained model (resnet50_model.h5)

data/ — Preprocessed image datasets

📝 Notes

Classification threshold: 0.5

Class mapping:

E → Anomaly (1)

SB → Non-Anomaly (0)

Training on CPU may be time-consuming

🛠 Troubleshooting
⏳ Training Takes Too Long

Reduce epochs (e.g., epochs=1)

Reduce batch size (e.g., batch_size=16)

❌ API Errors

Check Uvicorn logs

Ensure all dependencies are installed correctly

🗄 Database Connection Issues

Verify PostgreSQL is running

Confirm credentials in database.py
📊 Results

🎯 Validation Accuracy: ~86%

🔁 Epochs: 5

🧠 Model: ResNet50

Example Prediction:

Image: 311.jpg (E class)

Result: Anomaly Detected

Score: 1.0

🚀 Future Improvements

⚡ GPU acceleration

🔧 Fine-tune ResNet50

🎨 Data augmentation

📦 Batch inference API endpoints



