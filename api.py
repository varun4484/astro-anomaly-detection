from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import os
from preprocess import preprocess_image, convert_fits_to_png
from database import save_prediction
from werkzeug.utils import secure_filename

app = FastAPI(title="Astronomical Anomaly Detection API")
try:
    model = tf.keras.models.load_model("models/best_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/detect")
async def detect_anomaly(file: UploadFile = File(...)):
    """Endpoint to detect anomalies in uploaded astronomical images."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.fits')):
            raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, or FITS files are allowed")
        
        # Save uploaded file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Handle FITS files
        if filename.lower().endswith('.fits'):
            temp_png = os.path.join(UPLOAD_FOLDER, filename.replace('.fits', '.png'))
            convert_fits_to_png(UPLOAD_FOLDER, UPLOAD_FOLDER)
            file_path = temp_png if os.path.exists(temp_png) else file_path
        
        # Preprocess and predict
        image = preprocess_image(file_path)
        prediction = model.predict(image)[0][0]
        result = "Anomaly Detected" if prediction > 0.5 else "No Anomaly"
        
        # Save to database
        save_prediction(filename, float(prediction), result)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        if filename.lower().endswith('.fits') and os.path.exists(temp_png):
            os.remove(temp_png)
        
        return JSONResponse(content={
            "filename": filename,
            "prediction_score": float(prediction),
            "result": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")