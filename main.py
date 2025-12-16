import json
import logging
import os
import hashlib
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Ensure uploads directory exists
UPLOAD_DIR = "frontend/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    logger.info(f"Created uploads directory: {UPLOAD_DIR}")

# Load ResNet50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Modify the final layer for binary classification (Anomaly vs Normal)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Anomaly Detected (1), Normal (0)
model = model.to(device)
model.eval()
logger.info("Loaded ResNet50 model")

# Define image transformations for prediction and training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load predictions at startup
predictions = []
PREDICTIONS_FILE = "frontend/predictions.json"
try:
    with open(PREDICTIONS_FILE, "r") as f:
        predictions = json.load(f)
    logger.info(f"Loaded {len(predictions)} predictions from {PREDICTIONS_FILE}")
except FileNotFoundError:
    logger.warning(f"{PREDICTIONS_FILE} not found, starting with empty list")
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f)
except Exception as e:
    logger.error(f"Error loading {PREDICTIONS_FILE}: {str(e)}")

# Function to preprocess image for prediction (used in /detect endpoint)
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Function to train the model at startup
def train_model_at_startup():
    logger.info("Starting model training at startup...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Load data from predictions.json
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {PREDICTIONS_FILE} for training: {str(e)}")
        return

    if not data:
        logger.info("No data available for training.")
        return

    # Prepare training data
    images = []
    labels = []
    updated_predictions = []
    for entry in data:
        image_path = entry.get("image_path")
        result = entry.get("result")
        trained = entry.get("trained", False)  # Default to False if not present
        if not image_path or not result:
            updated_predictions.append(entry)
            continue

        # Skip if already trained
        if trained:
            updated_predictions.append(entry)
            continue

        # Map result to label (Anomaly Detected: 1, Normal: 0)
        label = 1 if result == "Anomaly Detected" else 0

        # Load image
        try:
            # Adjust path: image_path is "/static/uploads/filename", we need "frontend/uploads/filename"
            filename = image_path.split("/")[-1]
            full_path = os.path.join(UPLOAD_DIR, filename)
            image = Image.open(full_path).convert("RGB")
            image_tensor = transform(image).to(device)  # Remove .unsqueeze(0)
            images.append(image_tensor)
            labels.append(label)
            # Mark as trained
            entry["trained"] = True
            updated_predictions.append(entry)
        except Exception as e:
            logger.error(f"Failed to load image {image_path} for training: {str(e)}")
            updated_predictions.append(entry)
            continue

    if not images:
        logger.info("No new images available for training.")
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(updated_predictions, f)
        return

    # Update predictions.json with trained flags
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(updated_predictions, f)
    logger.info(f"Updated {PREDICTIONS_FILE} with trained flags")

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    # Stack images into a batch
    images = torch.stack(images, dim=0).to(device)

    # Fine-tune the model
    epochs = 1  # Small number of epochs for quick fine-tuning
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    model.eval()
    logger.info("Model training at startup completed.")

# Train the model before the API starts
train_model_at_startup()

# Serve index.html at the root URL
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("frontend/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to serve index page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve index page: {str(e)}")

# Serve about.html at the /about URL
@app.get("/about", response_class=HTMLResponse)
async def serve_about():
    try:
        with open("frontend/about.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to serve about page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve about page: {str(e)}")

# Serve history.html at the /history URL
@app.get("/history", response_class=HTMLResponse)
async def serve_history():
    try:
        with open("frontend/history.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to serve history page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve history page: {str(e)}")

# Serve login.html at the /login URL
@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    try:
        with open("frontend/login.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to serve login page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve login page: {str(e)}")

# Serve history data for the History page
@app.get("/history_data")
async def get_history_data():
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)
        return predictions
    except FileNotFoundError:
        logger.warning(f"{PREDICTIONS_FILE} not found, returning empty list")
        return []
    except Exception as e:
        logger.error(f"Failed to load history data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load history data: {str(e)}")

# Clear history endpoint
@app.post("/clear_history")
async def clear_history():
    try:
        # Clear uploads folder
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {str(e)}")
            logger.info(f"Cleared all files in {UPLOAD_DIR}")
        else:
            logger.warning(f"Uploads directory {UPLOAD_DIR} does not exist")

        # Clear predictions.json
        global predictions
        predictions = []
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"Cleared {PREDICTIONS_FILE}")

        # Optional: Reset the model to pre-trained state
        # Comment out the following block if you don't want to reset the model
        # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 2)
        # model = model.to(device)
        # model.eval()
        # logger.info("Reset model to pre-trained state")

        return {"message": "History cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

# Compute SHA-256 hash of an image
def compute_image_hash(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()

# Handle image upload and prediction
@app.post("/detect")
async def detect_anomaly(file: UploadFile = File(...)):
    try:
        # Read image contents
        contents = await file.read()

        # Compute hash to check for duplicates
        image_hash = compute_image_hash(contents)

        # Save the image (even if it's a duplicate, we'll use a new filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        image_path = f"/static/uploads/{filename}"
        full_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(full_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved image to {full_path}")

        # Check if image hash exists in predictions to reuse prediction
        existing_prediction = None
        for prediction in predictions:
            if prediction.get("hash") == image_hash:
                existing_prediction = prediction
                break

        if existing_prediction:
            # Duplicate found, reuse prediction but create a new entry with new timestamp
            logger.info(f"Duplicate image detected (hash: {image_hash}), reusing prediction but logging new entry.")
            score = existing_prediction["score"]
            result = existing_prediction["result"]
        else:
            # Preprocess image and make prediction
            image_tensor = preprocess_image(contents)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1).item()
                score = probabilities[0][predicted].item()
                result = "Anomaly Detected" if predicted == 1 else "Normal"

        # Create new prediction entry (always, even for duplicates)
        prediction = {
            "image_path": image_path,
            "result": result,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "hash": image_hash,
            "trained": False  # Mark as trained (will be trained on next startup)
        }

        # Add to predictions list and save
        predictions.append(prediction)
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"Saved prediction to {PREDICTIONS_FILE}")

        return prediction
    except Exception as e:
        logger.error(f"Failed to process detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process detection: {str(e)}")