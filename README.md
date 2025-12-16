A Deep Learning Approach for Detecting Unusual Celestial Phenomena in Astronomical Datasets
Overview
This project develops a deep learning pipeline to detect anomalies in astronomical images, specifically distinguishing between celestial objects labeled as E (anomalies) and SB (non-anomalies). The pipeline uses a ResNet50 model, trained on a dataset of preprocessed images, and exposes an API for real-time anomaly detection.
Dataset

Source: The dataset consists of preprocessed astronomical images in .jpg format, resized to 227x227 pixels.
Structure:
Training set: data/images_E_S_SB_227x227_a_03_train_processed/ (90,524 images)
E/: Anomaly class
SB/: Non-anomaly class


Test set: data/images_E_S_SB_227x227_a_03_test_processed/ (10,057 images)
E/: Anomaly class
SB/: Non-anomaly class





Requirements

Python 3.12
PostgreSQL (for storing predictions)
Dependencies listed in requirements.txt

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd anomaly-detection-astro


Create and Activate a Virtual Environment:
python -m venv venv
.\venv\Scripts\activate.bat  # On Windows


Install Dependencies:
pip install -r requirements.txt


Set Up the Database:

Ensure PostgreSQL is installed and running.
Create a database named anomaly_detection:psql -U postgres -c "CREATE DATABASE anomaly_detection;"


Run setup_db.py to create the predictions table:python setup_db.py


Update database.py with your PostgreSQL password:self.conn = psycopg2.connect(
    dbname="anomaly_detection",
    user="postgres",
    password="yourpassword",  # Replace with your password
    host="localhost"
)




Train the Model:

Run the training script to train the ResNet50 model:python train_model.py


The trained model will be saved to models/resnet50_model.h5.



Usage

Start the API Server:

Run the FastAPI server using Uvicorn:uvicorn main:app --reload


The server will be available at http://127.0.0.1:8000.


Test the API:

Use curl to send an image for anomaly detection:curl.exe -X POST -F "file=@C:/path/to/image.jpg" http://127.0.0.1:8000/detect


Example response:{"result":"Anomaly Detected","score":1.0}




Check Predictions in the Database:

Connect to PostgreSQL:psql -U postgres -d anomaly_detection


Query the predictions:SELECT * FROM predictions;


Example output:id | image_name | score |      result      |         timestamp
----+------------+-------+------------------+----------------------------
1  | 311.jpg    |     1 | Anomaly Detected | 2025-06-08 15:16:18.835798





Files

train_model.py: Trains the ResNet50 model on the dataset.
main.py: FastAPI application for anomaly detection.
database.py: Manages PostgreSQL database interactions.
setup_db.py: Sets up the database schema.
requirements.txt: Lists project dependencies.
models/: Directory to store the trained model (resnet50_model.h5).
data/: Directory containing the preprocessed image datasets.

Notes

The model classifies images as "Anomaly Detected" if the prediction score is greater than 0.5.
Class E is treated as the anomaly class (label 1), and SB as the non-anomaly class (label 0).
Training may take significant time on a CPU (e.g., ~10 minutes for 5 epochs). To speed up, reduce the number of epochs or batch size in train_model.py.

Troubleshooting

Training Takes Too Long:
Reduce epochs in train_model.py (e.g., epochs=1).
Reduce batch size (e.g., batch_size=16).


API Errors:
Check Uvicorn logs for detailed error messages.
Ensure all dependencies are installed (pip install -r requirements.txt).


Database Connection Issues:
Verify PostgreSQL is running and the password in database.py is correct.



Results

The model was trained for 5 epochs, achieving a validation accuracy of approximately 86% (values may vary due to random initialization).
Example prediction:
Image: 311.jpg (from E class)
Result: Anomaly Detected
Score: 1.0



Future Improvements

Use a GPU to speed up training.
Fine-tune the ResNet50 model for better accuracy.
Add data augmentation to improve model robustness.
Implement additional API endpoints for batch processing.

A Deep Learning Approach for Detecting Unusual Celestial Phenomena in Astronomical Datasets
Overview
This project develops a deep learning pipeline to detect anomalies in astronomical images, specifically distinguishing between celestial objects labeled as E (anomalies) and SB (non-anomalies). The pipeline uses a ResNet50 model, trained on a dataset of preprocessed images, and exposes an API for real-time anomaly detection.
Dataset

Source: The dataset consists of preprocessed astronomical images in .jpg format, resized to 227x227 pixels.
Structure:
Training set: data/images_E_S_SB_227x227_a_03_train_processed/ (90,524 images)
E/: Anomaly class
SB/: Non-anomaly class


Test set: data/images_E_S_SB_227x227_a_03_test_processed/ (10,057 images)
E/: Anomaly class
SB/: Non-anomaly class





Requirements

Python 3.12
PostgreSQL (for storing predictions)
Dependencies listed in requirements.txt

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd anomaly-detection-astro


Create and Activate a Virtual Environment:
python -m venv venv
.\venv\Scripts\activate.bat  # On Windows


Install Dependencies:
pip install -r requirements.txt


Set Up the Database:

Ensure PostgreSQL is installed and running.
Create a database named anomaly_detection:psql -U postgres -c "CREATE DATABASE anomaly_detection;"


Run setup_db.py to create the predictions table:python setup_db.py


Update database.py with your PostgreSQL password:self.conn = psycopg2.connect(
    dbname="anomaly_detection",
    user="postgres",
    password="yourpassword",  # Replace with your password
    host="localhost"
)




Train the Model:

Run the training script to train the ResNet50 model:python train_model.py


The trained model will be saved to models/resnet50_model.h5.



Usage

Start the API Server:

Run the FastAPI server using Uvicorn:uvicorn main:app --reload


The server will be available at http://127.0.0.1:8000.


Test the API:

Use curl to send an image for anomaly detection:curl.exe -X POST -F "file=@C:/path/to/image.jpg" http://127.0.0.1:8000/detect


Example response:{"result":"Anomaly Detected","score":1.0}




Check Predictions in the Database:

Connect to PostgreSQL:psql -U postgres -d anomaly_detection


Query the predictions:SELECT * FROM predictions;


Example output:id | image_name | score |      result      |         timestamp
----+------------+-------+------------------+----------------------------
1  | 311.jpg    |     1 | Anomaly Detected | 2025-06-08 15:16:18.835798





Files

train_model.py: Trains the ResNet50 model on the dataset.
main.py: FastAPI application for anomaly detection.
database.py: Manages PostgreSQL database interactions.
setup_db.py: Sets up the database schema.
requirements.txt: Lists project dependencies.
models/: Directory to store the trained model (resnet50_model.h5).
data/: Directory containing the preprocessed image datasets.

Notes

The model classifies images as "Anomaly Detected" if the prediction score is greater than 0.5.
Class E is treated as the anomaly class (label 1), and SB as the non-anomaly class (label 0).
Training may take significant time on a CPU (e.g., ~10 minutes for 5 epochs). To speed up, reduce the number of epochs or batch size in train_model.py.

Troubleshooting

Training Takes Too Long:
Reduce epochs in train_model.py (e.g., epochs=1).
Reduce batch size (e.g., batch_size=16).


API Errors:
Check Uvicorn logs for detailed error messages.
Ensure all dependencies are installed (pip install -r requirements.txt).


Database Connection Issues:
Verify PostgreSQL is running and the password in database.py is correct.



Results

The model was trained for 5 epochs, achieving a validation accuracy of approximately 86% (values may vary due to random initialization).
Example prediction:
Image: 311.jpg (from E class)
Result: Anomaly Detected
Score: 1.0



Future Improvements

Use a GPU to speed up training.
Fine-tune the ResNet50 model for better accuracy.
Add data augmentation to improve model robustness.
Implement additional API endpoints for batch processing.

