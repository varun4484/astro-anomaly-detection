import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from preprocess import get_data_generator
import numpy as np

def build_resnet50_model():
    """Build and compile ResNet50 model for binary classification."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze base model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(data_dir, model_path='models/best_model.h5', epochs=30):
    """Train the ResNet50 model."""
    try:
        train_generator, val_generator = get_data_generator(data_dir)
        model = build_resnet50_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
        ]
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        # Evaluate on validation set
        val_generator.reset()
        y_pred = (model.predict(val_generator) > 0.5).astype("int32")
        y_true = val_generator.classes
        print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
        return model, history
    except Exception as e:
        print(f"Error during training: {e}")
        raise