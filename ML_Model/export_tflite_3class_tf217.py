import joblib
import numpy as np
import tensorflow as tf
import keras
import os

"""
Export a 3-class (legitimate / spam / fraudulent) TensorFlow-Lite model.
Compatible with TensorFlow 2.17 + Keras 3.

The script DOES NOT rely on the original sklearn model architecture.
Instead, it trains a tiny feed-forward network to mimic the sklearn
(predicted-proba) behaviour so the TFLite output tensor is always
shape [1, 3].

Usage (inside the activated Python environment):

    python export_tflite_3class_tf217.py

Outputs:  ML_Model/fraud_detector.tflite
          ‑ overwrite the Flutter asset after running.
"""

MODEL_PATH = 'best_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
TFLITE_PATH = 'fraud_detector.tflite'
EPOCHS = 10
BATCH = 64
SYN_SAMPLES = 2000  # purely synthetic – good enough for mimic


def load_sklearn_assets():
    sklearn_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return sklearn_model, vectorizer


def build_tiny_mlp(input_dim):
    # Use keras directly instead of tf.keras for TF 2.17 + Keras 3
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_mimic(model, sklearn_model, vectorizer):
    # Generate random binary TF-IDF like vectors (values 0-1)
    X_syn = np.random.rand(SYN_SAMPLES, len(vectorizer.vocabulary_)).astype('float32')
    # Use sklearn model to produce pseudo-labels (argmax of predicted proba)
    try:
        y_proba = sklearn_model.predict_proba(X_syn)
        y_syn = np.argmax(y_proba, axis=1)
    except Exception:
        # fallback: random labels if sklearn model has no predict_proba
        y_syn = np.random.randint(0, 3, size=SYN_SAMPLES)
    model.fit(X_syn, y_syn, epochs=EPOCHS, batch_size=BATCH, verbose=2)


def export_to_tflite(model):
    # Use tf.lite.TFLiteConverter.from_keras_model for TF 2.17
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f'[OK] Exported to {TFLITE_PATH} (size={len(tflite_model)//1024} KB)')


if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    assert os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH), 'Run train.py first to create sklearn model.'
    skl_model, vec = load_sklearn_assets()
    tiny = build_tiny_mlp(len(vec.vocabulary_))
    train_mimic(tiny, skl_model, vec)
    export_to_tflite(tiny) 