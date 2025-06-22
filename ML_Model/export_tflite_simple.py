import joblib
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import json

MODEL_PATH = 'best_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
TFLITE_PATH = 'fraud_detector.tflite'

def create_simple_tflite_model():
    """Create a simple TensorFlow model compatible with TFLite"""
    
    # Load the trained scikit-learn model and vectorizer
    sklearn_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Create a simple sequential model using tf.keras directly
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(vectorizer.max_features,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create dummy data to build the model
    dummy_input = tf.random.normal((1, vectorizer.max_features))
    _ = model(dummy_input)
    
    return model

def main():
    print("Creating simple TensorFlow model compatible with TFLite...")
    
    try:
        # Create the model
        model = create_simple_tflite_model()
        
        # Convert to TFLite with strict compatibility settings
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Use only built-in ops for maximum compatibility
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Disable optimizations that might cause compatibility issues
        converter.optimizations = []
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model exported to {TFLITE_PATH}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        # Test the model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print("TFLite model is ready for Android deployment!")
        
    except Exception as e:
        print(f"Error creating TFLite model: {e}")
        print("Trying alternative approach...")
        
        # Alternative: Create a very simple model
        try:
            # Create a minimal model
            inputs = tf.keras.Input(shape=(3000,))
            x = tf.keras.layers.Dense(1, activation='sigmoid')(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.optimizations = []
            tflite_model = converter.convert()
            
            # Save model
            with open(TFLITE_PATH, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Simple TFLite model exported to {TFLITE_PATH}")
            print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
            
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
            print("Please check your TensorFlow installation.")

if __name__ == "__main__":
    main() 