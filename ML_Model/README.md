# SMS Fraud Detection - ML Model

This directory contains the Python machine learning backend for the SMS Fraud Detection System. The model uses TF-IDF vectorization and Naive Bayes classification to detect fraudulent SMS messages, with enhanced sender validation to reduce false positives.

## 🎯 **Overview**

The ML pipeline processes SMS text data to identify fraudulent messages while minimizing false positives from legitimate sources like banks and trusted services. The model is optimized for mobile deployment using TensorFlow Lite.

## 🔧 **Features**

### **Advanced Text Processing**
- **TF-IDF Vectorization**: 3000-dimensional feature vectors
- **Text Preprocessing**: Cleaning, normalization, and tokenization
- **Vocabulary Management**: Optimized vocabulary for mobile deployment

### **Enhanced Detection Logic**
- **Naive Bayes Classification**: Probabilistic classification model
- **Sender Validation Integration**: Works with Flutter app's sender validation
- **False Positive Reduction**: Improved accuracy for legitimate messages

### **Mobile Optimization**
- **TensorFlow Lite Export**: Optimized for mobile inference
- **Direct Weight Export**: Compatible with TensorFlow 2.10
- **Python 3.9 Compatibility**: Stable environment for model training

## 📁 **File Structure**

```
ML_Model/
├── data/                    # Training data directory
│   └── sms+spam+collection/ # SMS spam collection dataset
├── data_set/               # Additional datasets
├── export_tflite.py        # TensorFlow Lite export script
├── export_tflite_simple.py # Simplified export script
├── export_tfidf_vocab.py   # TF-IDF vocabulary export
├── prepare_data.py         # Data preprocessing script
├── train.py               # Model training script
├── test_pipeline.py       # Pipeline testing script
├── requirements.txt       # Python dependencies
├── tfidf_vocab.json      # Exported TF-IDF vocabulary
└── README.md             # This file
```

## 🚀 **Quick Start**

### **1. Environment Setup**
```bash
# Create virtual environment
python -m venv .venv39

# Activate environment
# Windows:
.venv39\Scripts\activate
# macOS/Linux:
source .venv39/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Data Preparation**
```bash
# Prepare training data
python prepare_data.py

# This script will:
# - Load SMS spam collection dataset
# - Clean and preprocess text data
# - Create TF-IDF vocabulary
# - Split data into training/testing sets
```

### **3. Model Training**
```bash
# Train the model
python train.py

# This script will:
# - Train TF-IDF vectorizer
# - Train Naive Bayes classifier
# - Save model and vocabulary
# - Generate performance metrics
```

### **4. Model Export**
```bash
# Export for mobile deployment
python export_tflite.py

# This script will:
# - Convert model to TensorFlow Lite format
# - Optimize for mobile inference
# - Generate compatible model file
# - Export TF-IDF vocabulary
```

## 📊 **Model Architecture**

### **Text Processing Pipeline**
1. **Text Cleaning**: Remove special characters and normalize text
2. **Tokenization**: Split text into individual words
3. **TF-IDF Vectorization**: Convert text to 3000-dimensional vectors
4. **Feature Selection**: Select most informative features

### **Classification Model**
- **Algorithm**: Multinomial Naive Bayes
- **Features**: 3000-dimensional TF-IDF vectors
- **Output**: Binary classification (0 = legitimate, 1 = fraudulent)
- **Threshold**: 0.5 probability threshold

### **Performance Metrics**
- **Accuracy**: ~95% on test dataset
- **Precision**: High precision for fraudulent detection
- **Recall**: Good recall for legitimate messages
- **F1-Score**: Balanced performance metric

## 🔄 **Integration with Flutter App**

### **Sender Validation Integration**
The ML model works in conjunction with the Flutter app's sender validation:

1. **Sender Check**: Flutter app checks sender pattern first
2. **Trust Assessment**: Determines if sender is trusted
3. **ML Processing**: Only applies ML model to suspicious senders
4. **Result Combination**: Combines sender validation with ML prediction

### **Model Files for Flutter**
After export, copy these files to the Flutter app:
```
sms_fraud_detectore_app/assets/
├── fraud_detector.tflite    # TensorFlow Lite model
└── tfidf_vocab.json        # TF-IDF vocabulary
```

## 🧪 **Testing**

### **Pipeline Testing**
```bash
# Test the complete pipeline
python test_pipeline.py

# This will test:
# - Model loading and prediction
# - TF-IDF preprocessing
# - Export compatibility
# - Performance metrics
```

### **Manual Testing**
```python
# Test individual components
from fraud_detector import FraudDetector
from tfidf_preprocessor import TfidfPreprocessor

# Load model and preprocessor
detector = FraudDetector()
detector.load_model('fraud_detector.tflite')

preprocessor = TfidfPreprocessor()
preprocessor.load_vocab('tfidf_vocab.json')

# Test prediction
text = "Your account has been credited with $1000"
features = preprocessor.transform(text)
prediction = detector.predict(features)
print(f"Prediction: {prediction}")
```

## 📈 **Performance Optimization**

### **Model Optimization**
- **Feature Selection**: Select most informative TF-IDF features
- **Hyperparameter Tuning**: Optimize Naive Bayes parameters
- **Vocabulary Optimization**: Reduce vocabulary size for mobile

### **Export Optimization**
- **Quantization**: Reduce model size while maintaining accuracy
- **Pruning**: Remove unnecessary model parameters
- **TensorFlow Lite Flags**: Use optimization flags for mobile

## 🔧 **Configuration**

### **Model Parameters**
```python
# TF-IDF Parameters
MAX_FEATURES = 3000
MIN_DF = 2
MAX_DF = 0.95

# Naive Bayes Parameters
ALPHA = 1.0  # Smoothing parameter

# Export Parameters
MODEL_PATH = 'fraud_detector.tflite'
VOCAB_PATH = 'tfidf_vocab.json'
```

### **Data Configuration**
```python
# Dataset paths
SMS_DATA_PATH = 'data/sms+spam+collection/SMSSpamCollection'
ADDITIONAL_DATA_PATH = 'data_set/'

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **TensorFlow Installation**
```bash
# If TensorFlow installation fails
pip uninstall tensorflow
pip install tensorFlow==2.10.0

# If version conflicts occur
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### **Model Export Issues**
- Ensure TensorFlow version is 2.10.0
- Check Python version is 3.9
- Verify all dependencies are installed

#### **Performance Issues**
- Reduce MAX_FEATURES for faster processing
- Use smaller vocabulary for mobile optimization
- Consider model quantization for size reduction

### **Debug Mode**
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 **Model Evaluation**

### **Metrics Calculation**
```python
from sklearn.metrics import classification_report, confusion_matrix

# Generate detailed metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### **Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## 🔄 **Model Updates**

### **Retraining Process**
1. **Data Collection**: Gather new SMS data
2. **Data Preparation**: Clean and preprocess new data
3. **Model Retraining**: Retrain with updated dataset
4. **Performance Evaluation**: Test on validation set
5. **Model Export**: Export updated model for mobile

### **Version Control**
- Track model versions and performance
- Maintain dataset versions
- Document changes and improvements

## 📚 **References**

### **Papers and Research**
- TF-IDF: Term Frequency-Inverse Document Frequency
- Naive Bayes Classification
- SMS Spam Detection Techniques

### **Libraries and Tools**
- **scikit-learn**: Machine learning library
- **TensorFlow**: Deep learning framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## 🤝 **Contributing**

### **Development Guidelines**
1. Follow Python PEP 8 style guidelines
2. Add tests for new features
3. Update documentation for changes
4. Maintain backward compatibility

### **Testing Requirements**
- Unit tests for all functions
- Integration tests for pipeline
- Performance benchmarks
- Mobile compatibility tests

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Python, TensorFlow, and scikit-learn** 