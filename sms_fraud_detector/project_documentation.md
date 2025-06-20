# SMS Fraud Detector — Project Documentation

## Overview
A cross-platform (Flutter) mobile app that detects fraudulent SMS messages in real time using a machine learning model (TF-IDF + classifier) trained in Python and deployed via TensorFlow Lite (TFLite). All processing is local for privacy.

---

## System Architecture

### 1. **ML Layer (Python)**
- **Dataset:** UCI SMS Spam Collection (or custom, labeled as ham/spam)
- **Preprocessing:**
  - Clean text (lowercase, remove punctuation, stopwords)
  - TF-IDF vectorization (scikit-learn)
- **Model Training:**
  - Naive Bayes (baseline)
  - XGBoost (optional, for higher accuracy)
- **Evaluation:**
  - Accuracy, precision, recall, confusion matrix
- **Export:**
  - Best model to `fraud_detector.tflite` (via Keras if needed)
  - TF-IDF vectorizer vocabulary/IDF to `tfidf_vocab.json`

### 2. **Flutter App (Dart)**
- **SMS Receiver:** Listens for incoming SMS using `telephony` plugin
- **Preprocessing:**
  - Loads TF-IDF vocab/IDF from JSON
  - Cleans and vectorizes SMS text to match Python pipeline
- **Inference:**
  - Loads TFLite model using `tflite_flutter`
  - Runs prediction on SMS vector
- **UI:**
  - SMS log/history
  - Real-time detection result (Legitimate/Fraudulent)
  - Toggle for detection ON/OFF
  - Feedback button ("Mark as mistake")
  - Notifications/alerts

---

## ML Pipeline (Python)

1. **Data Preparation**
   - Download and place dataset as `data/sms_spam.csv`
   - Format: `label,text` (label: ham/spam)

2. **Training** (`train.py`)
   - Loads and cleans data
   - Splits into train/test
   - Fits TF-IDF vectorizer
   - Trains Naive Bayes and XGBoost
   - Evaluates and saves best model as `best_model.pkl`
   - Saves vectorizer as `vectorizer.pkl`

3. **Export**
   - `export_tflite.py`: Converts Keras model to TFLite (`fraud_detector.tflite`)
   - `export_tfidf_vocab.py`: Exports TF-IDF vocab/IDF as `tfidf_vocab.json`

---

## Flutter App — Code Structure

```
sms_fraud_detector/
├── lib/
│   ├── main.dart                # App entry point
│   ├── sms_log_page.dart        # Main UI: SMS log, toggle, feedback
│   ├── tfidf_preprocessor.dart  # Loads vocab/IDF, vectorizes SMS
│   ├── fraud_detector.dart      # Loads TFLite model, runs inference
│   ├── sms_receiver.dart        # Listens for incoming SMS
│   └── widgets/                 # (Optional) UI widgets
├── assets/
│   ├── fraud_detector.tflite    # TFLite model
│   └── tfidf_vocab.json         # TF-IDF vocab/IDF
├── pubspec.yaml                 # Dependencies, asset paths
```

---

## Component Details

### **1. TF-IDF Preprocessor (`tfidf_preprocessor.dart`)**
- Loads vocab/IDF from JSON (exported from Python)
- Cleans SMS text (lowercase, remove punctuation, stopwords)
- Converts text to TF-IDF feature vector (same as Python)

### **2. Fraud Detector (`fraud_detector.dart`)**
- Loads TFLite model from assets
- Accepts TF-IDF vector as input
- Runs inference, outputs 0 (legitimate) or 1 (fraudulent)

### **3. SMS Receiver (`sms_receiver.dart`)**
- Uses `telephony` plugin to listen for incoming SMS
- On new SMS, extracts sender and body, passes to detection pipeline

### **4. Main UI (`sms_log_page.dart`, `main.dart`)**
- Displays list of received SMS with detection results
- Toggle to enable/disable detection
- Button to mark detection as mistake (feedback)
- Shows notifications/alerts for fraudulent SMS

---

## Workflow: End-to-End

1. **User receives SMS**
2. **SMS Receiver** triggers, extracts SMS body
3. **TF-IDF Preprocessor** cleans and vectorizes SMS
4. **Fraud Detector** runs TFLite inference
5. **UI** updates log, shows result, and notifies user if fraudulent
6. **User** can mark as mistake (feedback for future improvement)

---

## Privacy & Security
- All processing is on-device
- No SMS or data is uploaded or sent to any server
- Model and vocab are bundled as assets
- User can disable detection at any time

---

## How to Build & Run

1. **Train and export model in Python**
   - Place `fraud_detector.tflite` and `tfidf_vocab.json` in `assets/`
2. **Install Flutter dependencies**
   - Run `flutter pub get`
3. **Run the app**
   - `flutter run` (on device/emulator)
4. **Grant SMS permissions** when prompted

---

## Extending the Project
- Add deep learning models (LSTM/BERT) for higher accuracy (requires more resources)
- Add user feedback loop to improve model
- Support iOS (with minor changes)
- Add analytics (local only) for detection stats

---

## References
- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [Flutter](https://flutter.dev/)
- [tflite_flutter](https://pub.dev/packages/tflite_flutter)
- [telephony](https://pub.dev/packages/telephony)
- [scikit-learn](https://scikit-learn.org/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

---

## Contact / Authors
- Project by: [Your Name]
- For questions, contact: [Your Email] 