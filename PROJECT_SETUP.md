# 🚀 SMS Fraud Detection System - Complete Setup Guide  _(updated 2025-06-21)_

## Overview
This project implements an AI-powered SMS fraud detection system with a Python ML backend and a Flutter Android app frontend. The system provides real-time SMS fraud detection with a Google Messages-like interface.

## System Architecture
- **ML Backend**: Python 3.9 + TensorFlow 2.10 + Scikit-learn
- **Mobile App**: Flutter with TensorFlow Lite integration
- **Features**: Device SMS sync, real-time detection, thread-based UI, fraud alerts

## Prerequisites

### Required Software
- **Python 3.9.13** (required for TensorFlow 2.10 compatibility)
- **Flutter SDK** (latest stable version)
- **Android Studio** (for Android development)
- **Git** (for version control)

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for ML training)
- **Storage**: At least 5GB free space
- **Android Device**: For testing the mobile app

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Smart Detection of Malicious SMS"
```

### 2. ML Model Setup

#### 2.1 Create Python 3.9 Virtual Environment
```bash
cd ML_Model
python -m venv .venv39
```

#### 2.2 Activate Virtual Environment
**Windows:**
```bash
.venv39\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv39/bin/activate
```

#### 2.3 Install Python Dependencies
```bash
pip install "numpy<2.0"
pip install tensorflow==2.10.0
pip install scikit-learn==1.6.1
pip install pandas==2.3.0
pip install joblib==1.5.1
```

#### 2.4 Train and Export ML Model
```bash
python train.py
python export_tflite.py
```

**Expected Output:**
```
[info] Training TensorFlow model to mimic sklearn behavior...
[success] Model exported to fraud_detector.tflite
[info] Input shape: [   1 3000]
[info] Output shape: [1 1]
[info] Model size: 192.1 KB
```

#### 2.5 Copy Model Files to Flutter App
```bash
copy fraud_detector.tflite ..\sms_fraud_detectore_app\assets\
copy tfidf_vocab.json ..\sms_fraud_detectore_app\assets\
```

### 3. Flutter App Setup

#### 3.1 Navigate to Flutter Project
```bash
cd ../sms_fraud_detectore_app
```

#### 3.2 Install Flutter Dependencies
```bash
flutter pub get
```

#### 3.3 Verify Dependencies
The following packages should be installed:
- `tflite_flutter: ^0.11.0` (TensorFlow Lite)
- `provider: ^6.0.0` (State management)
- `telephony: ^0.2.0` (SMS access)
- `permission_handler: ^11.0.1` (Permissions)

#### 3.4 Android Permissions
The app requires the following permissions (already configured in `android/app/src/main/AndroidManifest.xml`):
```xml
<uses-permission android:name="android.permission.READ_SMS" />
<uses-permission android:name="android.permission.RECEIVE_SMS" />
<uses-permission android:name="android.permission.SEND_SMS" />
<uses-permission android:name="android.permission.READ_CONTACTS" />
<uses-permission android:name="android.permission.READ_PHONE_STATE" />
<uses-permission android:name="android.permission.READ_PHONE_NUMBERS" />
```

### 4. Build and Deploy

#### 4.1 Build Flutter App
```bash
flutter clean
flutter build apk --debug
```

#### 4.2 Install on Device
```bash
flutter install
```

**Or manually install the APK:**
```bash
adb install build\app\outputs\flutter-apk\app-debug.apk
```

## Project Structure

```
Smart Detection of Malicious SMS/
├── ML_Model/                          # Python ML backend
│   ├── .venv39/                       # Python 3.9 virtual environment
│   ├── data/                          # Training datasets
│   ├── train.py                       # Model training script
│   ├── export_tflite.py              # TensorFlow Lite export
│   ├── fraud_detector.tflite         # Exported ML model
│   └── tfidf_vocab.json              # TF-IDF vocabulary
├── sms_fraud_detectore_app/           # Flutter mobile app
│   ├── lib/                          # Dart source code
│   │   ├── main.dart                 # App entry point
│   │   ├── fraud_detector.dart       # TensorFlow Lite integration
│   │   ├── sms_log_state.dart        # State management
│   │   ├── thread_list_page.dart     # SMS thread list UI
│   │   ├── thread_page.dart          # Individual chat UI
│   │   └── sms_permission_helper.dart # Permission handling
│   ├── assets/                       # App assets
│   │   ├── fraud_detector.tflite     # ML model
│   │   └── tfidf_vocab.json          # Vocabulary
│   └── android/                      # Android configuration
└── Documentation/                    # Project documentation
```

## Key Features

### ML Model Features
- **Model Type**: Multinomial Naive Bayes with TensorFlow mimic
- **Input**: TF-IDF features (3000 dimensions)
- **Output**: Binary classification (legitimate/fraudulent)
- **Model Size**: ~192KB (optimized for mobile)
- **Compatibility**: TensorFlow 2.10 + TFLite

### Flutter App Features
- **SMS Sync**: Complete device SMS synchronization
- **Real-time Detection**: Live fraud detection for new SMS
- **Thread-based UI**: Google Messages-like interface
- **Fraud Alerts**: Visual indicators for fraudulent messages
- **Manual Sync**: User-triggered SMS refresh
- **Permission Management**: Runtime permission handling

### UI Components
- **Thread List**: All SMS conversations with fraud indicators
- **Chat Interface**: Individual conversation threads
- **Detection Dashboard**: Statistics and controls
- **Logs Page**: Detailed detection history
- **Sync Indicators**: Loading states and progress

## Troubleshooting

### Common Issues

#### 1. TensorFlow Import Errors
**Problem**: NumPy version conflicts
**Solution**: Ensure NumPy < 2.0 is installed
```bash
pip install "numpy<2.0"
```

#### 2. Flutter Build Failures
**Problem**: Kotlin version conflicts
**Solution**: Use compatible packages (telephony instead of sms_advanced)
```bash
flutter clean
flutter pub get
```

#### 3. Permission Denied
**Problem**: SMS permissions not granted
**Solution**: Grant permissions manually in device settings or reinstall app

#### 4. Model Loading Errors
**Problem**: TFLite model not found
**Solution**: Ensure model files are in assets directory
```bash
copy ML_Model\fraud_detector.tflite sms_fraud_detectore_app\assets\
```

### Performance Optimization
- **Model Size**: Optimized to ~192KB for mobile deployment
- **Memory Usage**: Efficient state management with Provider
- **UI Performance**: Lazy loading and optimized list rendering

## Development Workflow

### 1. ML Model Updates
1. Modify training data in `ML_Model/data/`
2. Retrain model: `python train.py`
3. Export to TFLite: `python export_tflite.py`
4. Copy to Flutter assets
5. Rebuild Flutter app

### 2. Flutter App Updates
1. Modify Dart code in `lib/`
2. Test changes: `flutter run`
3. Build for deployment: `flutter build apk --release`

### 3. Testing
1. **Unit Tests**: `flutter test`
2. **Integration Tests**: Manual testing on device
3. **Performance Tests**: Monitor memory and CPU usage

## Deployment

### Production Build
```bash
flutter build apk --release
```

### App Store Deployment
1. Generate signed APK
2. Test thoroughly on multiple devices
3. Upload to Google Play Store

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs in Android Studio
3. Verify all dependencies are correctly installed
4. Ensure Python 3.9 and TensorFlow 2.10 compatibility

## Version History

### v2.0 (Current)
- Complete SMS sync implementation
- Google Messages-like UI
- TensorFlow 2.10 compatibility
- Enhanced fraud detection
- Real-time SMS monitoring

### v1.0 (Previous)
- Basic SMS detection
- Simple UI
- TensorFlow Lite integration

## 📋 Prerequisites

- **Python 3.9** (required for TensorFlow 2.10 export)
- **Flutter SDK** (latest stable)
- **Android Studio** (for Android development)
- **Git** (for version control)

## 🏗️ Project Structure

```
Smart Detection of Malicious SMS/
├── ML_Model/                    # Python ML pipeline
│   ├── train.py                # Model training
│   ├── export_tflite.py        # TFLite export
│   ├── export_tfidf_vocab.py   # Vocabulary export
│   ├── requirements.txt        # Python dependencies
│   ├── best_model.pkl          # ✅ Generated
│   ├── vectorizer.pkl          # ✅ Generated
│   ├── fraud_detector.tflite   # ✅ Generated
│   ├── tfidf_vocab.json        # ✅ Generated
│   └── data/                   # Processed dataset
│       └── sms_spam.csv        # Used for training
├── sms_fraud_detectore_app/    # Flutter Android app
│   ├── lib/                    # Dart source code
│   ├── pubspec.yaml           # Flutter dependencies
│   └── assets/                # Model files (generated)
│       ├── fraud_detector.tflite   # ✅ Present
│       └── tfidf_vocab.json        # ✅ Present
└── PROJECT_SETUP.md           # This file
```

## 🛠️ Step-by-Step Setup

### Phase 1: ML Model Setup (Python)

> **Note:** The ML model files (`best_model.pkl`, `vectorizer.pkl`, `fraud_detector.tflite`, `tfidf_vocab.json`) have already been generated and are present in the ML_Model directory. Data preparation and export scripts were fixed and run successfully.

#### 1.1 Create an isolated Python 3.9 environment
> **Why?** TensorFlow ≤ 2.10 still writes TFLite op-versions that the mobile runtime understands.  
> Wheels for these versions only exist up to Python 3.9.

```powershell
# From the repo root
cd ML_Model
# Create & activate venv (Windows PowerShell)
py -3.9 -m venv .venv39
.\.venv39\Scripts\Activate.ps1

# Install minimal deps
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir \
    tensorflow-cpu==2.10.0 \
    scikit-learn pandas joblib
```

#### 1.2 (Re-)export model & vocabulary
```powershell
# Still inside ML_Model & venv
python train.py                 # optional – retrain MultinomialNB
python export_tfidf_vocab.py    # regenerates tfidf_vocab.json
python export_tflite.py         # generates fraud_detector.tflite (≈ 386 KB)

# Copy artefacts to Flutter assets
Copy-Item fraud_detector.tflite ..\sms_fraud_detectore_app\assets\ -Force
Copy-Item tfidf_vocab.json      ..\sms_fraud_detectore_app\assets\ -Force
```

You can now `deactivate` the venv and continue with the Flutter steps.

### Phase 2: Flutter App Setup

#### 2.1 Install Flutter Dependencies
```bash
cd sms_fraud_detectore_app
flutter pub get
```

#### 2.2 Updated pubspec.yaml dependencies
```
dependencies:
  flutter: { sdk: flutter }
  tflite_flutter: ^0.11.0  # stable runtime (op v2 support)
  sms_receiver: ^0.4.2     # AGP-8 compatible SMS listener
  provider: ^6.0.0
```

#### 2.3 SMS Permissions
`sms_receiver` relies on the Android **SMS User Consent API** (no direct READ/RECEIVE permission).  No extra permissions are required in the manifest.  If you kept the old `<uses-permission android:name="android.permission.RECEIVE_SMS"/>` lines they can stay but are not strictly needed.

#### 2.4 Gradle compatibility workaround
The root-level `android/build.gradle.kts` now automatically
1. assigns a fallback `android.namespace` to legacy libraries (fixes the AGP-8 "Namespace not specified" error);
2. strips the obsolete `package="…"` attribute from third-party manifests at build-time.
No manual action required — the script runs each build.

#### 2.5 Build and Run
```bash
flutter clean   # optional first time after upgrade
flutter run     # debug build on attached device
```

#### 2.6 TFLite model compatibility note
`fraud_detector.tflite` is exported with **TensorFlow 2.10 + TFLITE_BUILTINS** only.  
If you retrain later, always use the Python 3.9 venv and make sure the export script leaves `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]` unchanged.

## 🧪 Testing the System

### 1. Test ML Model (Python)
```bash
cd ML_Model
python -c "
import joblib
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('best_model.pkl')

test_messages = [
    'Hi mom, can you pick me up?',
    'URGENT: Your account suspended. Click here: http://fake.com',
    'Meeting at 3 PM tomorrow',
    'CONGRATULATIONS! You won $1,000,000!'
]

for msg in test_messages:
    features = vectorizer.transform([msg])
    pred = model.predict(features)[0]
    print(f'{msg} -> {"Fraudulent" if pred else "Legitimate"}')
"
```

### 2. Test Flutter App
1. Install the app on your Android device
2. Grant SMS permissions when prompted
3. Use the "Test Detection" button to simulate SMS
4. Check the SMS log for detection results

## 📱 App Features

### Main Screen
- **Status Indicator**: Shows if detection is active
- **Toggle Switch**: Enable/disable detection
- **Statistics**: Real-time SMS counts
- **Test Button**: Simulate SMS detection

### SMS Log
- **History**: All processed SMS messages
- **Classification**: Legitimate vs Fraudulent
- **Details**: Sender, timestamp, confidence

### Notifications
- **Real-time Alerts**: For incoming SMS
- **Color-coded**: Green (legitimate) vs Red (fraudulent)
- **Quick Actions**: View details or dismiss

## 🔍 Troubleshooting

### Common Issues

#### ML Model Issues
```bash
# Missing dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Memory issues
# Reduce max_features in train.py (line 35)

# Dataset format
# Ensure CSV has 'label' and 'text' columns
```

#### Flutter Issues
```bash
# Dependencies
flutter clean
flutter pub get

# Android permissions
# Check AndroidManifest.xml

# Model files
# Ensure assets/ directory contains required files
```

#### TFLite Issues
```bash
# Model compatibility
# Check TensorFlow version compatibility

# File size
# Use quantization in export_tflite.py
```

## ✅ Project is Ready for End-to-End Testing

All model and vocab files are in place. You can now run the app and test real-time SMS fraud detection!

## 📊 Expected Performance

### Model Metrics
- **Accuracy**: 95-98%
- **Precision**: 90-95%
- **Recall**: 85-90%
- **Model Size**: < 1MB

### App Performance
- **Inference Time**: < 100ms per SMS
- **Memory Usage**: < 50MB
- **Battery Impact**: Minimal (local processing)

## 🚀 Deployment

### Development
```bash
# Run in debug mode
flutter run --debug
```

### Production
```bash
# Build APK
flutter build apk --release

# Build App Bundle
flutter build appbundle --release
```

### Distribution
- **Internal Testing**: Use APK file
- **Play Store**: Use App Bundle
- **Enterprise**: Custom distribution

## 📈 Next Steps

### Immediate Improvements
1. **Better UI/UX**: Material Design 3
2. **Advanced Features**: SMS blocking, reporting
3. **Performance**: Model optimization
4. **Testing**: Unit and integration tests

### Future Enhancements
1. **Deep Learning**: LSTM/Transformer models
2. **Privacy**: Federated learning
3. **Real-time Learning**: Online model updates
4. **Cross-platform**: iOS support

## 📚 Resources

- [Flutter Documentation](https://flutter.dev/docs)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [SMS Spam Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Android Permissions](https://developer.android.com/guide/topics/permissions)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs
3. Test individual components
4. Verify file paths and permissions

---

**Happy Coding! 🎉** 

## Enhanced Features Setup

### Sender Validation Configuration
The sender validation feature is automatically configured and includes:

**Trusted Sender Patterns:**
- Alphanumeric IDs: `^[A-Z0-9\-]{3,15}$`
- Short Codes: `^[0-9]{4,6}$`
- Examples: HDFCBK, VM-AIRTEL, BX-ICICIB

**Suspicious Sender Patterns:**
- Phone Numbers: `^\+[0-9]{1,4}`
- Examples: +91 98123xxxxx, +1 555-1234

### Detection Dashboard Features
The enhanced dashboard includes:
- **Status Card**: Dynamic protection status with animations
- **Statistics Grid**: 4-metric dashboard with real-time updates
- **Control Panel**: Professional settings interface
- **Activity Feed**: Recent detection history
- **Scan Button**: Manual scanning functionality

## Testing and Validation

### 1. Model Testing
```bash
cd ML_Model
python test_pipeline.py

# This will:
# - Test model loading
# - Validate predictions
# - Check TF-IDF preprocessing
# - Verify export compatibility
```

### 2. App Testing
```bash
cd sms_fraud_detectore_app

# Run tests
flutter test

# Test on device
flutter run --debug
```

### 3. Sender Validation Testing
Test the sender validation with various sender patterns:
- **Trusted**: HDFCBK, VM-AIRTEL, 12345
- **Suspicious**: +91 98123xxxxx, +1 555-1234
- **Unknown**: Random strings

### 4. Dashboard Testing
Verify dashboard functionality:
- **Status Updates**: Toggle protection on/off
- **Statistics**: Check real-time metric updates
- **Animations**: Verify smooth transitions
- **Responsiveness**: Test on different screen sizes

## Troubleshooting

### Common Issues

#### Python/TensorFlow Issues
```bash
# If TensorFlow installation fails
pip uninstall tensorflow
pip install tensorflow==2.10.0

# If version conflicts occur
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### Flutter Issues
```bash
# Clean and rebuild
flutter clean
flutter pub get
flutter run

# If dependencies fail
flutter pub cache repair
flutter pub get
```

#### Model Loading Issues
- Ensure `fraud_detector.tflite` is in `assets/` folder
- Verify `tfidf_vocab.json` is properly formatted
- Check TensorFlow Lite compatibility

#### SMS Permission Issues
- Grant SMS permissions manually in device settings
- Test with different Android versions
- Verify telephony package integration

### Debug Mode
Enable debug logging in the app:
```dart
// In main.dart
debugPrint('Sender: $sender, Pattern: ${_analyzeSender(sender)}');
debugPrint('ML Prediction: $prediction, Reason: $reason');
```

## Performance Optimization

### Model Optimization
- **Quantization**: Reduce model size while maintaining accuracy
- **Pruning**: Remove unnecessary model parameters
- **Optimization**: Use TensorFlow Lite optimization flags

### App Performance
- **Lazy Loading**: Load components on demand
- **Caching**: Cache processed results
- **Background Processing**: Use isolates for heavy computations

## Deployment

### Production Build
```bash
# Build optimized APK
flutter build apk --release --target-platform android-arm64

# Build App Bundle (recommended for Play Store)
flutter build appbundle --release
```

### Distribution
- **Internal Testing**: Use Firebase App Distribution
- **Play Store**: Follow Google Play guidelines
- **Direct APK**: Share APK file directly

## Maintenance

### Regular Updates
- **Dependencies**: Keep Flutter and packages updated
- **Model**: Retrain model with new data periodically
- **Security**: Update Android permissions as needed

### Monitoring
- **Performance**: Monitor app performance metrics
- **Accuracy**: Track detection accuracy over time
- **User Feedback**: Collect and analyze user reports

## Support

### Documentation
- **API Reference**: Flutter and TensorFlow documentation
- **Community**: Flutter and TensorFlow communities
- **Issues**: GitHub issues for bug reports

### Resources
- **Flutter Docs**: https://flutter.dev/docs
- **TensorFlow Docs**: https://tensorflow.org/docs
- **Telephony Package**: https://pub.dev/packages/telephony

## Conclusion

This setup guide provides all necessary steps to get the SMS Fraud Detection System running with its enhanced features. The combination of sender validation and modern UI creates a robust, user-friendly security application.

For additional support or questions, refer to the project documentation or create an issue in the project repository. 