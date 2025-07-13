# 🚀 Project Setup Guide - Smart Detection of Malicious SMS

*Last Updated: 2025-07-13*

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 (required for TensorFlow 2.10 compatibility)
- **Flutter**: 3.16.0 or higher
- **Android SDK**: API level 21+ (Android 5.0+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for models and dependencies

### Development Environment
- **IDE**: VS Code, Android Studio, or IntelliJ IDEA
- **Git**: For version control
- **Android Device/Emulator**: For testing Flutter app

## 🏗️ Backend Setup (Python ML Pipeline)

### 1. Python Environment Setup

```bash
# Create virtual environment
python -m venv sms_fraud_detection_env

# Activate virtual environment
# Windows
sms_fraud_detection_env\Scripts\activate
# macOS/Linux
source sms_fraud_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies with Version Pinning

```bash
# Navigate to ML_Model directory
cd ML_Model

# Install exact versions to avoid conflicts
pip install scikit-learn==1.3.0
pip install tensorflow==2.10.0
pip install pandas==1.5.3
pip install numpy==1.24.3  # CRITICAL: NumPy ≥2.0 breaks TF 2.10
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install nltk==3.8.1
pip install joblib==1.3.2
```

### 3. Dependency Version Conflicts Troubleshooting

#### 3.1 NumPy Version Conflicts
```bash
# Problem: NumPy ≥2.0 breaks TensorFlow 2.10
# Error: "module 'numpy' has no attribute 'bool'"

# Solution 1: Force downgrade NumPy
pip install "numpy<2.0,>=1.24.3"

# Solution 2: Complete environment reset
pip uninstall numpy tensorflow scikit-learn -y
pip install numpy==1.24.3
pip install tensorflow==2.10.0
pip install scikit-learn==1.3.0

# Verify versions
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
```

#### 3.2 TensorFlow Compatibility Issues
```bash
# Problem: TensorFlow 2.10 requires specific Python version
# Error: "No matching distribution found for tensorflow==2.10.0"

# Solution: Check Python version compatibility
python --version  # Should be 3.9.x

# If using Python 3.11+, downgrade or use conda
conda create -n sms_fraud python=3.9
conda activate sms_fraud
pip install tensorflow==2.10.0
```

#### 3.3 Scikit-learn Version Conflicts
```bash
# Problem: Scikit-learn 1.4+ changes API
# Error: "ImportError: cannot import name 'LabelEncoder' from 'sklearn.preprocessing'"

# Solution: Pin to compatible version
pip install scikit-learn==1.3.0

# Check compatibility
python -c "from sklearn.preprocessing import LabelEncoder; print('OK')"
```

#### 3.4 Pandas Future Warnings
```bash
# Problem: Pandas 2.0+ deprecation warnings
# Warning: "FutureWarning: The behavior of DataFrame.to_numpy"

# Solution: Use compatible version
pip install pandas==1.5.3

# Suppress warnings in code (temporary)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```

### 4. Environment Verification Script

Create `verify_environment.py`:
```python
#!/usr/bin/env python3
"""Verify ML environment setup"""

import sys
import importlib

def check_version(package, expected_version):
    try:
        module = importlib.import_module(package)
        actual_version = module.__version__
        status = "✅" if actual_version == expected_version else "⚠️"
        print(f"{status} {package}: {actual_version} (expected: {expected_version})")
        return actual_version == expected_version
    except ImportError:
        print(f"❌ {package}: Not installed")
        return False

def main():
    print("🔍 Verifying ML Environment Setup")
    print("=" * 40)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python: {python_version}")
    
    # Check package versions
    requirements = {
        'numpy': '1.24.3',
        'tensorflow': '2.10.0',
        'sklearn': '1.3.0',
        'pandas': '1.5.3',
        'nltk': '3.8.1',
        'joblib': '1.3.2'
    }
    
    all_good = True
    for package, version in requirements.items():
        if not check_version(package, version):
            all_good = False
    
    if all_good:
        print("\n✅ All dependencies are correctly installed!")
    else:
        print("\n❌ Some dependencies need attention. See troubleshooting above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run verification:
```bash
python verify_environment.py
```

## 🔄 CI/CD Pipeline Setup

### 1. GitHub Actions Configuration

Create `.github/workflows/ci.yml`:
```yaml
name: SMS Fraud Detection CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-ml-model:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        cd ML_Model
        python -m pip install --upgrade pip
        pip install numpy==1.24.3  # Install NumPy first
        pip install tensorflow==2.10.0
        pip install scikit-learn==1.3.0
        pip install pandas==1.5.3
        pip install matplotlib==3.7.1
        pip install nltk==3.8.1
        pip install joblib==1.3.2
        pip install pytest==7.4.0
    
    - name: Verify environment
      run: |
        cd ML_Model
        python verify_environment.py
    
    - name: Download NLTK data
      run: |
        python -c "import nltk; nltk.download('stopwords')"
    
    - name: Run ML tests
      run: |
        cd ML_Model
        python -m pytest tests/ -v
    
    - name: Train model
      run: |
        cd ML_Model
        python train.py --test-mode
    
    - name: Test model export
      run: |
        cd ML_Model
        python export_tflite.py --verify
        python export_tfidf_vocab.py --verify
    
    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: ml-models
        path: |
          ML_Model/fraud_detector.tflite
          ML_Model/tfidf_vocab.json
          ML_Model/best_model.pkl
        retention-days: 30

  test-flutter-app:
    runs-on: ubuntu-latest
    needs: test-ml-model
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Flutter
      uses: subosito/flutter-action@v2
      with:
        flutter-version: '3.16.0'
        channel: 'stable'
    
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        name: ml-models
        path: sms_fraud_detectore_app/assets/
    
    - name: Install Flutter dependencies
      run: |
        cd sms_fraud_detectore_app
        flutter pub get
    
    - name: Run Flutter tests
      run: |
        cd sms_fraud_detectore_app
        flutter test
    
    - name: Analyze Flutter code
      run: |
        cd sms_fraud_detectore_app
        flutter analyze
    
    - name: Build APK
      run: |
        cd sms_fraud_detectore_app
        flutter build apk --release
    
    - name: Upload APK
      uses: actions/upload-artifact@v3
      with:
        name: app-release
        path: sms_fraud_detectore_app/build/app/outputs/apk/release/app-release.apk
        retention-days: 30

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif
    
    - name: Dependency vulnerability scan
      run: |
        cd ML_Model
        pip install safety
        safety check --json > safety-report.json || true
    
    - name: Upload security results
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          security-scan-results.sarif
          ML_Model/safety-report.json
```

### 2. Pre-commit Hooks Setup

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
  
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        files: ^ML_Model/.*\.py$
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        files: ^ML_Model/.*\.py$
        args: [--max-line-length=88]
  
  - repo: https://github.com/dart-lang/dart_style
    rev: 2.3.2
    hooks:
      - id: dart-format
        files: ^sms_fraud_detectore_app/.*\.dart$
```

Install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

### 3. Automated Testing Setup

Create `ML_Model/tests/test_model.py`:
```python
import pytest
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class TestMLModel:
    def test_model_loading(self):
        """Test that saved model can be loaded"""
        try:
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            assert model is not None
        except FileNotFoundError:
            pytest.skip("Model not found, run training first")
    
    def test_vectorizer_loading(self):
        """Test that vectorizer can be loaded"""
        try:
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            assert isinstance(vectorizer, TfidfVectorizer)
        except FileNotFoundError:
            pytest.skip("Vectorizer not found, run training first")
    
    def test_model_prediction(self):
        """Test model prediction functionality"""
        try:
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            
            # Test messages
            test_messages = [
                "Hello, how are you?",
                "URGENT: Click here to claim your prize!",
                "Your OTP is 123456"
            ]
            
            for message in test_messages:
                features = vectorizer.transform([message])
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]
                
                assert prediction in [0, 1]  # Valid class
                assert len(probability) == 2  # Two classes
                assert abs(sum(probability) - 1.0) < 0.001  # Probabilities sum to 1
                
        except FileNotFoundError:
            pytest.skip("Model files not found, run training first")
    
    def test_tflite_export(self):
        """Test TensorFlow Lite export"""
        import os
        if os.path.exists('fraud_detector.tflite'):
            file_size = os.path.getsize('fraud_detector.tflite')
            assert file_size > 1000  # Should be at least 1KB
            assert file_size < 1000000  # Should be less than 1MB
```

Create `ML_Model/tests/test_preprocessing.py`:
```python
import pytest
import re
from train import preprocess_text

class TestPreprocessing:
    def test_emoji_handling(self):
        """Test emoji preprocessing"""
        text_with_emoji = "Great offer! 😀🎉 Click now!"
        processed = preprocess_text(text_with_emoji)
        assert '😀' not in processed
        assert '🎉' not in processed
        assert 'great offer click now' in processed.lower()
    
    def test_special_characters(self):
        """Test special character handling"""
        text = "URGENT!!! Call now @ 123-456-7890 #winner"
        processed = preprocess_text(text)
        assert processed is not None
        assert len(processed) > 0
    
    def test_empty_text(self):
        """Test empty text handling"""
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
    
    def test_numeric_handling(self):
        """Test numeric content handling"""
        text = "Your OTP is 123456. Valid for 5 minutes."
        processed = preprocess_text(text)
        assert 'otp' in processed.lower()
        assert 'valid' in processed.lower()
```

### 4. Deployment Pipeline

Create `scripts/deploy.sh`:
```bash
#!/bin/bash

set -e

echo "🚀 Starting deployment pipeline..."

# 1. Run tests
echo "📝 Running tests..."
cd ML_Model
python -m pytest tests/ -v
cd ..

# 2. Train model
echo "🤖 Training model..."
cd ML_Model
python train.py
python export_tflite.py
python export_tfidf_vocab.py
cd ..

# 3. Copy assets to Flutter app
echo "📱 Updating Flutter assets..."
cp ML_Model/fraud_detector.tflite sms_fraud_detectore_app/assets/
cp ML_Model/tfidf_vocab.json sms_fraud_detectore_app/assets/

# 4. Build Flutter app
echo "🔨 Building Flutter app..."
cd sms_fraud_detectore_app
flutter clean
flutter pub get
flutter build apk --release
cd ..

# 5. Run final tests
echo "🧪 Running integration tests..."
cd sms_fraud_detectore_app
flutter test
cd ..

echo "✅ Deployment pipeline completed successfully!"
echo "📦 APK location: sms_fraud_detectore_app/build/app/outputs/apk/release/app-release.apk"
```

Make executable:
```bash
chmod +x scripts/deploy.sh
```

### 5. Monitoring and Alerts

Create `.github/workflows/monitor.yml`:
```yaml
name: Performance Monitoring

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:

jobs:
  performance-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Check model performance
      run: |
        cd ML_Model
        python performance_monitor.py
    
    - name: Send alerts if performance degraded
      if: failure()
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        text: "🚨 Model performance has degraded!"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## 📱 Frontend Setup (Flutter App)

### 1. Flutter Environment Setup

```bash
# Check Flutter installation
flutter doctor

# Ensure Android SDK is properly configured
flutter doctor --android-licenses
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone <repository-url>
cd sms_fraud_detectore_app

# Install dependencies
flutter pub get

# Copy ML model assets (if not done automatically)
cp ../ML_Model/fraud_detector.tflite assets/
cp ../ML_Model/tfidf_vocab.json assets/
```

### 3. Android Configuration

#### 3.1 Update `android/app/build.gradle`:
```gradle
android {
    compileSdkVersion 34
    
    defaultConfig {
        minSdkVersion 21
        targetSdkVersion 34
        versionCode flutterVersionCode.toInteger()
        versionName flutterVersionName
        
        // Enable multidex for large apps
        multiDexEnabled true
    }
    
    buildTypes {
        release {
            // Enable code shrinking
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
            signingConfig signingConfigs.debug
        }
    }
}

dependencies {
    implementation 'androidx.multidex:multidex:2.0.1'
}
```

#### 3.2 Create `android/app/proguard-rules.pro`:
```proguard
# Keep TensorFlow Lite classes
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }

# Keep model assets
-keep class **.tflite { *; }

# Keep Flutter classes
-keep class io.flutter.** { *; }
-keep class io.flutter.plugin.** { *; }
```

### 4. Permissions Configuration

Update `android/app/src/main/AndroidManifest.xml`:
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
          xmlns:tools="http://schemas.android.com/tools">
    
    <!-- SMS permissions -->
    <uses-permission android:name="android.permission.READ_SMS" />
    <uses-permission android:name="android.permission.RECEIVE_SMS" />
    
    <!-- Storage permissions -->
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" 
                     android:maxSdkVersion="28" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" 
                     android:maxSdkVersion="32" />
    
    <!-- Optional: for sender verification -->
    <uses-permission android:name="android.permission.INTERNET" />
    
    <application
        android:name="androidx.multidex.MultiDexApplication"
        android:label="SMS Fraud Detector"
        android:icon="@mipmap/ic_launcher">
        
        <!-- SMS receiver -->
        <receiver android:name=".SmsReceiver"
                  android:enabled="true"
                  android:exported="true">
            <intent-filter android:priority="1000">
                <action android:name="android.provider.Telephony.SMS_RECEIVED" />
            </intent-filter>
        </receiver>
        
        <!-- Main activity -->
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTop"
            android:theme="@style/LaunchTheme"
            android:configChanges="orientation|keyboardHidden|keyboard|screenSize|smallestScreenSize|locale|layoutDirection|fontScale|screenLayout|density|uiMode"
            android:hardwareAccelerated="true"
            android:windowSoftInputMode="adjustResize">
            
            <meta-data
                android:name="io.flutter.embedding.android.NormalTheme"
                android:resource="@style/NormalTheme" />
            
            <intent-filter>
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        
        <meta-data
            android:name="flutterEmbedding"
            android:value="2" />
    </application>
</manifest>
```

## 🧪 Testing and Validation

### 1. Unit Testing

```bash
# Test ML model
cd ML_Model
python -m pytest tests/ -v

# Test Flutter app
cd sms_fraud_detectore_app
flutter test
```

### 2. Integration Testing

```bash
# Full pipeline test
./scripts/deploy.sh

# Manual testing
cd sms_fraud_detectore_app
flutter run
```

### 3. Performance Testing

```bash
# ML model performance
cd ML_Model
python benchmark_model.py

# Flutter app performance
cd sms_fraud_detectore_app
flutter run --profile
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "No module named 'tensorflow'"
```bash
# Solution: Reinstall TensorFlow with correct version
pip uninstall tensorflow
pip install tensorflow==2.10.0
```

#### Issue: "NumPy version incompatibility"
```bash
# Solution: Downgrade NumPy
pip install "numpy<2.0,>=1.24.3"
```

#### Issue: Flutter build fails with "Execution failed for task ':app:compileFlutterBuildDebug'"
```bash
# Solution: Clean and rebuild
flutter clean
flutter pub get
flutter build apk --release
```

#### Issue: "Permission denied" for SMS access
```bash
# Solution: Check AndroidManifest.xml permissions and request runtime permissions
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
adb shell dumpsys meminfo com.example.sms_fraud_detectore_app
```

#### CPU Usage
```bash
# Monitor CPU usage
adb shell top -p $(adb shell pidof com.example.sms_fraud_detectore_app)
```

## 🚀 Production Deployment

### 1. Release Build

```bash
# Build release APK
cd sms_fraud_detectore_app
flutter build apk --release --shrink

# Build App Bundle (for Play Store)
flutter build appbundle --release
```

### 2. Signing Configuration

Create `android/key.properties`:
```properties
storePassword=your_store_password
keyPassword=your_key_password
keyAlias=your_key_alias
storeFile=path/to/your/keystore.jks
```

### 3. Play Store Upload

```bash
# Generate signed bundle
flutter build appbundle --release

# Upload to Play Console
# Use bundle: build/app/outputs/bundle/release/app-release.aab
```

---

*This setup guide provides comprehensive instructions for setting up the SMS Fraud Detection System with proper dependency management, CI/CD pipeline, and troubleshooting support.*