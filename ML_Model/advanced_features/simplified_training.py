#!/usr/bin/env python3
"""
Simplified Enhanced SMS Fraud Detection Training
Uses behavioral analysis without heavy semantic dependencies
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

class SimplifiedBehavioralDetector:
    def __init__(self):
        """Initialize simplified behavioral detector"""
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = None
        
        # Behavioral patterns for fraud detection
        self.urgency_patterns = [
            r'\b(urgent|immediate|now|asap|quickly|hurry|rush)\b',
            r'\b(expire|expires|expiring|deadline|limited time)\b',
            r'\b(act now|call now|click now|verify now|update now)\b'
        ]
        
        self.fear_patterns = [
            r'\b(suspended|blocked|closed|terminated|disabled)\b',
            r'\b(unauthorized|suspicious|fraud|security|breach)\b',
            r'\b(warning|alert|notice|violation|penalty)\b'
        ]
        
        self.reward_patterns = [
            r'\b(win|won|winner|prize|reward|gift|free|bonus)\b',
            r'\b(congratulations|selected|chosen|lucky|special)\b',
            r'\b(cashback|refund|money|cash|₹|rupees|lakh|crore)\b'
        ]
        
        self.authority_patterns = [
            r'\b(bank|government|tax|income tax|rbi|sebi|police)\b',
            r'\b(official|authorized|verified|certified|legal)\b',
            r'\b(department|ministry|bureau|agency|authority)\b'
        ]
        
        self.action_patterns = [
            r'\b(click|tap|call|reply|send|forward|share|download)\b',
            r'\b(verify|confirm|update|activate|register|submit)\b',
            r'\b(provide|enter|give|send|share|disclose)\b'
        ]

    def extract_behavioral_features(self, text: str) -> dict:
        """Extract behavioral features from text"""
        text_lower = text.lower()
        features = {}
        
        # 1. Pattern-based features
        features['urgency_score'] = sum(len(re.findall(pattern, text_lower)) for pattern in self.urgency_patterns)
        features['fear_score'] = sum(len(re.findall(pattern, text_lower)) for pattern in self.fear_patterns)
        features['reward_score'] = sum(len(re.findall(pattern, text_lower)) for pattern in self.reward_patterns)
        features['authority_score'] = sum(len(re.findall(pattern, text_lower)) for pattern in self.authority_patterns)
        features['action_score'] = sum(len(re.findall(pattern, text_lower)) for pattern in self.action_patterns)
        
        # 2. Structural features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # 3. Communication features
        features['has_url'] = 1 if re.search(r'http[s]?://|www\.|\.[a-z]{2,3}/', text_lower) else 0
        features['has_phone'] = 1 if re.search(r'\+?\d{10,}', text) else 0
        features['all_caps_words'] = len([word for word in text.split() if word.isupper() and len(word) > 2])
        
        return features

    def extract_combined_features(self, texts):
        """Combine TF-IDF with behavioral features"""
        # Get TF-IDF features
        tfidf_features = self.vectorizer.transform(texts).toarray()
        
        # Extract behavioral features
        behavioral_features = []
        for text in texts:
            features = self.extract_behavioral_features(text)
            behavioral_features.append(list(features.values()))
        
        behavioral_features = np.array(behavioral_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, behavioral_features])
        return combined_features

    def train(self, texts, labels):
        """Train the fraud detection model"""
        print("Extracting TF-IDF features...")
        self.vectorizer.fit(texts)
        
        print("Extracting combined features...")
        X = self.extract_combined_features(texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        print("Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
        
        return {'accuracy': accuracy}

    def predict(self, texts):
        """Predict fraud probability"""
        if self.classifier is None:
            raise ValueError("Model not trained yet!")
        
        X = self.extract_combined_features(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        return predictions, probabilities

    def analyze_message(self, text: str):
        """Analyze a single message"""
        behavioral_features = self.extract_behavioral_features(text)
        predictions, probabilities = self.predict([text])
        
        prediction = predictions[0]
        fraud_prob = probabilities[0][1] if len(probabilities[0]) > 1 else 0
        
        # Risk assessment
        risk_factors = []
        if behavioral_features['urgency_score'] > 0:
            risk_factors.append(f"Urgency language detected ({behavioral_features['urgency_score']} matches)")
        if behavioral_features['fear_score'] > 0:
            risk_factors.append(f"Fear/threat language ({behavioral_features['fear_score']} matches)")
        if behavioral_features['authority_score'] > 0:
            risk_factors.append(f"Authority mimicking ({behavioral_features['authority_score']} matches)")
        if behavioral_features['has_url'] == 1:
            risk_factors.append("Contains URL/link")
        if behavioral_features['uppercase_ratio'] > 0.3:
            risk_factors.append(f"Excessive capitals ({behavioral_features['uppercase_ratio']*100:.1f}%)")
        
        return {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': fraud_prob,
            'risk_factors': risk_factors,
            'behavioral_features': behavioral_features
        }

    def save_model(self, path: str):
        """Save the trained model"""
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

def load_and_prepare_data():
    """Load and combine datasets"""
    print("🔍 Loading SMS datasets...")
    
    # 1. Load Kaggle spam dataset
    kaggle_path = "../../datasetgenerateor/sms data set/sms_spam.csv"
    kaggle_df = pd.read_csv(kaggle_path)
    
    print(f"Kaggle dataset: {len(kaggle_df)} messages")
    print(f"  - Ham (legitimate): {len(kaggle_df[kaggle_df.label=='ham'])}")
    print(f"  - Spam: {len(kaggle_df[kaggle_df.label=='spam'])}")
    
    # 2. Load real SMS data (sample for legitimate examples)
    real_sms_path = "../../datasetgenerateor/sms data set/phone_sms_export_2025-07-14T09-30-54.278524.csv"
    real_df = pd.read_csv(real_sms_path)
    
    # Filter service messages (legitimate)
    service_messages = real_df[
        (real_df['address'].str.contains(r'^[A-Z]{2,6}-', na=False)) |
        (real_df['address'].str.len() <= 8) |
        (real_df['address'].str.contains(r'^[A-Z]+$', na=False))
    ].copy()
    
    # Take a sample to balance the dataset
    service_sample = service_messages.sample(n=min(1000, len(service_messages)), random_state=42)
    service_sample['label'] = 'ham'
    service_sample['text'] = service_sample['body']
    
    print(f"Real SMS sample: {len(service_sample)} service messages")
    
    # 3. Combine datasets
    combined_df = pd.concat([
        kaggle_df[['text', 'label']],
        service_sample[['text', 'label']]
    ], ignore_index=True)
    
    # Clean data
    combined_df['text'] = combined_df['text'].astype(str)
    combined_df = combined_df[combined_df['text'].str.len() > 10]
    combined_df = combined_df.drop_duplicates(subset=['text'])
    
    # Convert labels
    combined_df['binary_label'] = combined_df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"\n📊 FINAL DATASET:")
    print(f"Total messages: {len(combined_df)}")
    print(f"Legitimate: {len(combined_df[combined_df.binary_label==0])}")
    print(f"Spam/Fraud: {len(combined_df[combined_df.binary_label==1])}")
    
    return combined_df

def test_with_real_examples(detector):
    """Test with real-world examples"""
    print(f"\n🧪 TESTING WITH REAL EXAMPLES")
    print("=" * 50)
    
    test_messages = [
        # Your actual SMS examples
        "Rs.2.00 spent via Kotak Debit Card XX5673 at AMAZONAWSESC on 12/07/2025. Avl bal Rs.15566.11 Not you?Tap https://kotak.com/KBANKT/Fraud",
        "Your Apple Account code is: 490687. Do not share it with anyone.",
        "BILLS DUE ? Get EASY credit of Rs.1,03,000 at Rs.4849/month with QUICK disbursal! Apply now: a1.Freo.in/x0WKK0l T&C -Freo",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
        "Your account has been suspended. Click here immediately to verify: http://fake-bank.com"
    ]
    
    expected = ["LEGITIMATE", "LEGITIMATE", "FRAUD", "FRAUD", "FRAUD"]
    
    print("Analysis Results:")
    print("-" * 80)
    
    correct = 0
    for i, (message, exp) in enumerate(zip(test_messages, expected)):
        analysis = detector.analyze_message(message)
        pred = analysis['prediction']
        conf = analysis['fraud_probability']
        
        is_correct = pred == exp
        correct += is_correct
        
        print(f"\nMessage {i+1}: {message[:60]}...")
        print(f"Expected: {exp} | Predicted: {pred} | {'✓' if is_correct else '✗'}")
        print(f"Confidence: {conf:.2f}")
        if analysis['risk_factors']:
            print(f"Risk Factors: {'; '.join(analysis['risk_factors'])}")
        print("-" * 40)
    
    accuracy = correct / len(test_messages)
    print(f"\nReal-world Test Accuracy: {accuracy:.1%} ({correct}/{len(test_messages)})")
    return accuracy

def main():
    """Main training function"""
    print("🚀 SIMPLIFIED ENHANCED SMS FRAUD DETECTION")
    print("=" * 60)
    
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Prepare training data
        texts = df['text'].tolist()
        labels = df['binary_label'].tolist()
        
        # Train model
        detector = SimplifiedBehavioralDetector()
        print(f"\n🎯 TRAINING MODEL...")
        results = detector.train(texts, labels)
        
        # Test with real examples
        real_accuracy = test_with_real_examples(detector)
        
        # Save model
        model_path = "simplified_behavioral_model.pkl"
        detector.save_model(model_path)
        
        # Summary
        print(f"\n🎉 TRAINING COMPLETE!")
        print("=" * 50)
        print(f"📊 Training Accuracy: {results['accuracy']:.1%}")
        print(f"🌍 Real-world Test: {real_accuracy:.1%}")
        print(f"💾 Model saved: {model_path}")
        print(f"📈 Training messages: {len(df)}")
        print(f"🔧 Features: TF-IDF + Behavioral patterns")
        
        return detector
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
