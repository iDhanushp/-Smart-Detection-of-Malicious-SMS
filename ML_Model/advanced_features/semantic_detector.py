#!/usr/bin/env python3
"""
Advanced SMS Fraud Detection using Semantic Embeddings + Behavioral Analysis
Replaces keyword-only approach with SBERT + structural features
"""

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import List, Dict, Tuple
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SemanticFraudDetector:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize semantic fraud detector with behavioral analysis
        
        Args:
            model_name: SentenceTransformer model for embeddings
        """
        self.sentence_model = SentenceTransformer(model_name)
        self.classifier = None
        self.feature_scaler = None
        
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
    
    def extract_behavioral_features(self, text: str) -> Dict[str, float]:
        """Extract behavioral and structural features from text"""
        text_lower = text.lower()
        features = {}
        
        # 1. Urgency analysis
        urgency_score = sum(len(re.findall(pattern, text_lower)) 
                           for pattern in self.urgency_patterns)
        features['urgency_score'] = urgency_score / len(text.split())  # Normalized
        
        # 2. Fear/threat analysis
        fear_score = sum(len(re.findall(pattern, text_lower)) 
                        for pattern in self.fear_patterns)
        features['fear_score'] = fear_score / len(text.split())
        
        # 3. Reward/incentive analysis
        reward_score = sum(len(re.findall(pattern, text_lower)) 
                          for pattern in self.reward_patterns)
        features['reward_score'] = reward_score / len(text.split())
        
        # 4. Authority mimicking
        authority_score = sum(len(re.findall(pattern, text_lower)) 
                             for pattern in self.authority_patterns)
        features['authority_score'] = authority_score / len(text.split())
        
        # 5. Action requests
        action_score = sum(len(re.findall(pattern, text_lower)) 
                          for pattern in self.action_patterns)
        features['action_score'] = action_score / len(text.split())
        
        # 6. Structural features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
        features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # 7. Readability (complexity analysis)
        try:
            features['readability_score'] = flesch_reading_ease(text)
            features['grade_level'] = flesch_kincaid_grade(text)
        except:
            features['readability_score'] = 50.0  # Default
            features['grade_level'] = 8.0
        
        # 8. URL/Link detection
        features['has_url'] = 1 if re.search(r'http[s]?://|www\.|\.[a-z]{2,3}/', text_lower) else 0
        features['has_phone'] = 1 if re.search(r'\+?\d{10,}', text) else 0
        
        # 9. Punctuation patterns (spam often overuses punctuation)
        features['punctuation_density'] = len(re.findall(r'[!?.]{2,}', text)) / len(text.split())
        
        # 10. Time pressure indicators
        time_patterns = [r'\b(today|tomorrow|within \d+|in \d+ minutes|expires)\b']
        features['time_pressure'] = sum(len(re.findall(pattern, text_lower)) 
                                       for pattern in time_patterns) / len(text.split())
        
        return features
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings for text list"""
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def extract_combined_features(self, texts: List[str]) -> np.ndarray:
        """Combine semantic embeddings with behavioral features"""
        # Get semantic embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Extract behavioral features for each text
        behavioral_features = []
        for text in texts:
            features = self.extract_behavioral_features(text)
            behavioral_features.append(list(features.values()))
        
        behavioral_features = np.array(behavioral_features)
        
        # Combine embeddings with behavioral features
        combined_features = np.hstack([embeddings, behavioral_features])
        
        return combined_features
    
    def train(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Train the fraud detection model"""
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
        
        # Feature importance analysis
        feature_names = (
            [f'embed_{i}' for i in range(384)] +  # SBERT embeddings
            ['urgency_score', 'fear_score', 'reward_score', 'authority_score', 
             'action_score', 'char_count', 'word_count', 'avg_word_length',
             'uppercase_ratio', 'digit_ratio', 'special_char_ratio', 
             'exclamation_count', 'question_count', 'readability_score',
             'grade_level', 'has_url', 'has_phone', 'punctuation_density', 'time_pressure']
        )
        
        # Get top behavioral features
        behavioral_importance = self.classifier.feature_importances_[384:]  # Skip embeddings
        behavioral_names = feature_names[384:]
        
        top_features = sorted(zip(behavioral_names, behavioral_importance), 
                             key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\nTop 10 Behavioral Features:")
        for name, importance in top_features:
            print(f"{name}: {importance:.4f}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': dict(top_features)
        }
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fraud probability for texts"""
        if self.classifier is None:
            raise ValueError("Model not trained yet!")
        
        X = self.extract_combined_features(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        return predictions, probabilities
    
    def analyze_message(self, text: str) -> Dict[str, any]:
        """Detailed analysis of a single message"""
        behavioral_features = self.extract_behavioral_features(text)
        
        predictions, probabilities = self.predict([text])
        prediction = predictions[0]
        fraud_prob = probabilities[0][1] if len(probabilities[0]) > 1 else 0
        
        # Risk assessment
        risk_factors = []
        if behavioral_features['urgency_score'] > 0.1:
            risk_factors.append(f"High urgency language (score: {behavioral_features['urgency_score']:.2f})")
        if behavioral_features['fear_score'] > 0.1:
            risk_factors.append(f"Fear-inducing language (score: {behavioral_features['fear_score']:.2f})")
        if behavioral_features['authority_score'] > 0.1:
            risk_factors.append(f"Authority mimicking (score: {behavioral_features['authority_score']:.2f})")
        if behavioral_features['has_url'] == 1:
            risk_factors.append("Contains URL/link")
        if behavioral_features['uppercase_ratio'] > 0.3:
            risk_factors.append(f"Excessive capitals ({behavioral_features['uppercase_ratio']*100:.1f}%)")
        
        return {
            'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': fraud_prob,
            'risk_factors': risk_factors,
            'behavioral_features': behavioral_features,
            'urgency_level': 'HIGH' if behavioral_features['urgency_score'] > 0.15 else 
                           'MEDIUM' if behavioral_features['urgency_score'] > 0.05 else 'LOW',
            'fear_level': 'HIGH' if behavioral_features['fear_score'] > 0.15 else 
                        'MEDIUM' if behavioral_features['fear_score'] > 0.05 else 'LOW'
        }
    
    def save_model(self, path: str):
        """Save the trained model"""
        model_data = {
            'classifier': self.classifier,
            'sentence_model_name': self.sentence_model._modules['0'].auto_model.name_or_path
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model"""
        model_data = joblib.load(path)
        self.classifier = model_data['classifier']
        print(f"Model loaded from {path}")

def main():
    """Example usage and testing"""
    # Sample fraud messages for testing
    sample_messages = [
        "Your account has been suspended due to suspicious activity. Click here to verify: http://fake-bank.com",
        "Congratulations! You've won ₹50,000! Call 9999999999 immediately to claim your prize!",
        "URGENT: Your OTP is 123456. Do not share with anyone. -SBI",
        "Hi, can we meet for coffee tomorrow?",
        "Your order #12345 has been delivered. Thank you for shopping with us!",
        "FINAL NOTICE: Your account will be closed in 24 hours. Verify now or lose access forever!",
        "Free iPhone 14! Limited time offer! Click now: www.freephone.com",
        "Your UPI payment of ₹500 to John Doe was successful. Ref: 123456789"
    ]
    
    # Manual labels for testing (1=fraud, 0=legitimate)
    sample_labels = [1, 1, 0, 0, 0, 1, 1, 0]
    
    # Initialize and train detector
    detector = SemanticFraudDetector()
    
    print("Training semantic fraud detector...")
    results = detector.train(sample_messages, sample_labels)
    
    print(f"\n{'='*50}")
    print("DETAILED MESSAGE ANALYSIS")
    print(f"{'='*50}")
    
    # Analyze each message
    for i, message in enumerate(sample_messages):
        print(f"\nMessage {i+1}: {message}")
        analysis = detector.analyze_message(message)
        print(f"Prediction: {analysis['prediction']}")
        print(f"Fraud Probability: {analysis['fraud_probability']:.2f}")
        print(f"Urgency Level: {analysis['urgency_level']}")
        print(f"Fear Level: {analysis['fear_level']}")
        if analysis['risk_factors']:
            print(f"Risk Factors: {', '.join(analysis['risk_factors'])}")
        print("-" * 50)

if __name__ == "__main__":
    main()
