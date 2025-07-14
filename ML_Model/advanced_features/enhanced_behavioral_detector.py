#!/usr/bin/env python3
"""
Enhanced Behavioral SMS Fraud Detector
Improved pattern recognition and classification logic
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

class EnhancedBehavioralDetector:
    def __init__(self):
        """Initialize enhanced behavioral detector with comprehensive patterns"""
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1,2))
        self.classifier = None
        
        # Enhanced behavioral patterns
        self.urgency_patterns = [
            r'\b(urgent|immediate|now|asap|quickly|hurry|rush|fast)\b',
            r'\b(expire|expires|expiring|deadline|limited time|last chance)\b',
            r'\b(act now|call now|click now|verify now|update now|claim now)\b',
            r'\b(today|tomorrow|within \d+|before \d+|expires? (today|soon))\b'
        ]
        
        self.fear_patterns = [
            r'\b(suspended|blocked|closed|terminated|disabled|cancelled)\b',
            r'\b(unauthorized|suspicious|fraud|security|breach|violation)\b',
            r'\b(warning|alert|notice|penalty|fine|legal action)\b',
            r'\b(will be (blocked|closed|suspended|terminated))\b'
        ]
        
        self.reward_patterns = [
            r'\b(win|won|winner|prize|reward|gift|free|bonus|jackpot)\b',
            r'\b(congratulations|selected|chosen|lucky|special offer)\b',
            r'\b(cashback|refund|money|cash|₹|rupees|lakh|crore|\$|£)\b',
            r'\b(\d+%?\s?(off|discount|cashback|bonus))\b'
        ]
        
        self.authority_patterns = [
            r'\b(bank|government|tax|income tax|rbi|sebi|police|court)\b',
            r'\b(official|authorized|verified|certified|legal|ministry)\b',
            r'\b(department|bureau|agency|authority|administration)\b'
        ]
        
        self.action_patterns = [
            r'\b(click|tap|call|reply|send|forward|share|download)\b',
            r'\b(verify|confirm|update|activate|register|submit)\b',
            r'\b(provide|enter|give|send|share|disclose|reveal)\b'
        ]
        
        # Legitimate service patterns (to reduce false positives)
        self.legitimate_patterns = [
            r'\b(otp|verification code|security code)\b.*\bdo not share\b',
            r'\b(spent|credited|debited|payment|transaction)\b.*\b(rs\.?|₹)\d+',
            r'\b(avl\.?\s?bal|available balance|current balance)\b',
            r'-(hdfc|sbi|icici|axis|kotak|canara|pnb|bob|ubi|sbin)\b',
            r'\b(apple|google|microsoft|amazon|netflix)\s(account|code|otp)\b'
        ]

    def extract_behavioral_features(self, text: str) -> dict:
        """Extract comprehensive behavioral features"""
        text_lower = text.lower()
        features = {}
        
        # Pattern-based scores (weighted)
        features['urgency_score'] = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.urgency_patterns) * 2
        features['fear_score'] = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.fear_patterns) * 3
        features['reward_score'] = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.reward_patterns) * 2
        features['authority_score'] = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.authority_patterns) * 2
        features['action_score'] = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.action_patterns)
        
        # Legitimacy indicators (protective)
        features['legitimate_score'] = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in self.legitimate_patterns) * -3
        
        # Structural features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Communication features
        features['has_url'] = 2 if re.search(r'http[s]?://|www\.|\.[a-z]{2,3}/', text_lower) else 0
        features['has_phone'] = 1 if re.search(r'\+?\d{10,}', text) else 0
        features['all_caps_words'] = len([word for word in text.split() if word.isupper() and len(word) > 2])
        
        # Fraud-specific indicators
        features['money_amount'] = len(re.findall(r'(rs\.?|₹|\$|£)\s*\d+', text_lower))
        features['percentage_offers'] = len(re.findall(r'\d+%\s*(off|discount|cashback)', text_lower))
        features['suspicious_links'] = len(re.findall(r'(bit\.ly|tinyurl|goo\.gl|short\.link)', text_lower))
        
        # Calculate composite risk score
        features['risk_score'] = (
            features['urgency_score'] + 
            features['fear_score'] + 
            features['reward_score'] + 
            features['authority_score'] + 
            features['action_score'] +
            features['has_url'] +
            features['legitimate_score']  # This can be negative
        )
        
        return features

    def extract_combined_features(self, texts):
        """Combine TF-IDF with enhanced behavioral features"""
        # TF-IDF features
        tfidf_features = self.vectorizer.transform(texts).toarray()
        
        # Behavioral features
        behavioral_features = []
        for text in texts:
            features = self.extract_behavioral_features(text)
            behavioral_features.append(list(features.values()))
        
        behavioral_features = np.array(behavioral_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, behavioral_features])
        return combined_features

    def train(self, texts, labels):
        """Train enhanced model"""
        print("Extracting TF-IDF features...")
        self.vectorizer.fit(texts)
        
        print("Extracting enhanced behavioral features...")
        X = self.extract_combined_features(texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train enhanced classifier
        print("Training enhanced Random Forest...")
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',  # Handle imbalanced data
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nEnhanced Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {'accuracy': accuracy}

    def predict_with_reasoning(self, text: str):
        """Predict with detailed reasoning"""
        behavioral_features = self.extract_behavioral_features(text)
        predictions, probabilities = self.predict([text])
        
        prediction = predictions[0]
        fraud_prob = probabilities[0][1] if len(probabilities[0]) > 1 else 0
        
        # Enhanced risk assessment with reasoning
        risk_factors = []
        reasoning = []
        
        if behavioral_features['urgency_score'] > 0:
            risk_factors.append(f"Urgency manipulation ({behavioral_features['urgency_score']} indicators)")
            reasoning.append("Uses time pressure tactics")
        
        if behavioral_features['fear_score'] > 0:
            risk_factors.append(f"Fear/threat language ({behavioral_features['fear_score']} indicators)")
            reasoning.append("Contains intimidating or threatening language")
        
        if behavioral_features['reward_score'] > 0:
            risk_factors.append(f"Reward manipulation ({behavioral_features['reward_score']} indicators)")
            reasoning.append("Promises unrealistic rewards or prizes")
        
        if behavioral_features['authority_score'] > 0:
            risk_factors.append(f"Authority impersonation ({behavioral_features['authority_score']} indicators)")
            reasoning.append("Claims to be from authoritative source")
        
        if behavioral_features['has_url'] > 0:
            risk_factors.append("Contains suspicious links")
            reasoning.append("Includes external links requiring action")
        
        if behavioral_features['legitimate_score'] < 0:
            reasoning.append("Shows legitimate service patterns")
        
        if behavioral_features['uppercase_ratio'] > 0.3:
            risk_factors.append(f"Excessive capitals ({behavioral_features['uppercase_ratio']*100:.1f}%)")
            reasoning.append("Uses excessive capitalization for emphasis")
        
        # Enhanced classification logic
        risk_score = behavioral_features['risk_score']
        
        # Override based on behavioral analysis
        if risk_score >= 5 and behavioral_features['legitimate_score'] >= -2:
            final_prediction = 'FRAUD'
            confidence = max(fraud_prob, 0.8)
        elif behavioral_features['legitimate_score'] < -3:
            final_prediction = 'LEGITIMATE'
            confidence = min(fraud_prob, 0.3)
        else:
            final_prediction = 'FRAUD' if prediction == 1 else 'LEGITIMATE'
            confidence = fraud_prob
        
        return {
            'prediction': final_prediction,
            'fraud_probability': confidence,
            'risk_factors': risk_factors,
            'reasoning': reasoning,
            'risk_score': risk_score,
            'behavioral_features': behavioral_features
        }

    def predict(self, texts):
        """Standard prediction method"""
        if self.classifier is None:
            raise ValueError("Model not trained yet!")
        
        X = self.extract_combined_features(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        return predictions, probabilities

    def save_model(self, path: str):
        """Save the enhanced model"""
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer
        }
        joblib.dump(model_data, path)
        print(f"Enhanced model saved to {path}")

def test_enhanced_detector():
    """Test the enhanced detector with challenging examples"""
    print("🧪 TESTING ENHANCED BEHAVIORAL DETECTOR")
    print("=" * 60)
    
    # Load and prepare data (simplified for testing)
    kaggle_df = pd.read_csv("../../datasetgenerateor/sms data set/sms_spam.csv")
    real_df = pd.read_csv("../../datasetgenerateor/sms data set/phone_sms_export_2025-07-14T09-30-54.278524.csv")
    
    # Quick dataset preparation
    service_sample = real_df[real_df['address'].str.contains(r'^[A-Z]{2,6}-', na=False)].head(500)
    service_sample['label'] = 'ham'
    service_sample['text'] = service_sample['body']
    
    combined_df = pd.concat([
        kaggle_df[['text', 'label']].head(2000),
        service_sample[['text', 'label']]
    ], ignore_index=True)
    
    combined_df['binary_label'] = combined_df['label'].map({'ham': 0, 'spam': 1})
    
    # Train enhanced detector
    detector = EnhancedBehavioralDetector()
    texts = combined_df['text'].tolist()
    labels = combined_df['binary_label'].tolist()
    
    detector.train(texts, labels)
    
    # Test with challenging real examples
    test_cases = [
        ("Rs.2.00 spent via Kotak Debit Card XX5673 at AMAZONAWSESC on 12/07/2025. Avl bal Rs.15566.11 Not you?Tap https://kotak.com/KBANKT/Fraud", "LEGITIMATE"),
        ("Your Apple Account code is: 490687. Do not share it with anyone.", "LEGITIMATE"),
        ("BILLS DUE ? Get EASY credit of Rs.1,03,000 at Rs.4849/month with QUICK disbursal! Apply now: a1.Freo.in/x0WKK0l T&C -Freo", "FRAUD"),
        ("URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010", "FRAUD"),
        ("Your account has been suspended. Click here immediately to verify: http://fake-bank.com", "FRAUD"),
        ("FINAL NOTICE: Account will be closed in 24 hours. Verify now or lose access forever!", "FRAUD"),
        ("Congratulations! You've WON ₹50,000! Call immediately to claim your prize before it expires!", "FRAUD"),
        ("Your UPI payment of ₹500 to John Doe was successful. Ref: 123456789", "LEGITIMATE")
    ]
    
    print("\n📊 DETAILED ANALYSIS RESULTS:")
    print("=" * 80)
    
    correct = 0
    for i, (message, expected) in enumerate(test_cases):
        analysis = detector.predict_with_reasoning(message)
        prediction = analysis['prediction']
        confidence = analysis['fraud_probability']
        
        is_correct = prediction == expected
        correct += is_correct
        
        print(f"\nTest {i+1}: {message[:70]}...")
        print(f"Expected: {expected} | Predicted: {prediction} | {'✅' if is_correct else '❌'}")
        print(f"Confidence: {confidence:.2f} | Risk Score: {analysis['risk_score']}")
        
        if analysis['reasoning']:
            print(f"Reasoning: {'; '.join(analysis['reasoning'])}")
        
        if analysis['risk_factors']:
            print(f"Risk Factors: {'; '.join(analysis['risk_factors'][:3])}")
        
        print("-" * 80)
    
    accuracy = correct / len(test_cases)
    print(f"\n🎯 ENHANCED DETECTION ACCURACY: {accuracy:.1%} ({correct}/{len(test_cases)})")
    
    # Save enhanced model
    detector.save_model("enhanced_behavioral_model.pkl")
    
    return detector, accuracy

if __name__ == "__main__":
    test_enhanced_detector()
