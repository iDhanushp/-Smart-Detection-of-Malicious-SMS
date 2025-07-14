#!/usr/bin/env python3
"""
3-Class SMS Detection: LEGITIMATE, SPAM, FRAUD
Enhanced behavioral analysis with proper spam classification
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

class ThreeClassSMSDetector:
    def __init__(self):
        """Initialize 3-class SMS detector"""
        self.vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1,2))
        self.classifier = None
        
        # Enhanced behavioral patterns for 3-class classification
        
        # HIGH-RISK FRAUD patterns (account threats, government impersonation)
        self.fraud_patterns = [
            r'\b(account.*suspended|account.*blocked|account.*closed)\b',
            r'\b(government|tax|income tax|police|court|legal action)\b',
            r'\b(verify.*immediately|update.*now|confirm.*identity)\b',
            r'\b(click.*here|click.*link|tap.*link)\b.*\b(verify|confirm|update)\b'
        ]
        
        # SPAM patterns (promotional, marketing, prizes)
        self.spam_patterns = [
            r'\b(win|won|winner|prize|congratulations|selected)\b',
            r'\b(free|offer|deal|discount|cashback|bonus)\b',
            r'\b(limited.*time|last.*chance|expires.*today)\b',
            r'\b(call.*now|sms.*now|reply.*now)\b'
        ]
        
        # LEGITIMATE service patterns (banks, OTPs, transactions)
        self.legitimate_patterns = [
            r'\b(otp|verification.*code|security.*code)\b.*\bdo.*not.*share\b',
            r'\b(spent|credited|debited|payment|transaction)\b.*\b(rs\.?|₹)\d+',
            r'\b(avl\.?\s?bal|available.*balance|current.*balance)\b',
            r'-(hdfc|sbi|icici|axis|kotak|canara|pnb|bob|ubi|sbin|airtel|jio|vodafone)\b',
            r'\b(apple|google|microsoft|amazon|netflix|uber|ola|zomato|swiggy)\b'
        ]

    def extract_behavioral_features(self, text: str) -> dict:
        """Extract 3-class behavioral features"""
        text_lower = text.lower()
        features = {}
        
        # FRAUD indicators (high-risk account threats)
        features['fraud_account_threat'] = len(re.findall(r'\b(suspended|blocked|closed|terminated)\b.*\b(account|service)\b', text_lower))
        features['fraud_authority'] = len(re.findall(r'\b(government|tax|police|court|legal)\b', text_lower))
        features['fraud_urgency'] = len(re.findall(r'\b(immediate|urgent|now|asap)\b.*\b(verify|update|confirm)\b', text_lower))
        features['fraud_phishing'] = len(re.findall(r'\b(click|tap).*\b(here|link|url)\b', text_lower))
        
        # SPAM indicators (promotional, marketing)
        features['spam_prizes'] = len(re.findall(r'\b(win|won|winner|prize|lottery|jackpot)\b', text_lower))
        features['spam_offers'] = len(re.findall(r'\b(free|offer|deal|discount|cashback|bonus)\b', text_lower))
        features['spam_pressure'] = len(re.findall(r'\b(limited.*time|last.*chance|expires.*today|act.*now)\b', text_lower))
        features['spam_marketing'] = len(re.findall(r'\b(call.*now|sms.*now|reply.*stop|unsubscribe)\b', text_lower))
        
        # LEGITIMATE indicators (banks, services, OTPs)
        features['legit_otp'] = len(re.findall(r'\b(otp|verification.*code|security.*code)\b', text_lower))
        features['legit_transaction'] = len(re.findall(r'\b(spent|credited|debited|payment|successful)\b', text_lower))
        features['legit_bank_code'] = len(re.findall(r'-(hdfc|sbi|icici|axis|kotak|canara|airtel|jio)\b', text_lower))
        features['legit_service'] = len(re.findall(r'\b(apple|google|amazon|uber|zomato|delivery)\b', text_lower))
        features['legit_warning'] = len(re.findall(r'\bdo.*not.*share\b', text_lower))
        
        # Structural features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['has_url'] = 1 if re.search(r'http[s]?://|www\.|\.[a-z]{2,3}/', text_lower) else 0
        features['has_phone'] = 1 if re.search(r'\+?\d{10,}', text) else 0
        features['money_amount'] = len(re.findall(r'(rs\.?|₹|\$|£)\s*\d+', text_lower))
        
        # Calculate class-specific scores
        features['fraud_score'] = (
            features['fraud_account_threat'] * 3 +
            features['fraud_authority'] * 2 +
            features['fraud_urgency'] * 2 +
            features['fraud_phishing'] * 2
        )
        
        features['spam_score'] = (
            features['spam_prizes'] * 2 +
            features['spam_offers'] * 1 +
            features['spam_pressure'] * 2 +
            features['spam_marketing'] * 1
        )
        
        features['legit_score'] = (
            features['legit_otp'] * 3 +
            features['legit_transaction'] * 2 +
            features['legit_bank_code'] * 3 +
            features['legit_service'] * 1 +
            features['legit_warning'] * 2
        )
        
        return features

    def extract_combined_features(self, texts):
        """Combine TF-IDF with 3-class behavioral features"""
        tfidf_features = self.vectorizer.transform(texts).toarray()
        
        behavioral_features = []
        for text in texts:
            features = self.extract_behavioral_features(text)
            behavioral_features.append(list(features.values()))
        
        behavioral_features = np.array(behavioral_features)
        combined_features = np.hstack([tfidf_features, behavioral_features])
        return combined_features

    def prepare_3class_data(self):
        """Prepare 3-class dataset from available data"""
        print("📊 PREPARING 3-CLASS DATASET")
        
        # Load Kaggle dataset
        kaggle_df = pd.read_csv("../../datasetgenerateor/sms data set/sms_spam.csv")
        
        # Load real SMS data
        real_df = pd.read_csv("../../datasetgenerateor/sms data set/phone_sms_export_2025-07-14T09-30-54.278524.csv")
        
        # Separate kaggle data
        ham_messages = kaggle_df[kaggle_df['label'] == 'ham']['text'].tolist()
        spam_messages = kaggle_df[kaggle_df['label'] == 'spam']['text'].tolist()
        
        # Get legitimate service messages from real data
        service_messages = real_df[
            real_df['address'].str.contains(r'^[A-Z]{2,6}-', na=False)
        ]['body'].head(800).tolist()
        
        # Create enhanced 3-class dataset
        texts = []
        labels = []
        
        # LEGITIMATE class (0) - Ham + Service messages
        legitimate_texts = ham_messages[:1500] + service_messages
        texts.extend(legitimate_texts)
        labels.extend([0] * len(legitimate_texts))
        
        # SPAM class (1) - Promotional spam from Kaggle
        texts.extend(spam_messages)
        labels.extend([1] * len(spam_messages))
        
        # FRAUD class (2) - Create fraud examples from spam that match fraud patterns
        fraud_texts = []
        for msg in spam_messages:
            # Convert some spam to fraud by identifying high-risk patterns
            msg_lower = msg.lower()
            if any(pattern in msg_lower for pattern in ['account', 'suspend', 'verify', 'urgent', 'click']):
                # Transform spam to fraud-like message
                fraud_msg = msg.replace('Free', 'URGENT').replace('Win', 'Account suspended')
                fraud_texts.append(fraud_msg)
                if len(fraud_texts) >= 200:
                    break
        
        # Add some manually created fraud examples
        manual_fraud = [
            "URGENT: Your account has been suspended due to suspicious activity. Click here to verify immediately: fake-bank.com",
            "Government Tax Notice: Your PAN is blocked. Update details within 24 hours or face legal action.",
            "Bank Security Alert: Unauthorized access detected. Verify your identity now: malicious-link.com",
            "Police Notice: Your number is involved in cyber crime. Report immediately or face arrest.",
            "Income Tax Department: Refund pending. Verify details to avoid penalty: fake-tax.gov",
            "Your account will be closed permanently. Click to prevent closure: phishing-site.com",
            "Security breach detected. Update password immediately: fake-security.com",
            "Court summons issued. Verify identity to avoid legal action: fake-court.com"
        ] * 30  # Repeat to get more fraud examples
        
        fraud_texts.extend(manual_fraud)
        texts.extend(fraud_texts)
        labels.extend([2] * len(fraud_texts))
        
        print(f"Dataset created:")
        print(f"  LEGITIMATE: {labels.count(0)} messages")
        print(f"  SPAM: {labels.count(1)} messages") 
        print(f"  FRAUD: {labels.count(2)} messages")
        print(f"  Total: {len(texts)} messages")
        
        return texts, labels

    def train_3class(self):
        """Train 3-class classifier"""
        texts, labels = self.prepare_3class_data()
        
        print("\n🎯 TRAINING 3-CLASS MODEL...")
        self.vectorizer.fit(texts)
        X = self.extract_combined_features(texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n3-Class Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['LEGITIMATE', 'SPAM', 'FRAUD']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"       LEGIT  SPAM  FRAUD")
        for i, row_label in enumerate(['LEGIT', 'SPAM ', 'FRAUD']):
            print(f"{row_label}: {cm[i]}")
        
        return accuracy

    def predict_3class(self, text: str):
        """Predict with 3-class logic and detailed reasoning"""
        if self.classifier is None:
            raise ValueError("Model not trained yet!")
        
        behavioral_features = self.extract_behavioral_features(text)
        
        # Get model prediction
        X = self.extract_combined_features([text])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Enhanced classification logic using behavioral scores
        fraud_score = behavioral_features['fraud_score']
        spam_score = behavioral_features['spam_score']
        legit_score = behavioral_features['legit_score']
        
        # Override prediction based on behavioral analysis
        if legit_score >= 3:
            final_class = 'LEGITIMATE'
            confidence = max(probabilities[0], 0.7)
        elif fraud_score >= 4:
            final_class = 'FRAUD'
            confidence = max(probabilities[2], 0.8)
        elif spam_score >= 3:
            final_class = 'SPAM'
            confidence = max(probabilities[1], 0.7)
        else:
            # Use model prediction
            class_names = ['LEGITIMATE', 'SPAM', 'FRAUD']
            final_class = class_names[prediction]
            confidence = probabilities[prediction]
        
        # Generate reasoning
        reasoning = []
        risk_factors = []
        
        if behavioral_features['fraud_account_threat'] > 0:
            reasoning.append("Contains account suspension/blocking threats")
            risk_factors.append("Account threat language")
        
        if behavioral_features['fraud_authority'] > 0:
            reasoning.append("Impersonates government/legal authority")
            risk_factors.append("Authority impersonation")
        
        if behavioral_features['spam_prizes'] > 0:
            reasoning.append("Promises prizes or winnings")
            risk_factors.append("Prize/lottery scheme")
        
        if behavioral_features['spam_offers'] > 0:
            reasoning.append("Contains promotional offers")
            risk_factors.append("Marketing promotion")
        
        if behavioral_features['legit_otp'] > 0:
            reasoning.append("Contains legitimate OTP/verification code")
        
        if behavioral_features['legit_bank_code'] > 0:
            reasoning.append("From verified bank/service code")
        
        return {
            'prediction': final_class,
            'confidence': confidence,
            'probabilities': {
                'LEGITIMATE': probabilities[0],
                'SPAM': probabilities[1], 
                'FRAUD': probabilities[2]
            },
            'behavioral_scores': {
                'fraud_score': fraud_score,
                'spam_score': spam_score,
                'legit_score': legit_score
            },
            'reasoning': reasoning,
            'risk_factors': risk_factors
        }

    def save_model(self, path: str):
        """Save 3-class model"""
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer
        }
        joblib.dump(model_data, path)
        print(f"3-class model saved to {path}")

def test_3class_detector():
    """Test 3-class SMS detector"""
    print("🚀 TESTING 3-CLASS SMS DETECTOR")
    print("=" * 60)
    
    detector = ThreeClassSMSDetector()
    accuracy = detector.train_3class()
    
    # Test with comprehensive examples
    test_cases = [
        # LEGITIMATE examples
        ("Rs.2.00 spent via Kotak Debit Card XX5673 at AMAZONAWSESC on 12/07/2025. Avl bal Rs.15566.11 -KOTAKB", "LEGITIMATE"),
        ("Your Apple Account code is: 490687. Do not share it with anyone.", "LEGITIMATE"),
        ("Your UPI payment of ₹500 to John Doe was successful. Ref: 123456789", "LEGITIMATE"),
        
        # SPAM examples  
        ("Congratulations! You've WON ₹50,000! Call immediately to claim your prize before it expires!", "SPAM"),
        ("Limited time offer! Get 50% cashback on your next recharge. Offer valid today only!", "SPAM"),
        ("Free iPhone 14! You're selected for our special promotion. Reply YES to claim now!", "SPAM"),
        
        # FRAUD examples
        ("URGENT: Your account has been suspended. Click here immediately to verify: http://fake-bank.com", "FRAUD"),
        ("Government Tax Notice: Your PAN is disabled. Update within 24 hours or face legal action.", "FRAUD"),
        ("Bank Security Alert: Unauthorized access detected. Verify identity now: malicious-link.com", "FRAUD")
    ]
    
    print(f"\n📊 COMPREHENSIVE 3-CLASS ANALYSIS:")
    print("=" * 80)
    
    correct = 0
    class_results = {'LEGITIMATE': 0, 'SPAM': 0, 'FRAUD': 0}
    
    for i, (message, expected) in enumerate(test_cases):
        analysis = detector.predict_3class(message)
        prediction = analysis['prediction']
        confidence = analysis['confidence']
        
        is_correct = prediction == expected
        correct += is_correct
        class_results[prediction] += 1
        
        print(f"\nTest {i+1}: {message[:70]}...")
        print(f"Expected: {expected} | Predicted: {prediction} | {'✅' if is_correct else '❌'}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Probabilities: LEGIT={analysis['probabilities']['LEGITIMATE']:.2f}, "
              f"SPAM={analysis['probabilities']['SPAM']:.2f}, "
              f"FRAUD={analysis['probabilities']['FRAUD']:.2f}")
        
        if analysis['reasoning']:
            print(f"Reasoning: {'; '.join(analysis['reasoning'])}")
        
        print("-" * 80)
    
    accuracy_test = correct / len(test_cases)
    print(f"\n🎯 3-CLASS TEST ACCURACY: {accuracy_test:.1%} ({correct}/{len(test_cases)})")
    print(f"Classification Distribution: {class_results}")
    
    # Save model
    detector.save_model("three_class_sms_model.pkl")
    
    return detector

if __name__ == "__main__":
    test_3class_detector()
