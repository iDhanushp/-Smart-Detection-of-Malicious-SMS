#!/usr/bin/env python3
"""
Export Enhanced Behavioral SMS Detector to TensorFlow Lite
Creates a production-ready 3-class TFLite model with behavioral features
"""

import pandas as pd
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    try:
        import tensorflow.compat.v2 as tf
        tf.enable_v2_behavior()
        from tensorflow.compat.v2 import keras
    except ImportError:
        # Fallback for older TF versions
        import tensorflow as tf
        import tensorflow.keras as keras
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from enhanced_behavioral_labeler import BehavioralSMSLabeler

class BehavioralModelTrainer:
    def __init__(self):
        self.labeler = BehavioralSMSLabeler()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_training_data(self, num_samples=5000):
        """Generate comprehensive training data with behavioral features"""
        print("🏗️ Generating comprehensive training dataset...")
        
        # Fraud examples
        fraud_messages = [
            ("URGENT: Your account has been SUSPENDED! Update now or lose access forever!", "+919876543210"),
            ("Security Alert: Verify your OTP and PIN immediately to prevent account closure", "+447123456789"),
            ("Income Tax Notice: Update PAN within 24 hours or face legal action", "INCOMETAX"),
            ("Your credit card is BLOCKED due to suspicious activity. Confirm details now", "+1234567890"),
            ("Bank Alert: Unauthorized transaction detected. Provide CVV to secure account", "HDFC"),
            ("Government Notice: Your Aadhaar is deactivated. Update immediately or penalty", "UIDAI"),
            ("Final Warning: Account will be terminated in 2 hours. Verify now!", "+919988776655"),
            ("Emergency: Your debit card is compromised. Share OTP to block fraudulent use", "SBI"),
        ]
        
        # Spam examples  
        spam_messages = [
            ("🎉 CONGRATULATIONS! You won ₹50,000! Claim your prize now!", "WINNER"),
            ("MEGA SALE! 70% OFF everything! Limited time offer! Shop now!", "DEALS"),
            ("Earn ₹5000 daily from home! No investment! Guaranteed income!", "+919123456789"),
            ("Free iPhone! You're selected for special promotion! Click to claim!", "PROMO"),
            ("Hot singles in your area! Chat now for free! Premium membership available!", "DATING"),
            ("Lose 10kg in 10 days! Revolutionary weight loss supplement! Order now!", "HEALTH"),
            ("Make money online! Join our network marketing! Unlimited earning potential!", "+918765432109"),
            ("Luxury cruise worth ₹2 lakhs! You're our lucky winner! Book now!", "TRAVEL"),
        ]
        
        # Legitimate examples
        legit_messages = [
            ("Your OTP for SBI login is 123456. Do not share. Valid for 10 minutes.", "AD-SBIINB"),
            ("UPI payment of ₹1500 to Swiggy successful. Txn ID: 123456789012", "AX-PAYTM"),
            ("Your Zomato order is out for delivery. Track: zomato.com/track", "ZM-ZOMATO"),
            ("Hi, are we meeting at 6 PM today? Let me know if reschedule needed.", "+919876543211"),
            ("Electricity bill for March: ₹2345. Due: 31st March. Pay online.", "BESCOM"),
            ("Appointment reminder: Dr. Smith tomorrow 3 PM. Arrive 15 mins early.", "CLINIC"),
            ("Your train PNR 1234567890 confirmed. Journey: Delhi to Mumbai tomorrow.", "IRCTC"),
            ("Account balance: ₹15,430. Last transaction: ₹500 debit at ATM.", "AX-HDFC"),
        ]
        
        # Generate expanded dataset
        all_data = []
        
        # Add base examples with variations
        for messages, label_idx in [(fraud_messages, 2), (spam_messages, 1), (legit_messages, 0)]:
            for msg, sender in messages:
                # Original message
                features = self.extract_features(msg, sender)
                all_data.append({**features, 'label': label_idx, 'text': msg, 'sender': sender})
                
                # Generate variations for better generalization
                for i in range(num_samples // (len(fraud_messages) + len(spam_messages) + len(legit_messages)) - 1):
                    if label_idx == 2:  # Fraud - add urgent variations
                        varied_msg = self.add_fraud_variations(msg)
                    elif label_idx == 1:  # Spam - add promotional variations  
                        varied_msg = self.add_spam_variations(msg)
                    else:  # Legit - add service variations
                        varied_msg = self.add_legit_variations(msg)
                    
                    features = self.extract_features(varied_msg, sender)
                    all_data.append({**features, 'label': label_idx, 'text': varied_msg, 'sender': sender})
        
        df = pd.DataFrame(all_data)
        print(f"✅ Generated {len(df)} training samples")
        print(f"📊 Class distribution: Legit: {(df['label']==0).sum()}, Spam: {(df['label']==1).sum()}, Fraud: {(df['label']==2).sum()}")
        
        return df
    
    def add_fraud_variations(self, msg):
        """Add variations to fraud messages for better training"""
        urgent_words = ['URGENT', 'IMMEDIATE', 'ASAP', 'CRITICAL', 'EMERGENCY']
        threat_words = ['suspended', 'blocked', 'terminated', 'disabled', 'frozen']
        
        import random
        # Randomly add urgency
        if random.random() > 0.5:
            msg = f"{random.choice(urgent_words)}: {msg}"
        
        # Sometimes replace threat words
        for word in threat_words:
            if word in msg.lower() and random.random() > 0.7:
                msg = msg.replace(word, random.choice(threat_words))
        
        return msg
    
    def add_spam_variations(self, msg):
        """Add variations to spam messages"""
        import random
        prize_amounts = ['₹10,000', '₹25,000', '₹50,000', '₹1,00,000']
        offer_words = ['MEGA', 'SUPER', 'ULTRA', 'AMAZING', 'INCREDIBLE']
        
        # Random prize amounts
        for amount in ['₹50,000', '₹25,000']:
            if amount in msg:
                msg = msg.replace(amount, random.choice(prize_amounts))
        
        # Random offer intensifiers
        if random.random() > 0.6:
            msg = f"{random.choice(offer_words)} {msg}"
        
        return msg
    
    def add_legit_variations(self, msg):
        """Add variations to legitimate messages"""
        import random
        # Simple variations for legitimate messages
        if 'OTP' in msg and random.random() > 0.5:
            msg = msg.replace('OTP', 'One Time Password')
        
        return msg
    
    def extract_features(self, text, sender):
        """Extract comprehensive behavioral and structural features"""
        # Get behavioral analysis
        classification, confidence, analysis = self.labeler.classify_message_advanced(text, sender)
        
        behavioral_signals = analysis['behavioral_signals']
        structural_features = analysis['structural_features']
        scores = analysis['scores']
        
        # Create comprehensive feature vector
        features = {
            # Behavioral patterns (normalized 0-1)
            'urgency_immediate': behavioral_signals.get('urgency_immediate', 0),
            'urgency_time_pressure': behavioral_signals.get('urgency_time_pressure', 0),
            'fear_account_threats': behavioral_signals.get('fear_account_threats', 0),
            'fear_loss_threats': behavioral_signals.get('fear_loss_threats', 0),
            'reward_money': behavioral_signals.get('reward_money', 0),
            'reward_prizes': behavioral_signals.get('reward_prizes', 0),
            'authority_financial': behavioral_signals.get('authority_financial', 0),
            'authority_government': behavioral_signals.get('authority_government', 0),
            'action_data_harvesting': behavioral_signals.get('action_data_harvesting', 0),
            'action_immediate': behavioral_signals.get('action_immediate', 0),
            
            # Aggregated scores
            'total_urgency': behavioral_signals.get('total_urgency', 0),
            'total_fear': behavioral_signals.get('total_fear', 0),
            'total_reward': behavioral_signals.get('total_reward', 0),
            'total_authority': behavioral_signals.get('total_authority', 0),
            'total_action': behavioral_signals.get('total_action', 0),
            
            # Structural features (normalized)
            'length_normalized': min(structural_features.get('length', 0) / 500, 1.0),
            'word_count_normalized': min(structural_features.get('word_count', 0) / 100, 1.0),
            'uppercase_ratio': structural_features.get('uppercase_ratio', 0),
            'digit_ratio': structural_features.get('digit_ratio', 0),
            'special_char_ratio': structural_features.get('special_char_ratio', 0),
            'exclamation_count_normalized': min(structural_features.get('exclamation_count', 0) / 5, 1.0),
            'caps_words_normalized': min(structural_features.get('caps_words', 0) / 10, 1.0),
            'has_url': structural_features.get('has_url', 0),
            'has_phone': structural_features.get('has_phone', 0),
            
            # Sender analysis
            'sender_is_phone': 1 if sender and sender.startswith('+') else 0,
            'sender_is_service': 1 if sender and (len(sender) <= 6 or '-' in sender) else 0,
            'sender_length_normalized': min(len(sender) / 20, 1.0) if sender else 0,
            
            # Model scores as features
            'fraud_score': scores.get('fraud', 0),
            'spam_score': scores.get('spam', 0),
            'legit_score': scores.get('legit', 0),
        }
        
        return features
    
    def train_model(self, df):
        """Train RandomForest model on behavioral features"""
        print("🤖 Training 3-class behavioral model...")
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in ['label', 'text', 'sender']]
        self.feature_names = feature_columns
        
        X = df[feature_columns].fillna(0)  # Handle any NaN values
        y = df['label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained with {accuracy:.3f} accuracy")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['LEGIT', 'SPAM', 'FRAUD']))
        
        return X_test, y_test
    
    def create_tf_model(self, X_test):
        """Create TensorFlow model that mimics the sklearn model"""
        print("🔄 Creating TensorFlow model for TFLite export...")
        
        input_shape = X_test.shape[1]
        
        # Create neural network that learns to mimic RandomForest
        tf_model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(3, activation='softmax')  # 3-class output
        ])
        
        tf_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return tf_model
    
    def train_tf_model(self, tf_model, X_test, y_test):
        """Train TF model to mimic sklearn predictions"""
        print("🧠 Training TensorFlow model to mimic behavioral classifier...")
        
        # Generate larger synthetic dataset using sklearn model
        n_synthetic = 10000
        X_synthetic = np.random.randn(n_synthetic, len(self.feature_names))
        X_synthetic = self.scaler.transform(X_synthetic)  # Scale synthetic data
        
        # Get sklearn predictions as labels for TF model
        y_synthetic = self.model.predict(X_synthetic)
        
        # Train TF model
        tf_model.fit(
            X_synthetic, y_synthetic,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return tf_model
    
    def export_tflite(self, tf_model):
        """Export model to TensorFlow Lite"""
        print("📱 Exporting to TensorFlow Lite...")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = "../sms_fraud_detectore_app/assets/advanced_fraud_detector.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved: {tflite_path}")
        print(f"📏 Model size: {len(tflite_model) / 1024:.1f} KB")
        
        return tflite_path
    
    def export_feature_config(self):
        """Export feature configuration for Flutter integration"""
        config = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'classes': ['LEGITIMATE', 'SPAM', 'FRAUD'],
            'model_version': '3.0.0',
            'export_date': pd.Timestamp.now().isoformat()
        }
        
        config_path = "../sms_fraud_detectore_app/assets/behavioral_model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Feature config saved: {config_path}")
        
        return config_path
    
    def save_models(self):
        """Save sklearn models for future use"""
        joblib.dump(self.model, 'advanced_behavioral_model.pkl')
        joblib.dump(self.scaler, 'behavioral_feature_scaler.pkl')
        print("✅ Sklearn models saved")

def main():
    """Main training and export pipeline"""
    print("🚀 ADVANCED BEHAVIORAL SMS DETECTOR - TFLITE EXPORT")
    print("=" * 80)
    
    trainer = BehavioralModelTrainer()
    
    # 1. Generate training data
    df = trainer.generate_training_data(num_samples=5000)
    
    # 2. Train behavioral model
    X_test, y_test = trainer.train_model(df)
    
    # 3. Create and train TensorFlow model
    tf_model = trainer.create_tf_model(X_test)
    tf_model = trainer.train_tf_model(tf_model, X_test, y_test)
    
    # 4. Export to TFLite
    tflite_path = trainer.export_tflite(tf_model)
    
    # 5. Export feature configuration
    config_path = trainer.export_feature_config()
    
    # 6. Save models
    trainer.save_models()
    
    print("\n🎉 EXPORT COMPLETE!")
    print(f"📱 TFLite model: {tflite_path}")
    print(f"⚙️ Feature config: {config_path}")
    print("\n✅ Ready for Flutter integration!")

if __name__ == "__main__":
    main()
