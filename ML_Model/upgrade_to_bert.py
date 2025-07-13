"""
DistilBERT Tiny Implementation for SMS Fraud Detection
Replaces TF-IDF + Naive Bayes with DistilBERT for better context understanding
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.quantization import quantize_dynamic
import numpy as np
import json
import os
from typing import List, Dict, Tuple
import tensorflow as tf

class DistilBERTFraudDetector:
    """
    Advanced SMS fraud detector using DistilBERT Tiny with quantization
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # Legitimate, Spam, Fraudulent
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess SMS text for DistilBERT input
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Tokenize with DistilBERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,  # DistilBERT max length
            return_tensors='pt'
        )
        
        return encoding
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize SMS text
        """
        import re
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove phone numbers (keep format for analysis)
        text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict fraud probability for SMS text
        """
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess text
            inputs = self.preprocess_text(text)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Convert to dictionary
            labels = ['legitimate', 'spam', 'fraudulent']
            predictions = {
                label: prob.item() 
                for label, prob in zip(labels, probabilities[0])
            }
            
        return predictions
    
    def quantize_model(self):
        """
        Quantize model to reduce size by ~75%
        """
        print("Quantizing DistilBERT model...")
        
        # Dynamic quantization
        self.model = quantize_dynamic(
            self.model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        print("Model quantization completed!")
        
    def export_tflite(self, output_path: str = "fraud_detector_bert.tflite"):
        """
        Export quantized model to TensorFlow Lite
        """
        print("Exporting to TensorFlow Lite...")
        
        # Convert PyTorch model to ONNX first
        onnx_path = "fraud_detector_bert.onnx"
        self._export_to_onnx(onnx_path)
        
        # Convert ONNX to TensorFlow Lite
        self._onnx_to_tflite(onnx_path, output_path)
        
        print(f"Model exported to {output_path}")
        
    def _export_to_onnx(self, onnx_path: str):
        """
        Export PyTorch model to ONNX format
        """
        import onnx
        import onnxruntime
        
        # Create dummy input
        dummy_input = self.tokenizer(
            "test message",
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
    def _onnx_to_tflite(self, onnx_path: str, tflite_path: str):
        """
        Convert ONNX model to TensorFlow Lite
        """
        import onnx2tf
        
        # Convert ONNX to TensorFlow
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path="./tf_model",
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=True
        )
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model("./tf_model")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
    def fine_tune(self, training_data: List[Tuple[str, int]], epochs: int = 3):
        """
        Fine-tune the model on SMS fraud detection data
        """
        print("Fine-tuning DistilBERT model...")
        
        # Prepare training data
        train_encodings = []
        train_labels = []
        
        for text, label in training_data:
            encoding = self.preprocess_text(text)
            train_encodings.append(encoding)
            train_labels.append(label)
        
        # Create dataset
        class SMSDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
                
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings[idx].items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
                
            def __len__(self):
                return len(self.labels)
        
        train_dataset = SMSDataset(train_encodings, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        print("Fine-tuning completed!")

class AdaptiveThresholdManager:
    """
    Dynamic thresholding system for adaptive fraud detection
    """
    
    def __init__(self):
        self.base_thresholds = {
            'fraud': 0.7,
            'spam': 0.6,
            'legitimate': 0.8
        }
        self.user_behavior_history = []
        self.false_positive_rate = 0.0
        self.missed_fraud_rate = 0.0
        
    def adjust_thresholds(self, user_feedback: Dict, message_patterns: Dict):
        """
        Dynamically adjust thresholds based on user feedback and patterns
        """
        # Update rates from feedback
        if 'false_positive_rate' in user_feedback:
            self.false_positive_rate = user_feedback['false_positive_rate']
            
        if 'missed_fraud_rate' in user_feedback:
            self.missed_fraud_rate = user_feedback['missed_fraud_rate']
        
        # Adjust thresholds based on performance
        if self.false_positive_rate > 0.1:
            # Increase thresholds to reduce false positives
            self.base_thresholds['fraud'] += 0.05
            self.base_thresholds['spam'] += 0.03
            print(f"Reducing false positives: fraud={self.base_thresholds['fraud']:.2f}, spam={self.base_thresholds['spam']:.2f}")
            
        if self.missed_fraud_rate > 0.05:
            # Decrease thresholds to catch more fraud
            self.base_thresholds['fraud'] -= 0.03
            print(f"Catching more fraud: fraud={self.base_thresholds['fraud']:.2f}")
            
        # Ensure thresholds stay within reasonable bounds
        self.base_thresholds['fraud'] = max(0.3, min(0.9, self.base_thresholds['fraud']))
        self.base_thresholds['spam'] = max(0.3, min(0.9, self.base_thresholds['spam']))
        self.base_thresholds['legitimate'] = max(0.5, min(0.95, self.base_thresholds['legitimate']))
        
    def get_adaptive_thresholds(self, sender_type: str, message_length: int) -> Dict[str, float]:
        """
        Get context-aware thresholds based on sender and message characteristics
        """
        base = self.base_thresholds.copy()
        
        # Adjust for sender type
        if sender_type == 'trusted_bank':
            base['fraud'] += 0.1  # Higher threshold for trusted senders
            base['spam'] += 0.05
        elif sender_type == 'unknown_number':
            base['fraud'] -= 0.05  # Lower threshold for unknown senders
            base['spam'] -= 0.03
        elif sender_type == 'alphanumeric':
            base['fraud'] += 0.05  # Slightly higher for alphanumeric senders
        elif sender_type == 'phone_number':
            base['fraud'] -= 0.03  # Lower for phone number senders
            
        # Adjust for message characteristics
        if message_length > 200:
            base['spam'] -= 0.02  # Longer messages more likely to be spam
        elif message_length < 20:
            base['fraud'] += 0.02  # Very short messages less likely to be fraud
            
        # Adjust for time patterns (if available)
        # This could be extended with time-based analysis
        
        return base
    
    def save_thresholds(self, filepath: str = "adaptive_thresholds.json"):
        """
        Save current thresholds to file
        """
        with open(filepath, 'w') as f:
            json.dump({
                'thresholds': self.base_thresholds,
                'false_positive_rate': self.false_positive_rate,
                'missed_fraud_rate': self.missed_fraud_rate,
                'history_length': len(self.user_behavior_history)
            }, f, indent=2)
    
    def load_thresholds(self, filepath: str = "adaptive_thresholds.json"):
        """
        Load thresholds from file
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.base_thresholds = data['thresholds']
                self.false_positive_rate = data.get('false_positive_rate', 0.0)
                self.missed_fraud_rate = data.get('missed_fraud_rate', 0.0)

def main():
    """
    Main function to demonstrate DistilBERT upgrade
    """
    print("🚀 Upgrading SMS Fraud Detection to DistilBERT...")
    
    # Initialize detector
    detector = DistilBERTFraudDetector()
    
    # Test predictions
    test_messages = [
        "Your account has been suspended. Click here to verify: http://fake-bank.com",
        "Your OTP is 123456. Do not share with anyone.",
        "Hi, this is your bank. Your account needs verification.",
        "Congratulations! You've won $1000. Claim now!",
        "Meeting reminder: Tomorrow at 2 PM in conference room."
    ]
    
    print("\n📱 Testing DistilBERT Predictions:")
    for message in test_messages:
        predictions = detector.predict(message)
        print(f"\nMessage: {message[:50]}...")
        for label, prob in predictions.items():
            print(f"  {label}: {prob:.3f}")
    
    # Initialize adaptive thresholds
    threshold_manager = AdaptiveThresholdManager()
    
    # Test adaptive thresholds
    print("\n🎯 Testing Adaptive Thresholds:")
    test_cases = [
        ("trusted_bank", 150),
        ("unknown_number", 80),
        ("alphanumeric", 200),
        ("phone_number", 120)
    ]
    
    for sender_type, length in test_cases:
        thresholds = threshold_manager.get_adaptive_thresholds(sender_type, length)
        print(f"\n{sender_type} (length: {length}):")
        for label, threshold in thresholds.items():
            print(f"  {label}: {threshold:.3f}")
    
    print("\n✅ DistilBERT upgrade demonstration completed!")

if __name__ == "__main__":
    main() 