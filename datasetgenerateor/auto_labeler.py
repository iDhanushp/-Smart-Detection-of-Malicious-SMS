#!/usr/bin/env python3
"""
AI-powered SMS labeling script using rules and patterns.
"""

import pandas as pd
import re
import argparse
import os
from typing import Dict, List, Tuple

class SMSAutoLabeler:
    def __init__(self):
        # Spam keywords and patterns
        self.spam_keywords = [
            'free', 'win', 'winner', 'congratulations', 'prize', 'lottery',
            'click here', 'urgent', 'limited time', 'offer', 'discount',
            'cash', 'money', 'earn', 'income', 'business opportunity',
            'weight loss', 'pills', 'viagra', 'casino', 'gambling',
            'debt', 'credit', 'loan', 'mortgage', 'insurance',
            'subscribe', 'unsubscribe', 'text stop', 'reply stop',
            'guarantee', 'risk free', 'no obligation', 'act now',
            'call now', 'don\'t miss', 'hurry', 'instant', 'immediately'
        ]
        
        # Fraud-specific keywords
        self.fraud_keywords = [
            'account suspended', 'account blocked', 'verify account',
            'update payment', 'expired', 'suspended', 'blocked',
            'click link', 'verify now', 'confirm identity',
            'security alert', 'suspicious activity', 'unauthorized',
            'refund', 'tax refund', 'government', 'irs', 'revenue',
            'bank', 'paypal', 'amazon', 'apple', 'google',
            'password', 'pin', 'ssn', 'social security',
            'arrest', 'legal action', 'court', 'lawsuit',
            'inheritance', 'beneficiary', 'million dollars',
            'transfer money', 'wire transfer', 'bitcoin', 'cryptocurrency'
        ]
        
        # Legitimate patterns
        self.legit_patterns = [
            r'otp.*\d{4,6}',  # OTP codes
            r'verification.*code.*\d{4,6}',  # Verification codes
            r'your.*code.*is.*\d{4,6}',  # Your code is...
            r'password.*reset',  # Password reset from legitimate services
            r'appointment.*reminder',  # Appointment reminders
            r'delivery.*update',  # Delivery updates
            r'order.*confirmed',  # Order confirmations
            r'payment.*received',  # Payment confirmations
            r'thank.*you.*for',  # Thank you messages
        ]
        
        # Known legitimate senders (partial matches)
        self.legit_senders = [
            'bank', 'paypal', 'amazon', 'google', 'apple', 'microsoft',
            'uber', 'lyft', 'netflix', 'spotify', 'facebook', 'twitter',
            'instagram', 'whatsapp', 'telegram', 'govt', 'government',
            'hospital', 'clinic', 'pharmacy', 'school', 'university'
        ]
        
        # Suspicious sender patterns (for fraud detection)
        self.suspicious_sender_patterns = [
            r'^\+\d{1,3}\d{10,}',  # International numbers
            r'^\d{5,6}$',  # Short codes (can be legit or spam)
            r'^[A-Z]{2,}-[A-Z]{2,}',  # Weird alphanumeric patterns
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()
    
    def contains_keywords(self, text: str, keywords: List[str]) -> Tuple[bool, List[str]]:
        """Check if text contains any of the given keywords."""
        text = self.clean_text(text)
        found_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in text:
                found_keywords.append(keyword)
        
        return len(found_keywords) > 0, found_keywords
    
    def matches_patterns(self, text: str, patterns: List[str]) -> Tuple[bool, List[str]]:
        """Check if text matches any of the given regex patterns."""
        text = self.clean_text(text)
        matched_patterns = []
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched_patterns.append(pattern)
        
        return len(matched_patterns) > 0, matched_patterns
    
    def analyze_sender(self, sender: str) -> Dict[str, any]:
        """Analyze sender information."""
        if pd.isna(sender):
            return {"type": "unknown", "suspicious": False, "legit_indicator": False}
        
        sender = str(sender).strip()
        sender_lower = sender.lower()
        
        # Check if sender looks like a legitimate service
        legit_indicator = any(legit in sender_lower for legit in self.legit_senders)
        
        # Check for suspicious patterns
        suspicious = any(re.match(pattern, sender) for pattern in self.suspicious_sender_patterns)
        
        # Determine sender type
        if sender.isdigit():
            if len(sender) <= 6:
                sender_type = "short_code"
            else:
                sender_type = "phone_number"
        elif sender.startswith('+'):
            sender_type = "international_number"
            suspicious = True  # International numbers are often suspicious
        elif sender.isalpha():
            sender_type = "alphanumeric"
        else:
            sender_type = "mixed"
        
        return {
            "type": sender_type,
            "suspicious": suspicious,
            "legit_indicator": legit_indicator,
            "value": sender
        }
    
    def label_message(self, body: str, sender: str = None) -> Tuple[str, Dict[str, any]]:
        """
        Label a single SMS message.
        
        Returns:
            Tuple of (label, reasoning)
        """
        body = self.clean_text(body)
        sender_info = self.analyze_sender(sender)
        
        reasoning = {
            "sender_info": sender_info,
            "spam_keywords": [],
            "fraud_keywords": [],
            "legit_patterns": [],
            "confidence": 0.0
        }
        
        # Check for legitimate patterns first
        has_legit_patterns, legit_matches = self.matches_patterns(body, self.legit_patterns)
        reasoning["legit_patterns"] = legit_matches
        
        # Check for spam keywords
        has_spam_keywords, spam_matches = self.contains_keywords(body, self.spam_keywords)
        reasoning["spam_keywords"] = spam_matches
        
        # Check for fraud keywords
        has_fraud_keywords, fraud_matches = self.contains_keywords(body, self.fraud_keywords)
        reasoning["fraud_keywords"] = fraud_matches
        
        # Decision logic
        if has_legit_patterns and not has_fraud_keywords:
            if sender_info["legit_indicator"]:
                reasoning["confidence"] = 0.9
                return "legit", reasoning
            elif not sender_info["suspicious"]:
                reasoning["confidence"] = 0.7
                return "legit", reasoning
        
        # Check for fraud (high priority)
        if has_fraud_keywords:
            if sender_info["suspicious"] or sender_info["type"] == "international_number":
                reasoning["confidence"] = 0.95
                return "fraud", reasoning
            elif len(fraud_matches) >= 2:  # Multiple fraud indicators
                reasoning["confidence"] = 0.8
                return "fraud", reasoning
            else:
                reasoning["confidence"] = 0.6
                return "spam", reasoning
        
        # Check for spam
        if has_spam_keywords:
            if len(spam_matches) >= 3:  # Multiple spam indicators
                reasoning["confidence"] = 0.8
                return "spam", reasoning
            elif sender_info["suspicious"]:
                reasoning["confidence"] = 0.7
                return "spam", reasoning
            else:
                reasoning["confidence"] = 0.6
                return "spam", reasoning
        
        # Check sender-based classification
        if sender_info["suspicious"] and not sender_info["legit_indicator"]:
            reasoning["confidence"] = 0.5
            return "spam", reasoning
        
        # Default to legit if no clear indicators
        reasoning["confidence"] = 0.3
        return "legit", reasoning
    
    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label an entire dataset."""
        results = []
        
        for idx, row in df.iterrows():
            body = row.get('body', '')
            sender = row.get('address', '') or row.get('sender', '')
            
            label, reasoning = self.label_message(body, sender)
            
            results.append({
                'predicted_label': label,
                'confidence': reasoning['confidence'],
                'spam_keywords': ', '.join(reasoning['spam_keywords']),
                'fraud_keywords': ', '.join(reasoning['fraud_keywords']),
                'legit_patterns': ', '.join(reasoning['legit_patterns']),
                'sender_type': reasoning['sender_info']['type'],
                'sender_suspicious': reasoning['sender_info']['suspicious']
            })
        
        # Add results to dataframe
        result_df = df.copy()
        for key in results[0].keys():
            result_df[key] = [r[key] for r in results]
        
        return result_df

def main():
    parser = argparse.ArgumentParser(description='Auto-label SMS messages using AI rules')
    parser.add_argument('input_file', help='Input CSV file with SMS messages')
    parser.add_argument('-o', '--output', default='sms_auto_labeled.csv',
                       help='Output CSV file (default: sms_auto_labeled.csv)')
    parser.add_argument('--review-threshold', type=float, default=0.6,
                       help='Confidence threshold below which messages need review (default: 0.6)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    print(f"Loading SMS data from: {args.input_file}")
    
    try:
        df = pd.read_csv(args.input_file)
        print(f"Total messages loaded: {len(df)}")
        
        # Initialize labeler
        labeler = SMSAutoLabeler()
        
        # Label the dataset
        print("Labeling messages...")
        labeled_df = labeler.label_dataset(df)
        
        # Save results
        labeled_df.to_csv(args.output, index=False)
        print(f"Labeled dataset saved to: {args.output}")
        
        # Print summary
        label_counts = labeled_df['predicted_label'].value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(labeled_df)*100:.1f}%)")
        
        # Messages that need review
        low_confidence = labeled_df[labeled_df['confidence'] < args.review_threshold]
        print(f"\nMessages needing review (confidence < {args.review_threshold}): {len(low_confidence)}")
        
        if len(low_confidence) > 0:
            print(f"Review file: {args.output.replace('.csv', '_review.csv')}")
            low_confidence.to_csv(args.output.replace('.csv', '_review.csv'), index=False)
        
        print("\n✅ Auto-labeling completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 