#!/usr/bin/env python3
"""
Enhanced SMS Auto-Labeler with Behavioral Pattern Recognition
Replaces simple keyword matching with sophisticated behavioral analysis
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import argparse
import os
from datetime import datetime
import json

class BehavioralSMSLabeler:
    def __init__(self):
        """Initialize enhanced behavioral labeler"""
        
        # Psychological manipulation patterns
        self.urgency_patterns = {
            'immediate': [
                r'\b(urgent|immediate|now|asap|quickly|hurry|rush)\b',
                r'\b(expire|expires|expiring|deadline|limited time)\b',
                r'\b(act now|call now|click now|verify now|update now)\b',
                r'\b(within \d+ (hours?|minutes?|days?))\b',
                r'\b(today only|expires today|last chance)\b'
            ],
            'time_pressure': [
                r'\b(24 hours?|48 hours?|2 days?|tomorrow)\b',
                r'\b(before (midnight|end of day|close of business))\b',
                r'\b(final (notice|warning|reminder))\b'
            ]
        }
        
        self.fear_intimidation = {
            'account_threats': [
                r'\b(suspended|blocked|closed|terminated|disabled|frozen)\b',
                r'\b(unauthorized|suspicious|fraud|security|breach)\b',
                r'\b(warning|alert|notice|violation|penalty)\b',
                r'\b(legal action|court|arrest|police|investigation)\b'
            ],
            'loss_threats': [
                r'\b(lose access|will be lost|permanently deleted)\b',
                r'\b(account will be|service will be|card will be)\b',
                r'\b(unable to access|cannot use|restricted)\b'
            ]
        }
        
        self.reward_manipulation = {
            'prizes': [
                r'\b(win|won|winner|prize|reward|gift|free|bonus)\b',
                r'\b(congratulations|selected|chosen|lucky|special)\b',
                r'\b(exclusive|limited|rare|unique|once in lifetime)\b'
            ],
            'money': [
                r'\b(cashback|refund|money|cash|₹|rs\.?\s*\d+|rupees)\b',
                r'\b(\d+\s*(lakh|crore|thousand|million))\b',
                r'\b(earn|income|profit|commission|percentage)\b'
            ]
        }
        
        self.authority_mimicry = {
            'financial': [
                r'\b(bank|rbi|sebi|hdfc|icici|sbi|axis|kotak)\b',
                r'\b(credit card|debit card|net banking|mobile banking)\b',
                r'\b(loan|emi|payment|transaction|account)\b'
            ],
            'government': [
                r'\b(government|tax|income tax|gst|pan|aadhar)\b',
                r'\b(department|ministry|bureau|agency|authority)\b',
                r'\b(official|authorized|verified|certified|legal)\b'
            ],
            'tech_companies': [
                r'\b(google|facebook|amazon|apple|microsoft|whatsapp)\b',
                r'\b(otp|verification|security code|login)\b'
            ]
        }
        
        self.action_requests = {
            'data_harvesting': [
                r'\b(provide|enter|give|send|share|disclose)\b.*\b(otp|pin|password|cvv)\b',
                r'\b(update|verify|confirm)\b.*\b(details|information|kyc)\b',
                r'\b(bank details|account number|card number|ifsc)\b'
            ],
            'immediate_actions': [
                r'\b(click|tap|call|reply|send|forward|share|download)\b',
                r'\b(visit|go to|open|install|register|submit)\b'
            ]
        }
        
        # Legitimate service patterns (Indian context)
        self.legitimate_patterns = {
            'bank_codes': [
                r'^(AX|AD|JM|CP|VM|VK|BZ|TX|JD|BK|BP|JX|TM|QP|BV|JK|BH|TG|JG|VD)-',
                r'\b(SBIINB|HDFCBK|ICICIB|AXISBK|KOTAKB)\b',
                r'\b(AIRTEL|VODAFONE|JIO|BSNL)\b'
            ],
            'otp_patterns': [
                r'\botp\s*(is|code)?\s*:?\s*\d{4,6}\b',
                r'\bverification code\s*:?\s*\d{4,6}\b',
                r'\byour.*code.*is.*\d{4,6}\b',
                r'\b\d{4,6}\s*is\s*your.*otp\b'
            ],
            'service_notifications': [
                r'\b(order|delivery|booking|appointment|reservation)\b',
                r'\b(confirmed|successful|completed|scheduled)\b',
                r'\b(thank you|thanks|receipt|invoice)\b'
            ]
        }
        
        # Spam promotional patterns
        self.promotional_patterns = {
            'offers': [
                r'\b(offer|discount|sale|deal|bargain)\b',
                r'\b(\d+%\s*(off|discount)|up to \d+%)\b',
                r'\b(special|exclusive|limited|mega|super)\b'
            ],
            'subscription': [
                r'\b(subscribe|unsubscribe|text stop|reply stop)\b',
                r'\b(daily|weekly|monthly)\s*(tips|updates|news)\b'
            ]
        }
    
    def calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate weighted score for pattern matches"""
        text_lower = text.lower()
        matches = 0
        total_words = len(text.split())
        
        for pattern in patterns:
            matches += len(re.findall(pattern, text_lower))
        
        # Normalize by text length
        return matches / max(total_words, 1)
    
    def analyze_behavioral_signals(self, text: str) -> Dict[str, float]:
        """Comprehensive behavioral analysis of message"""
        signals = {}
        
        # Urgency analysis
        urgency_score = 0
        for category, patterns in self.urgency_patterns.items():
            score = self.calculate_pattern_score(text, patterns)
            signals[f'urgency_{category}'] = score
            urgency_score += score
        signals['total_urgency'] = urgency_score
        
        # Fear/intimidation analysis
        fear_score = 0
        for category, patterns in self.fear_intimidation.items():
            score = self.calculate_pattern_score(text, patterns)
            signals[f'fear_{category}'] = score
            fear_score += score
        signals['total_fear'] = fear_score
        
        # Reward manipulation analysis
        reward_score = 0
        for category, patterns in self.reward_manipulation.items():
            score = self.calculate_pattern_score(text, patterns)
            signals[f'reward_{category}'] = score
            reward_score += score
        signals['total_reward'] = reward_score
        
        # Authority mimicry analysis
        authority_score = 0
        for category, patterns in self.authority_mimicry.items():
            score = self.calculate_pattern_score(text, patterns)
            signals[f'authority_{category}'] = score
            authority_score += score
        signals['total_authority'] = authority_score
        
        # Action request analysis
        action_score = 0
        for category, patterns in self.action_requests.items():
            score = self.calculate_pattern_score(text, patterns)
            signals[f'action_{category}'] = score
            action_score += score
        signals['total_action'] = action_score
        
        return signals
    
    def analyze_structural_features(self, text: str) -> Dict[str, float]:
        """Analyze structural properties of message"""
        features = {}
        
        # Basic structure
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character analysis
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        
        # Punctuation patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['dots_count'] = text.count('...')
        features['caps_words'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        
        # URLs and contact info
        features['has_url'] = 1 if re.search(r'http[s]?://|www\.|[a-z]+\.[a-z]{2,3}', text.lower()) else 0
        features['has_phone'] = 1 if re.search(r'\+?\d{10,}', text) else 0
        features['has_email'] = 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0
        
        return features
    
    def calculate_legitimacy_score(self, text: str) -> float:
        """Calculate how legitimate the message appears"""
        legitimacy_score = 0
        
        for category, patterns in self.legitimate_patterns.items():
            score = self.calculate_pattern_score(text, patterns)
            if category == 'bank_codes':
                legitimacy_score += score * 2.0  # Bank codes are strong legitimacy indicators
            elif category == 'otp_patterns':
                legitimacy_score += score * 1.5
            else:
                legitimacy_score += score
        
        return min(legitimacy_score, 1.0)  # Cap at 1.0
    
    def classify_message_advanced(self, text: str, sender: str = "") -> Tuple[str, float, Dict]:
        """Advanced classification using behavioral analysis"""
        
        # Analyze behavioral signals
        behavioral_signals = self.analyze_behavioral_signals(text)
        structural_features = self.analyze_structural_features(text)
        legitimacy_score = self.calculate_legitimacy_score(text)
        
        # Classification logic
        fraud_score = 0
        spam_score = 0
        legit_score = legitimacy_score
        
        # Fraud indicators (high risk) - FIXED THRESHOLDS
        # Strong fraud signals - account threats + urgency
        if (behavioral_signals['total_fear'] > 0.05 and 
            behavioral_signals['total_urgency'] > 0.05):
            fraud_score += 0.7
        
        # Financial authority impersonation with data requests
        if behavioral_signals.get('authority_financial', 0) > 0.05:
            fraud_score += 0.6
            
        # Account threats are strong fraud indicators
        if behavioral_signals.get('fear_account_threats', 0) > 0.05:
            fraud_score += 0.8
        
        # URLs with threats = phishing
        if (behavioral_signals['total_fear'] > 0.05 and 
            structural_features['has_url'] == 1):
            fraud_score += 0.6
        
        # Data harvesting attempts (asking for OTP, PIN, etc.)
        if behavioral_signals.get('action_data_harvesting', 0) > 0:
            fraud_score += 0.9
        
        # Government/authority impersonation with threats
        if (behavioral_signals.get('authority_government', 0) > 0.05 and 
            behavioral_signals['total_fear'] > 0.05):
            fraud_score += 0.7
        
        # Spam indicators (medium risk) - ADJUSTED THRESHOLDS
        if (behavioral_signals['total_reward'] > 0.05 and 
            behavioral_signals['total_urgency'] > 0.03):
            spam_score += 0.6
        
        # Money/prize offers are spam
        if (behavioral_signals.get('reward_money', 0) > 0.05 or 
            behavioral_signals.get('reward_prizes', 0) > 0.05):
            spam_score += 0.7
        
        # Promotional language patterns
        promotional_keywords = ['offer', 'discount', 'sale', 'deal', 'limited time']
        if any(keyword in text.lower() for keyword in promotional_keywords):
            spam_score += 0.4
        
        # Adjust for sender patterns
        if sender:
            sender_lower = sender.lower()
            if re.match(r'^\+\d{10,}$', sender):  # International phone number
                if fraud_score > 0.2:
                    fraud_score += 0.3  # International numbers with threats = high fraud risk
                if spam_score > 0.2:
                    spam_score += 0.2
            elif re.match(r'^[A-Z]{2}-', sender):  # Bank code pattern
                legit_score += 0.5
                fraud_score *= 0.3  # Reduce fraud score for legitimate bank codes
                spam_score *= 0.3
            elif sender.isdigit() and len(sender) <= 6:  # Short codes
                legit_score += 0.3
                fraud_score *= 0.5
        
        # Override: If legitimate bank/service patterns detected, reduce fraud/spam scores
        if legitimacy_score > 0.3:
            fraud_score *= 0.5
            spam_score *= 0.7
        
        # Special case: Legitimate bank transaction notifications
        # These often contain words like "fraud" in security context but are legitimate
        if (sender and (re.match(r'^[A-Z]{2}-[A-Z]+B?-?[A-Z]$', sender) or 
                       'bank' in sender.lower() or 'kotakb' in sender.lower()) and
            ('sent rs.' in text.lower() or 'received rs.' in text.lower() or 
             'spent via' in text.lower() or 'credited to' in text.lower())):
            # This is a legitimate bank transaction notification
            fraud_score *= 0.1  # Drastically reduce fraud score
            spam_score *= 0.3
            legit_score += 0.8
        
        # Final classification with adjusted thresholds
        confidence = max(fraud_score, spam_score, legit_score)
        
        if fraud_score > max(spam_score, legit_score) and fraud_score > 0.3:  # Lowered threshold
            return "FRAUD", confidence, {
                'behavioral_signals': behavioral_signals,
                'structural_features': structural_features,
                'reasoning': self._generate_reasoning(behavioral_signals, 'fraud'),
                'scores': {'fraud': fraud_score, 'spam': spam_score, 'legit': legit_score}
            }
        elif spam_score > legit_score and spam_score > 0.25:  # Lowered threshold
            return "SPAM", confidence, {
                'behavioral_signals': behavioral_signals,
                'structural_features': structural_features,
                'reasoning': self._generate_reasoning(behavioral_signals, 'spam'),
                'scores': {'fraud': fraud_score, 'spam': spam_score, 'legit': legit_score}
            }
        else:
            return "LEGIT", confidence, {
                'behavioral_signals': behavioral_signals,
                'structural_features': structural_features,
                'reasoning': self._generate_reasoning(behavioral_signals, 'legit'),
                'scores': {'fraud': fraud_score, 'spam': spam_score, 'legit': legit_score}
            }
    
    def _generate_reasoning(self, signals: Dict[str, float], classification: str) -> List[str]:
        """Generate human-readable reasoning for classification"""
        reasons = []
        
        if classification == 'fraud':
            if signals.get('fear_account_threats', 0) > 0.05:
                reasons.append("Contains account threat language (suspended/blocked/closed)")
            if signals['total_fear'] > 0.05:
                reasons.append("Uses fear-inducing language")
            if signals['total_urgency'] > 0.05:
                reasons.append("Creates false urgency and time pressure")
            if signals.get('action_data_harvesting', 0) > 0:
                reasons.append("Requests sensitive personal/financial information")
            if signals.get('authority_financial', 0) > 0.05:
                reasons.append("Impersonates financial institutions")
            if signals.get('authority_government', 0) > 0.05:
                reasons.append("Impersonates government authority")
        
        elif classification == 'spam':
            if signals.get('reward_prizes', 0) > 0.05:
                reasons.append("Offers unrealistic rewards or prizes")
            if signals.get('reward_money', 0) > 0.05:
                reasons.append("Promises money or cash benefits")
            if signals['total_urgency'] > 0.03:
                reasons.append("Uses promotional urgency tactics")
            if signals['total_reward'] > 0.05:
                reasons.append("Contains promotional/marketing content")
        
        else:  # legit
            reasons.append("Appears to be legitimate service communication")
            if signals.get('authority_financial', 0) > 0 and signals['total_fear'] < 0.05:
                reasons.append("Matches legitimate banking communication patterns")
            if signals.get('authority_tech_companies', 0) > 0 and signals['total_fear'] < 0.05:
                reasons.append("Appears to be from legitimate tech service")
        
        if not reasons:
            reasons.append(f"Classified as {classification.lower()} based on overall pattern analysis")
        
        return reasons
    
    def process_dataset(self, df: pd.DataFrame, text_column: str = 'body', 
                       sender_column: str = 'address') -> pd.DataFrame:
        """Process entire dataset with enhanced labeling"""
        results = []
        
        print(f"Processing {len(df)} messages with behavioral analysis...")
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(df)} messages")
            
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            sender = str(row[sender_column]) if pd.notna(row[sender_column]) else ""
            
            if not text.strip():
                continue
            
            classification, confidence, analysis = self.classify_message_advanced(text, sender)
            
            results.append({
                'id': idx,
                'text': text,
                'sender': sender,
                'classification': classification,
                'confidence': confidence,
                'reasoning': '; '.join(analysis['reasoning']),
                'urgency_score': analysis['behavioral_signals']['total_urgency'],
                'fear_score': analysis['behavioral_signals']['total_fear'],
                'reward_score': analysis['behavioral_signals']['total_reward'],
                'authority_score': analysis['behavioral_signals']['total_authority'],
                'action_score': analysis['behavioral_signals']['total_action']
            })
        
        result_df = pd.DataFrame(results)
        
        # Print statistics
        print(f"\nLabeling Results:")
        print(f"Total messages processed: {len(result_df)}")
        print(f"Classification distribution:")
        print(result_df['classification'].value_counts())
        print(f"\nAverage confidence by class:")
        print(result_df.groupby('classification')['confidence'].mean())
        
        return result_df

def main():
    """Example usage"""
    parser = argparse.ArgumentParser(description='Enhanced SMS Behavioral Labeler')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--text-column', default='body', help='Text column name')
    parser.add_argument('--sender-column', default='address', help='Sender column name')
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = BehavioralSMSLabeler()
    
    # Load dataset
    print(f"Loading dataset from {args.input}")
    df = pd.read_csv(args.input)
    
    # Process dataset
    labeled_df = labeler.process_dataset(df, args.text_column, args.sender_column)
    
    # Save results
    labeled_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Save analysis summary
    summary_path = args.output.replace('.csv', '_analysis_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_messages': len(labeled_df),
        'classification_counts': labeled_df['classification'].value_counts().to_dict(),
        'average_confidence': labeled_df.groupby('classification')['confidence'].mean().to_dict(),
        'high_confidence_threshold': 0.7,
        'high_confidence_count': len(labeled_df[labeled_df['confidence'] > 0.7])
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis summary saved to {summary_path}")

if __name__ == "__main__":
    # Quick test with sample messages
    if len(os.sys.argv) == 1:
        labeler = BehavioralSMSLabeler()
        
        test_messages = [
            ("Your account has been SUSPENDED due to suspicious activity. URGENT: Click here to verify immediately or lose access forever!", "+919876543210"),
            ("Congratulations! You've WON ₹5 LAKH! Claim your prize NOW! Limited time offer expires in 2 hours!", "WINNER123"),
            ("Your OTP for SBI Mobile Banking is 123456. Do not share with anyone. -SBIINB", "AD-SBIINB"),
            ("Hi, are we still meeting for lunch tomorrow?", "+919123456789"),
            ("Your Zomato order #12345 has been delivered. Enjoy your meal!", "ZM-ZOMATO"),
            ("FINAL NOTICE: Your credit card will be blocked in 24 hours. Update your KYC details now!", "+1234567890")
        ]
        
        print("Testing Enhanced Behavioral SMS Labeler")
        print("=" * 60)
        
        for i, (message, sender) in enumerate(test_messages, 1):
            classification, confidence, analysis = labeler.classify_message_advanced(message, sender)
            print(f"\nMessage {i}: {message}")
            print(f"Sender: {sender}")
            print(f"Classification: {classification} (Confidence: {confidence:.2f})")
            print(f"Reasoning: {'; '.join(analysis['reasoning'])}")
            
            # Show detailed scores
            scores = analysis.get('scores', {})
            print(f"Detailed Scores - Fraud: {scores.get('fraud', 0):.2f}, "
                  f"Spam: {scores.get('spam', 0):.2f}, "
                  f"Legit: {scores.get('legit', 0):.2f}")
            
            print(f"Behavioral Scores - Urgency: {analysis['behavioral_signals']['total_urgency']:.2f}, "
                  f"Fear: {analysis['behavioral_signals']['total_fear']:.2f}, "
                  f"Reward: {analysis['behavioral_signals']['total_reward']:.2f}, "
                  f"Authority: {analysis['behavioral_signals']['total_authority']:.2f}")
            
            # Show specific threat indicators
            if analysis['behavioral_signals'].get('fear_account_threats', 0) > 0:
                print(f"⚠️  Account Threat Score: {analysis['behavioral_signals']['fear_account_threats']:.2f}")
            if analysis['behavioral_signals'].get('action_data_harvesting', 0) > 0:
                print(f"🚨 Data Harvesting Score: {analysis['behavioral_signals']['action_data_harvesting']:.2f}")
            if analysis['structural_features']['has_url'] == 1:
                print(f"🔗 Contains URL/Link")
            
            print("=" * 80)
    else:
        main()
