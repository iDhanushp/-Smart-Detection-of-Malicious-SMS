#!/usr/bin/env python3
"""
Test the fix for bank transaction notifications
"""

from enhanced_behavioral_labeler import BehavioralSMSLabeler

def test_bank_transaction_fix():
    labeler = BehavioralSMSLabeler()
    
    # Test legitimate bank transaction
    bank_message = "Rs.2.00 spent via Kotak Debit Card XX5673 at AMAZONAWSESC on 12/07/2025. Avl bal Rs.15566.11 Not you?Tap https://kotak.com/KBANKT/Fraud"
    bank_sender = "AX-KOTAKB-S"
    
    result = labeler.classify_message_advanced(bank_message, bank_sender)
    
    print("🏦 TESTING BANK TRANSACTION NOTIFICATION")
    print("=" * 60)
    print(f"Message: {bank_message}")
    print(f"Sender: {bank_sender}")
    print(f"Classification: {result[0]}")
    print(f"Confidence: {result[1]:.2f}")
    print(f"Reasoning: {'; '.join(result[2]['reasoning'])}")
    print(f"Scores: {result[2]['scores']}")
    
    # Test actual fraud message
    fraud_message = "URGENT: Your account suspended! Verify now or lose access forever: fake-bank.com"
    fraud_sender = "+1234567890"
    
    result2 = labeler.classify_message_advanced(fraud_message, fraud_sender)
    
    print("\n🚨 TESTING ACTUAL FRAUD MESSAGE")
    print("=" * 60)
    print(f"Message: {fraud_message}")
    print(f"Sender: {fraud_sender}")
    print(f"Classification: {result2[0]}")
    print(f"Confidence: {result2[1]:.2f}")
    print(f"Reasoning: {'; '.join(result2[2]['reasoning'])}")
    print(f"Scores: {result2[2]['scores']}")

if __name__ == "__main__":
    test_bank_transaction_fix()
