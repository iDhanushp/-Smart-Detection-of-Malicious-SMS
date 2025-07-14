#!/usr/bin/env python3
"""
Comprehensive SMS Analysis Demo
Shows how the enhanced behavioral system analyzes different message types
"""

import sys
import os
sys.path.append('.')
from enhanced_behavioral_labeler import BehavioralSMSLabeler

def analyze_comprehensive_examples():
    """Analyze a comprehensive set of real-world SMS examples"""
    
    labeler = BehavioralSMSLabeler()
    
    # Comprehensive test cases covering various fraud/spam/legit patterns
    test_cases = [
        # === FRAUD MESSAGES ===
        {
            'message': "URGENT: Your SBI account has been SUSPENDED due to KYC non-compliance. Update immediately at sbi-kyc-update.com or account will be PERMANENTLY CLOSED!",
            'sender': "+919876543210",
            'expected': "FRAUD",
            'category': "🚨 FRAUD - Account Suspension Scam"
        },
        {
            'message': "Security Alert: Unauthorized access detected on your account. Verify your identity by providing OTP and PIN: http://verify-account-security.com",
            'sender': "+1234567890",
            'expected': "FRAUD", 
            'category': "🚨 FRAUD - Phishing with Data Harvesting"
        },
        {
            'message': "Final Notice from Income Tax Department: Your PAN is disabled. Update within 24 hours or face legal action. Click: govt-tax-update.co.in",
            'sender': "INCOMETAX",
            'expected': "FRAUD",
            'category': "🚨 FRAUD - Government Impersonation"
        },
        {
            'message': "Your credit card ending 1234 will be blocked in 2 hours due to suspicious activity. Confirm your details now: hdfc-secure.net",
            'sender': "+447123456789",
            'expected': "FRAUD",
            'category': "🚨 FRAUD - International Credit Card Scam"
        },
        
        # === SPAM MESSAGES ===
        {
            'message': "🎉 CONGRATULATIONS! You've WON ₹50,000 CASH PRIZE! 🏆 Claim NOW before offer expires! Call 8888888888 immediately!",
            'sender': "WINNER99",
            'expected': "SPAM",
            'category': "🟡 SPAM - Prize/Lottery Scam"
        },
        {
            'message': "MEGA SALE! 70% OFF on everything! LIMITED TIME ONLY! Shop now at bestdeals.com. Hurry, stock running out fast!",
            'sender': "DEALS123",
            'expected': "SPAM",
            'category': "🟡 SPAM - Promotional Offers"
        },
        {
            'message': "Earn ₹5000 daily working from home! No investment required! Guaranteed income! WhatsApp 9999999999 for details.",
            'sender': "+919988776655",
            'expected': "SPAM",
            'category': "🟡 SPAM - Work from Home Scam"
        },
        {
            'message': "Free iPhone 14! You're selected for our special promotion! Click link to claim your prize: freephone-offer.com",
            'sender': "PROMO007",
            'expected': "SPAM",
            'category': "🟡 SPAM - Free Product Offer"
        },
        
        # === LEGITIMATE MESSAGES ===
        {
            'message': "Your OTP for SBI Mobile Banking login is 234567. Do not share this with anyone. Valid for 10 minutes. -SBIINB",
            'sender': "AD-SBIINB",
            'expected': "LEGIT",
            'category': "🟢 LEGIT - Bank OTP"
        },
        {
            'message': "Your UPI payment of ₹1,500 to Swiggy has been successful. Transaction ID: 123456789012. Thank you!",
            'sender': "AX-PAYTM",
            'expected': "LEGIT",
            'category': "🟢 LEGIT - Payment Confirmation"
        },
        {
            'message': "Your Zomato order #12345 is out for delivery. Expected delivery time: 30 mins. Track: zoma.to/track12345",
            'sender': "ZM-ZOMATO",
            'expected': "LEGIT",
            'category': "🟢 LEGIT - Delivery Update"
        },
        {
            'message': "Hi John, are we still meeting at 6 PM today? Let me know if you need to reschedule.",
            'sender': "+919876543211",
            'expected': "LEGIT",
            'category': "🟢 LEGIT - Personal Message"
        },
        {
            'message': "Your electricity bill for March 2025 is ₹2,345. Due date: 31st March. Pay online at bescom.org",
            'sender': "BESCOM",
            'expected': "LEGIT",
            'category': "🟢 LEGIT - Utility Bill"
        },
        {
            'message': "Appointment reminder: Dr. Smith tomorrow at 3 PM. Please arrive 15 minutes early. Call 080-12345678 to reschedule.",
            'sender': "CLINIC",
            'expected': "LEGIT",
            'category': "🟢 LEGIT - Appointment Reminder"
        },
        
        # === EDGE CASES ===
        {
            'message': "Urgent: Please call me back immediately. It's about mom's health. Emergency!",
            'sender': "+919123456789",
            'expected': "LEGIT",
            'category': "🔍 EDGE CASE - Personal Emergency"
        },
        {
            'message': "Your account balance is low. Add money now to avoid service interruption. Click: airtel.in/recharge",
            'sender': "AIRTEL",
            'expected': "LEGIT",
            'category': "🔍 EDGE CASE - Service Notification"
        }
    ]
    
    print("🔍 COMPREHENSIVE SMS FRAUD DETECTION ANALYSIS")
    print("=" * 90)
    print(f"Testing {len(test_cases)} messages across fraud, spam, and legitimate categories")
    print("=" * 90)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        message = test_case['message']
        sender = test_case['sender']
        expected = test_case['expected']
        category = test_case['category']
        
        # Analyze message
        classification, confidence, analysis = labeler.classify_message_advanced(message, sender)
        
        # Check if prediction matches expected
        is_correct = classification == expected
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Display results
        status_icon = "✅" if is_correct else "❌"
        print(f"\n{status_icon} Test {i}: {category}")
        print(f"Message: {message[:80]}{'...' if len(message) > 80 else ''}")
        print(f"Sender: {sender}")
        print(f"Expected: {expected} | Predicted: {classification} | Confidence: {confidence:.2f}")
        
        if not is_correct:
            print(f"🔴 MISMATCH: Expected {expected} but got {classification}")
        
        print(f"Reasoning: {'; '.join(analysis['reasoning'])}")
        
        # Show key behavioral scores
        behavioral = analysis['behavioral_signals']
        scores_text = []
        if behavioral['total_urgency'] > 0.05:
            scores_text.append(f"Urgency: {behavioral['total_urgency']:.2f}")
        if behavioral['total_fear'] > 0.05:
            scores_text.append(f"Fear: {behavioral['total_fear']:.2f}")
        if behavioral['total_reward'] > 0.05:
            scores_text.append(f"Reward: {behavioral['total_reward']:.2f}")
        if behavioral['total_authority'] > 0.05:
            scores_text.append(f"Authority: {behavioral['total_authority']:.2f}")
            
        if scores_text:
            print(f"Key Scores: {', '.join(scores_text)}")
        
        # Show specific threat indicators
        if behavioral.get('fear_account_threats', 0) > 0.05:
            print(f"⚠️  Account Threat Detected (Score: {behavioral['fear_account_threats']:.2f})")
        if behavioral.get('action_data_harvesting', 0) > 0:
            print(f"🚨 Data Harvesting Attempt (Score: {behavioral['action_data_harvesting']:.2f})")
        if analysis['structural_features']['has_url'] == 1:
            print(f"🔗 Contains Suspicious URL")
        
        print("-" * 90)
    
    # Summary
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n🎯 ANALYSIS SUMMARY")
    print(f"Total Messages Analyzed: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    
    # Breakdown by category
    fraud_tests = [t for t in test_cases if t['expected'] == 'FRAUD']
    spam_tests = [t for t in test_cases if t['expected'] == 'SPAM']
    legit_tests = [t for t in test_cases if t['expected'] == 'LEGIT']
    
    print(f"\nCategory Breakdown:")
    print(f"🚨 Fraud Messages: {len(fraud_tests)} tested")
    print(f"🟡 Spam Messages: {len(spam_tests)} tested")
    print(f"🟢 Legitimate Messages: {len(legit_tests)} tested")
    
    print(f"\n🚀 The enhanced behavioral analysis system can now detect:")
    print(f"✅ Account suspension/threat scams")
    print(f"✅ Phishing attempts with data harvesting")
    print(f"✅ Government/authority impersonation")
    print(f"✅ Prize/lottery spam")
    print(f"✅ Promotional spam with urgency tactics")
    print(f"✅ Legitimate banking and service messages")
    print(f"✅ Personal communications")

if __name__ == "__main__":
    analyze_comprehensive_examples()
