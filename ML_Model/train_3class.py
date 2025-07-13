import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths
DATA_PATH = os.path.join('data', 'sms_spam.csv')
VECTORIZER_PATH = 'vectorizer.pkl'
MODEL_PATH = 'best_model.pkl'

# 1. Load Data
print("Loading SMS data...")
df = pd.read_csv(DATA_PATH, encoding='latin-1')
if 'label' not in df.columns or 'text' not in df.columns:
    # UCI format: v1,v2
    df.columns = ['label', 'text'] + list(df.columns[2:])
df = df[['label', 'text']]

# Enhanced three-class classification (legitimate=0, spam=1, fraud=2)
def classify_message_3class(text, sender=""):
    """
    Classify SMS message into 3 classes based on content and sender
    0 = Legitimate
    1 = Spam  
    2 = Fraud
    """
    text_lower = text.lower()
    sender_lower = sender.lower()
    
    # Fraud indicators - high priority
    fraud_keywords = [
        'urgent', 'account suspended', 'verify now', 'click here immediately', 
        'bank security', 'password expired', 'unauthorized access', 'verify account',
        'security alert', 'account locked', 'verify identity', 'suspicious activity',
        'fraud alert', 'account compromised', 'verify details', 'security breach',
        'account blocked', 'confirm your identity', 'update your information',
        'action required immediately', 'suspended account', 'verify your account'
    ]
    
    # Spam indicators - commercial/promotional
    spam_keywords = [
        'limited time', 'offer', 'discount', 'sale', 'free', 'win', 'prize',
        'congratulations', 'winner', 'claim', 'exclusive', 'special offer',
        '50% off', 'buy now', 'limited offer', 'flash sale', 'clearance',
        'promotion', 'deal', 'bargain', 'save money', 'best price',
        'call now', 'text stop', 'reply stop', 'unsubscribe'
    ]
    
    # Legitimate indicators - trusted sources
    legit_keywords = [
        'otp', 'verification code', 'one time password', 'appointment',
        'reminder', 'booking confirmed', 'payment received', 'transaction',
        'delivery', 'order', 'receipt', 'invoice', 'statement'
    ]
    
    # Count keyword matches
    fraud_score = sum(1 for keyword in fraud_keywords if keyword in text_lower)
    spam_score = sum(1 for keyword in spam_keywords if keyword in text_lower)
    legit_score = sum(1 for keyword in legit_keywords if keyword in text_lower)
    
    # Sender analysis
    is_phone_number = bool(re.match(r'^\+?\d{10,15}$', sender))
    is_trusted_sender = any(pattern in sender_lower for pattern in [
        'bank', 'sbi', 'hdfc', 'icici', 'axis', 'kotak', 'airtel', 'jio', 'vi'
    ])
    
    # Classification logic
    if fraud_score >= 2 or (fraud_score >= 1 and is_phone_number):
        return 2  # Fraud
    elif fraud_score >= 1 and ('verify' in text_lower and 'account' in text_lower):
        return 2  # Fraud
    elif spam_score >= 2 or ('free' in text_lower and 'offer' in text_lower):
        return 1  # Spam
    elif legit_score >= 1 and is_trusted_sender:
        return 0  # Legitimate
    elif is_trusted_sender and fraud_score == 0:
        return 0  # Legitimate
    elif spam_score >= 1:
        return 1  # Spam
    else:
        return 0  # Legitimate (default)

# Apply 3-class classification
print("Applying 3-class classification...")
df['label'] = df['text'].apply(lambda x: classify_message_3class(x))

print(f"Class distribution:")
print(f"Legitimate (0): {(df['label'] == 0).sum()}")
print(f"Spam (1): {(df['label'] == 1).sum()}")
print(f"Fraud (2): {(df['label'] == 2).sum()}")

# 2. Preprocess
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

print("Preprocessing text...")
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].astype(str).apply(clean_text)
df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))

# 3. Split
print("Splitting data...")
txt_train, txt_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 4. Vectorize
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train = vectorizer.fit_transform(txt_train)
X_test = vectorizer.transform(txt_test)

# 5. Train Models
print("Training models...")
models = {
    'MultinomialNB': MultinomialNB(alpha=1.0),
    'XGBoost': XGBClassifier(
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        verbosity=0,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
}
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    
    results[name] = {
        'model': model, 
        'accuracy': acc, 
        'report': classification_report(y_test, preds),
        'confusion': confusion_matrix(y_test, preds),
        'probabilities': proba
    }
    
    print(f'{name} Accuracy: {acc:.4f}')
    print(results[name]['report'])
    print('Confusion Matrix:')
    print(results[name]['confusion'])

# 6. Save Best Model & Vectorizer
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f'\nBest model: {best_name} (Accuracy: {results[best_name]["accuracy"]:.4f})')

# Save model and vectorizer
joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f'Model saved to {MODEL_PATH}')
print(f'Vectorizer saved to {VECTORIZER_PATH}')

# Test some examples
print("\n" + "="*50)
print("Testing model with sample messages:")
print("="*50)

test_messages = [
    "Your account has been suspended. Verify now at link.com",
    "Free offer! Get 50% discount. Buy now!",
    "Your OTP is 123456. Valid for 10 minutes.",
    "Urgent: Click here to verify your bank account immediately",
    "Congratulations! You've won a prize. Claim now!",
    "Your appointment is confirmed for tomorrow at 2 PM"
]

for msg in test_messages:
    # Preprocess the message
    clean_msg = clean_text(msg)
    clean_msg = ' '.join([w for w in clean_msg.split() if w not in stop_words])
    
    # Vectorize and predict
    msg_vector = vectorizer.transform([clean_msg])
    prediction = best_model.predict(msg_vector)[0]
    probabilities = best_model.predict_proba(msg_vector)[0]
    
    labels = ['Legitimate', 'Spam', 'Fraud']
    print(f"\nMessage: {msg}")
    print(f"Prediction: {labels[prediction]}")
    print(f"Probabilities: Legit={probabilities[0]:.3f}, Spam={probabilities[1]:.3f}, Fraud={probabilities[2]:.3f}")

print(f"\nTraining complete! Model ready for TensorFlow Lite export.") 