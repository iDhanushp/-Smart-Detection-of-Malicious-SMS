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

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths
DATA_PATH = os.path.join('data', 'sms_spam.csv')
VECTORIZER_PATH = 'vectorizer.pkl'
MODEL_PATH = 'best_model.pkl'

# 1. Load Data
df = pd.read_csv(DATA_PATH, encoding='latin-1')
if 'label' not in df.columns or 'text' not in df.columns:
    # UCI format: v1,v2
    df.columns = ['label', 'text'] + list(df.columns[2:])
df = df[['label', 'text']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Preprocess
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

stop_words = set(stopwords.words('english'))
df['text'] = df['text'].astype(str).apply(clean_text)
df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))

# 3. Split
txt_train, txt_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. Vectorize
vectorizer = TfidfVectorizer(max_features=3000)
X_train = vectorizer.fit_transform(txt_train)
X_test = vectorizer.transform(txt_test)

# 5. Train Models
models = {
    'NaiveBayes': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'accuracy': acc, 'report': classification_report(y_test, preds), 'confusion': confusion_matrix(y_test, preds)}
    print(f'\n{name} Accuracy: {acc:.4f}')
    print(results[name]['report'])
    print('Confusion Matrix:\n', results[name]['confusion'])

# 6. Save Best Model & Vectorizer
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f'\nBest model: {best_name}')

joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f'Model saved to {MODEL_PATH}')
print(f'Vectorizer saved to {VECTORIZER_PATH}') 