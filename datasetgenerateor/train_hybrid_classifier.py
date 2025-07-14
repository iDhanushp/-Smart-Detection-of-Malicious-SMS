import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# Load the SBERT-augmented dataset
df = pd.read_csv("new csv/sample_labeled_fixed_with_sbert.csv")  # Change filename as needed

# Prepare labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'].astype(str))

# TF-IDF features
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
tfidf_features = tfidf.fit_transform(df['body'].astype(str)).toarray()

# SBERT features (all columns starting with 'sbert_')
sbert_cols = [col for col in df.columns if col.startswith('sbert_')]
sbert_features = df[sbert_cols].values

# Combine features
X = np.hstack([tfidf_features, sbert_features])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Train XGBoost
model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, label_encoder.inverse_transform(y_pred)))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and encoders
joblib.dump({
    'model': model,
    'label_encoder': label_encoder,
    'tfidf': tfidf,
    'sbert_cols': sbert_cols
}, "hybrid_sms_classifier.pkl")
print("✅ Hybrid model saved as hybrid_sms_classifier.pkl")

# Export TF-IDF vocab/IDF for Dart
vocab = tfidf.vocabulary_
idf = tfidf.idf_.tolist()
with open("tfidf_vocab.json", "w") as f:
    json.dump({"vocab": vocab, "idf": idf}, f)
print("✅ Exported tfidf_vocab.json") 