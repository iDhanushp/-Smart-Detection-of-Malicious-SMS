import joblib
import json
import nltk
from nltk.corpus import stopwords

# Load the trained vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Get vocabulary and IDF values
vocabulary = vectorizer.vocabulary_
idf_values = vectorizer.idf_.tolist()

# Get stop words
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

# Create export data with proper type conversion
export_data = {
    'vocabulary': {str(k): int(v) for k, v in vocabulary.items()},
    'idf': [float(x) for x in idf_values],
    'stop_words': stop_words,
    'max_features': int(vectorizer.max_features)
}

# Save as JSON
with open('tfidf_vocab.json', 'w') as f:
    json.dump(export_data, f, indent=2)

print(f"Vocabulary exported to tfidf_vocab.json")
print(f"Vocabulary size: {len(vocabulary)}")
print(f"IDF values: {len(idf_values)}")
print(f"Stop words: {len(stop_words)}") 