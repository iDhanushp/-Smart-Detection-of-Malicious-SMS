import joblib
import json

VECTORIZER_PATH = 'vectorizer.pkl'
OUTPUT_JSON = 'tfidf_vocab.json'

vectorizer = joblib.load(VECTORIZER_PATH)

# Export vocabulary and idf values
vocab = vectorizer.vocabulary_
idf = vectorizer.idf_.tolist()

# Map word to index and idf
export = {
    'vocabulary': vocab,
    'idf': idf,
    'stop_words': list(vectorizer.stop_words_) if hasattr(vectorizer, 'stop_words_') else []
}

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(export, f, ensure_ascii=False, indent=2)

print(f'TF-IDF vocabulary exported to {OUTPUT_JSON}') 