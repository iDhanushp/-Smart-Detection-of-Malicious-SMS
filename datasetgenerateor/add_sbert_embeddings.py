import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load your dataset
INPUT_CSV = "new csv/sample_labeled_fixed.csv"  # Change as needed
OUTPUT_CSV = "new csv/sample_labeled_fixed_with_sbert.csv"

print(f"Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Load SBERT model (MiniLM is fast and accurate)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all messages
print("Generating SBERT embeddings...")
embeddings = model.encode(df['body'].astype(str).tolist(), show_progress_bar=True, batch_size=64)

# Add embeddings as new columns
embeddings_df = pd.DataFrame(embeddings, columns=[f'sbert_{i}' for i in range(embeddings.shape[1])])
df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)

# Save the new dataset
print(f"Saving to: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Done! Saved as {OUTPUT_CSV}") 