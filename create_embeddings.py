import pandas as pd
from openai import OpenAI
import numpy as np
import pickle
import os

# --- CONFIGURATION ---
CSV_FILE_PATH = 'Dentaly URLS - Dentaly US.csv'  # Change this to the CSV you want to process
EMBEDDINGS_FILE_PATH = 'dentaly_us_embeddings.pkl'
EMBEDDING_MODEL = "text-embedding-3-small"
# ---------------------

print("Starting the embedding generation process...")

# 1. Initialize OpenAI Client
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    print("Error: OPENAI_API_KEY not found. Please set it as an environment variable.")
    exit()

# 2. Load and Prepare Data
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Successfully loaded {CSV_FILE_PATH} with {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: The file {CSV_FILE_PATH} was not found.")
    exit()

# For this example, we'll combine title, meta, and details for a rich context.
# We also handle potential missing data by filling NaNs with empty strings.
df['combined_text'] = (
    "Title: " + df['title'].fillna('') + "; " +
    "Meta: " + df['meta'].fillna('') + "; " +
    "Content: " + df['Page Detail'].fillna('')
)

# 3. Generate Embeddings
print(f"Generating embeddings using the '{EMBEDDING_MODEL}' model. This may take a while...")
embeddings_data = []
for index, row in df.iterrows():
    try:
        text_to_embed = row['combined_text']
        if not text_to_embed.strip():
            print(f"Skipping row {index} due to empty content.")
            continue

        response = client.embeddings.create(
            input=text_to_embed,
            model=EMBEDDING_MODEL
        )
        embedding_vector = response.data[0].embedding

        embeddings_data.append({
            'url': row['url'],
            'title': row['title'],
            'combined_text': text_to_embed,
            'vector': np.array(embedding_vector) # Store as a NumPy array
        })
        print(f"Processed row {index + 1}/{len(df)}")

    except Exception as e:
        print(f"An error occurred on row {index}: {e}")

# 4. Save the Embeddings
if embeddings_data:
    with open(EMBEDDINGS_FILE_PATH, 'wb') as f:
        pickle.dump(embeddings_data, f)
    print(f"\nSuccessfully created and saved embeddings to {EMBEDDINGS_FILE_PATH}")
else:
    print("\nNo embeddings were generated. The output file was not created.")