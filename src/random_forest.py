import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

# Path to your dataset
csv_path = "../data/processed_data/processed_dataset2.csv"

# Columns to use for training (excluding playlist_id, track_id, and label)
FEATURE_COLUMNS = [
    'acousticness','danceability','energy','instrumentalness','liveness',
    'loudness','speechiness','tempo','valence','popularity',
    'avg_acousticness','avg_danceability','avg_energy','avg_instrumentalness',
    'avg_liveness','avg_loudness','avg_speechiness','avg_tempo',
    'avg_valence','avg_popularity','norm_num_tracks','norm_num_albums',
    'norm_num_followers','norm_modified_at'
]

# --- Step 1: Load data in chunks and balance it ---
chunksize = 100_000
positives = []
negatives = []

print("Starting to read and process dataset in chunks...")

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
    print(f"Processing chunk #{i+1}")
    
    chunk_pos = chunk[chunk['label'] == 1]
    chunk_neg = chunk[chunk['label'] == 0].sample(frac=0.005, random_state=42)

    positives.append(chunk_pos)
    negatives.append(chunk_neg)

    print(f"  -> Positives in chunk: {len(chunk_pos)}")
    print(f"  -> Negatives sampled: {len(chunk_neg)}")

print("Finished reading and sampling data.")
print("Combining all sampled data into a single DataFrame...")

# Combine the collected positive and negative samples
df_balanced = pd.concat(positives + negatives, ignore_index=True)
df_balanced = shuffle(df_balanced, random_state=42)

print(f"Total dataset size after balancing: {len(df_balanced)}")
print(f"Total positives: {df_balanced['label'].sum()}")
print(f"Total negatives: {len(df_balanced) - df_balanced['label'].sum()}")

# --- Step 2: Split and train model ---
X = df_balanced[FEATURE_COLUMNS]
y = df_balanced['label']

print("Splitting dataset into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight={0: 1, 1: 100},
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

print("Training complete. Evaluating model...")

# --- Step 3: Evaluate model ---
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
