from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

# Path to data.
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

# --- Step 2: Split the dataset into train/test sets ---
X = df_balanced[FEATURE_COLUMNS]
y = df_balanced['label']

print("Splitting dataset into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- Step 3: Loop over different n_estimators and train the model ---
n_estimators_values = [50, 100, 200, 300, 500]  # List of n_estimators to test
results = []  # Store the results

for n_estimators in n_estimators_values:
    print(f"\nTraining the model with n_estimators = {n_estimators}")
    
    # Initialize the model
    weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)
    adaboost_clf = AdaBoostClassifier(
        estimator=weak_learner,
        n_estimators=n_estimators,
        random_state=42,
    )

    # Train the model
    adaboost_clf.fit(X_train, y_train)

    # Evaluate the model
    print(f"Evaluating the model with n_estimators = {n_estimators}")
    y_pred = adaboost_clf.predict(X_test)

    # Calculate accuracy and classification report
    accuracy = metrics.accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Store the results
    results.append({
        'n_estimators': n_estimators,
        'accuracy': accuracy,
        'classification_report': class_report
    })

# --- Step 4: Display the results ---
print("\nResults for different n_estimators:")
for result in results:
    print(f"n_estimators: {result['n_estimators']}")
    print(f"Accuracy: {result['accuracy']}")
    print(f"Classification Report:\n{result['classification_report']}")
    print("-" * 50)
