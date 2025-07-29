from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

# --- Path to dataset ---
csv_path = "../data/processed_data/processed_dataset2.csv"

# Columns to use for modeling (original features + collaborative score)
FEATURE_COLUMNS = [
    'acousticness','danceability','energy','instrumentalness','liveness',
    'loudness','speechiness','tempo','valence','popularity',
    'avg_acousticness','avg_danceability','avg_energy','avg_instrumentalness',
    'avg_liveness','avg_loudness','avg_speechiness','avg_tempo',
    'avg_valence','avg_popularity','norm_num_tracks','norm_num_albums',
    'norm_num_followers','norm_modified_at','collab_similarity_score'
]

# --- Step 1: Load dataset ---
print("Loading dataset...")
df = pd.read_csv(csv_path)

# --- Step 2: Collaborative Filtering Computation ---
print("Preparing collaborative filtering data...")
positive_df = df[df['label'] == 1]

playlist_id_map = {pid: i for i, pid in enumerate(positive_df['playlist_id'].unique())}
track_id_map = {tid: i for i, tid in enumerate(positive_df['track_id'].unique())}

row_idx = positive_df['playlist_id'].map(playlist_id_map)
col_idx = positive_df['track_id'].map(track_id_map)
data = np.ones(len(positive_df))

playlist_track_matrix = csr_matrix((data, (row_idx, col_idx)),
                                   shape=(len(playlist_id_map), len(track_id_map)))

print("Computing track-track similarity...")
track_similarity = cosine_similarity(playlist_track_matrix.T)

def compute_collab_similarity(row):
    pid = row['playlist_id']
    tid = row['track_id']
    
    if pid not in playlist_id_map or tid not in track_id_map:
        return 0.0

    playlist_idx = playlist_id_map[pid]
    track_idx = track_id_map[tid]

    playlist_track_indices = playlist_track_matrix[playlist_idx].nonzero()[1]
    if track_idx in playlist_track_indices:
        playlist_track_indices = playlist_track_indices[playlist_track_indices != track_idx]

    if len(playlist_track_indices) == 0:
        return 0.0

    similarity_scores = track_similarity[track_idx, playlist_track_indices]
    return np.mean(similarity_scores)

print("Scoring dataset with collaborative filtering...")
df['collab_similarity_score'] = df.apply(compute_collab_similarity, axis=1)

# --- Step 3: Train/Test Split ---
print("Preparing data for training...")
df = shuffle(df)
X = df[FEATURE_COLUMNS]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train AdaBoost Classifier ---
print("Training AdaBoost model...")
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 5: Evaluate ---
print("Evaluating model...")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))