import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
df_songs = pd.read_csv('data/processed_data/processed_dataset2.csv')  # Original song dataset
df_playlists = pd.read_csv('playlist_features_mean.csv')  # Aggregated playlist features

# Step 1: Create a user-item matrix (playlist_id vs. track_id)
# Create a pivot table with playlist_id as rows, track_id as columns, and label as values
pivot_table = df_songs.pivot(index='playlist_id', columns='track_id', values='label').fillna(0)

# Step 2: Calculate the cosine similarity between songs (columns in the pivot table)
song_similarity = cosine_similarity(pivot_table.T)  # Transpose to calculate similarity between songs

# Step 3: Create a DataFrame for the song similarity matrix
song_similarity_df = pd.DataFrame(song_similarity, index=pivot_table.columns, columns=pivot_table.columns)

# Step 4: Function to predict whether a song belongs to a playlist
def predict_song_belongs_to_playlist(track_id, playlist_id, pivot_table, song_similarity_df):
    # Get the songs in the playlist
    playlist_songs = pivot_table.loc[playlist_id]
    
    # Get the similarity scores for the given song with the playlist's songs
    song_similarities = song_similarity_df[track_id]
    
    # Get the weighted sum of similarities for the songs in the playlist
    weighted_sum = 0
    total_weight = 0
    
    for song, label in playlist_songs.items():
        if label == 1:  # Only consider songs that are part of the playlist
            weighted_sum += song_similarities[song]
            total_weight += 1
    
    # Calculate the predicted score
    predicted_score = weighted_sum / total_weight if total_weight > 0 else 0
    
    # Predict if the song should belong to the playlist (threshold based on similarity)
    return 1 if predicted_score > 0.5 else 0

# Step 5: Function to evaluate predictions for all songs in each playlist
def evaluate_playlist_predictions(playlist_id, pivot_table, song_similarity_df):
    # Get all songs in the specified playlist (label = 1)
    df_playlist_1 = df_songs[(df_songs['playlist_id'] == playlist_id) & (df_songs['label'] == 1)]
    
    # Get some negative examples (label = 0) from the playlist
    df_playlist_0 = df_songs[(df_songs['playlist_id'] == playlist_id) & (df_songs['label'] == 0)]
    
    # Limit the size of the negative sample to balance positive/negative examples
    df_playlist_0_sample = df_playlist_0.sample(min(50, len(df_playlist_0)), random_state=42)
    
    # Combine the positive (label=1) and negative (label=0) samples
    df_subset = pd.concat([df_playlist_1[['playlist_id', 'track_id', 'label']], df_playlist_0_sample[['playlist_id', 'track_id', 'label']]])

    # Prepare actual labels and predictions
    y_true = df_subset['label'].values
    y_pred = []
    
    # Predict for all songs in the subset (both positive and negative examples)
    for _, row in df_subset.iterrows():
        track_id = row['track_id']
        prediction = predict_song_belongs_to_playlist(track_id, playlist_id, pivot_table, song_similarity_df)
        y_pred.append(prediction)
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)  # Avoid division by zero
    
    return accuracy, report

# Step 6: Loop over all playlists and evaluate
def evaluate_all_playlists(pivot_table, song_similarity_df):
    all_accuracies = []
    all_reports = []
    
    # Loop through all unique playlist IDs
    for playlist_id in pivot_table.index:
        accuracy, report = evaluate_playlist_predictions(playlist_id, pivot_table, song_similarity_df)
        all_accuracies.append(accuracy)
        all_reports.append(report)
    
    # Calculate the overall accuracy and average report
    average_accuracy = np.mean(all_accuracies)
    
    print(f'Average accuracy across all playlists: {average_accuracy}')
    
    return all_reports

# Example: Evaluate all playlists and print results
all_reports = evaluate_all_playlists(pivot_table, song_similarity_df)

# # Optionally: If you want to print out the detailed classification report for each playlist, you can loop through all_reports
# for i, report in enumerate(all_reports):
#     print(f"Classification Report for Playlist {i}:")
#     print(report)


