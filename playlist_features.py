import pandas as pd



df=pd.read_csv('data/processed_data/processed_dataset2.csv')
#group by 'playlist_id' and calculate the mean for each column
playlist_features = df.groupby('playlist_id')[['acousticness', 'danceability', 'energy', 'instrumentalness', 
                                               'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity']].mean()


playlist_features.reset_index(inplace=True)


playlist_features.to_csv('playlist_features_mean.csv', index=False)


print(playlist_features.head())
