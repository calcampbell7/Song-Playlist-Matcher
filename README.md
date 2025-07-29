# playlist-matcher

## Installation

```bash
git clone https://github.com/jhideki/playlist-matcher.git
pip install -r requirements.txt
```

You may want to use a python venv to avoid installing deps globally.

### 📂 Project setup

#### Downloading Datasets

Run the following script to download the datasets

```bash
cd src/data
python data.py
```

Or alternatively you can download the datasets directly from:

[spotify_million](https://www.kaggle.com/datasets/himanshuwagh/spotify-million)  
[Spotify Audio Features](https://www.kaggle.com/datasets/tomigelo/spotify-audio-features)

Place the data files in the proper paths:

```
project-root/
├── data/
│   └── kagle/
│       ├── 1/
│       │   └── data/mpd.slice.*.json        # Playlist data (MPD slices)
│       └── 3/
│           └── SpotifyAudioFeaturesApril2019.csv  # Audio features CSV
```

1 -> spotify_million dataset
3 -> Spotify Audio Features dataset

To run the data processing module, ensure that rust is installed:
[Rust installation](https://www.rust-lang.org/tools/install)

You can also install the processed dataset from this link (alternatively you can reprocess the data via the `Data Preproecessing with Rust` section):  
[processed dataset](https://www.mediafire.com/file/ngmqdzi54uykv5c/processed_dataset.csv/file)

#### Data Preprocessing with Rust

To efficiently process large-scale Spotify data, this project includes a Rust-based preprocessing tool that prepares a training-ready dataset for the ML playlist-matching model.

- Loads Spotify playlist JSON slices (Million Playlist Dataset) and audio feature CSV data
- Filters playlists to only include songs present in the feature dataset
- For each playlist, creates a labeled row (`label = 1` if song is in playlist, otherwise `0`)
- Outputs a single CSV suitable for binary classification model training

#### Output

Processed dataset will be saved to:

```
data/processed_data/processed_dataset.csv
```

Each row contains:

- `playlist_id`
- `track_id`
- `label` (1 if track is in playlist, 0 otherwise)
- Audio features: `danceability`, `energy`, `valence`, etc.
- Playlist metadata (normlized)

#### Usage

```bash
cd processor/

# To process the data and create the final dataset
cargo run -- p

# To count how many rows were written
cargo run -- c
```

## Model Training

TODO
