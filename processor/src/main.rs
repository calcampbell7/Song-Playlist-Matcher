use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::atomic::{self, AtomicUsize},
    thread,
};

use crossbeam_channel::unbounded;
use csv::Writer;
use glob::glob;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct AudioFeature {
    track_id: String,
    acousticness: f32,
    danceability: f32,
    duration_ms: u32,
    energy: f32,
    instrumentalness: f32,
    key: u8,
    liveness: f32,
    loudness: f32,
    mode: u8,
    speechiness: f32,
    tempo: f32,
    valence: f32,
    popularity: u8,
}

#[derive(Serialize)]
struct OutputRow {
    playlist_id: u32,
    track_id: String,
    label: u8,
    acousticness: f32,
    danceability: f32,
    energy: f32,
    instrumentalness: f32,
    liveness: f32,
    loudness: f32,
    speechiness: f32,
    tempo: f32,
    valence: f32,
    popularity: u8,
    avg_acousticness: f32,
    avg_danceability: f32,
    avg_energy: f32,
    avg_instrumentalness: f32,
    avg_liveness: f32,
    avg_loudness: f32,
    avg_speechiness: f32,
    avg_tempo: f32,
    avg_valence: f32,
    avg_popularity: f32,
    norm_num_tracks: f32,
    norm_num_albums: f32,
    norm_num_followers: f32,
    norm_modified_at: f32,
}

fn normalize(value: f32, min: f32, max: f32) -> f32 {
    if max > min {
        (value - min) / (max - min)
    } else {
        0.0
    }
}

fn process_dataset(features_path: &str, output_file: &str) -> std::io::Result<()> {
    let mpd_glob = "../data/kaggle/1/data/mpd.slice.*.json";

    if Path::new(output_file).exists() {
        fs::remove_file(output_file)?;
    }

    // Load audio features
    println!("Loading audio features...");
    let mut rdr = csv::Reader::from_path(features_path)?;
    let mut audio_map: HashMap<String, AudioFeature> = HashMap::default();
    for result in rdr.deserialize() {
        let record: AudioFeature = result?;
        audio_map.insert(record.track_id.clone(), record);
    }

    println!("Audio feature count: {}", audio_map.len());

    let json_files: Vec<_> = glob(mpd_glob)
        .expect("Error globbing")
        .filter_map(Result::ok)
        .collect();

    let (min_time, max_time) = (1.0e9, 1.6e9);
    let (min_tracks, max_tracks) = (1.0, 250.0);
    let (min_albums, max_albums) = (1.0, 200.0);
    let (min_followers, max_followers) = (0.0, 1000.0);

    let track_intersection = AtomicUsize::new(0);

    let (sender, receiver) = unbounded();

    let output_file = output_file.to_string();
    let writer_handle = thread::spawn(move || {
        let mut output_writer = Writer::from_path(output_file).expect("Failed to open output CSV");
        output_writer
            .serialize(vec![
                "playlist_id",
                "track_id",
                "label",
                "acousticness",
                "danceability",
                "energy",
                "instrumentalness",
                "liveness",
                "loudness",
                "speechiness",
                "tempo",
                "valence",
                "popularity",
                "avg_acousticness",
                "avg_danceability",
                "avg_energy",
                "avg_instrumentalness",
                "avg_liveness",
                "avg_loudness",
                "avg_speechiness",
                "avg_tempo",
                "avg_valence",
                "avg_popularity",
                "norm_num_tracks",
                "norm_num_albums",
                "norm_num_followers",
                "norm_modified_at",
            ])
            .unwrap();

        for row in receiver {
            output_writer.serialize(row).unwrap();
        }
    });

    for file_path in &json_files {
        println!("Processing file: {}", file_path.display());
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let json: Value = serde_json::from_reader(reader)?;

        if let Some(playlists) = json.get("playlists").and_then(|v| v.as_array()) {
            for playlist in playlists {
                let playlist_id = playlist.get("pid").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

                println!("processing plalist {}", playlist_id);
                let tracks = playlist.get("tracks").and_then(|v| v.as_array()).expect("");
                let num_tracks = playlist
                    .get("num_tracks")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as f32;
                let num_albums = playlist
                    .get("num_albums")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as f32;
                let num_followers = playlist
                    .get("num_followers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as f32;
                let modified_at = playlist
                    .get("modified_at")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as f32;

                let norm_num_tracks = normalize(num_tracks, min_tracks, max_tracks);
                let norm_num_albums = normalize(num_albums, min_albums, max_albums);
                let norm_num_followers = normalize(num_followers, min_followers, max_followers);
                let norm_modified_at = normalize(modified_at, min_time, max_time);

                let mut included: HashSet<&str> = HashSet::default();
                let mut num_songs = 0;

                let mut avg_acousticness = 0.0;
                let mut avg_danceability = 0.0;
                let mut avg_energy = 0.0;
                let mut avg_instrumentalness = 0.0;
                let mut avg_liveness = 0.0;
                let mut avg_loudness = 0.0;
                let mut avg_speechiness = 0.0;
                let mut avg_tempo = 0.0;
                let mut avg_valence = 0.0;
                let mut avg_popularity = 0.0;
                for track in tracks {
                    if let Some(uri) = track.get("track_uri").and_then(|v| v.as_str()) {
                        if let Some(track_id) = uri.strip_prefix("spotify:track:") {
                            if let Some(song) = audio_map.get(track_id) {
                                track_intersection.fetch_add(1, atomic::Ordering::Relaxed);
                                included.insert(track_id);

                                num_songs += 1;

                                avg_acousticness += song.acousticness;
                                avg_danceability += song.danceability;
                                avg_energy += song.energy;
                                avg_instrumentalness += song.instrumentalness;
                                avg_liveness += song.liveness;
                                avg_loudness += song.loudness;
                                avg_speechiness += song.speechiness;
                                avg_tempo += song.tempo;
                                avg_valence += song.valence;
                                avg_popularity += song.popularity as f32;
                            }
                        }
                    }
                }

                if num_songs == 0 {
                    println!("num songs is 0");
                    continue;
                }

                avg_acousticness /= num_songs as f32;
                avg_danceability /= num_songs as f32;
                avg_energy /= num_songs as f32;
                avg_instrumentalness /= num_songs as f32;
                avg_liveness /= num_songs as f32;
                avg_loudness /= num_songs as f32;
                avg_speechiness /= num_songs as f32;
                avg_tempo /= num_songs as f32;
                avg_valence /= num_songs as f32;
                avg_popularity = avg_popularity as f32 / num_songs as f32;

                for (track_id, feat) in &audio_map {
                    let label = if included.contains(track_id.as_str()) {
                        1
                    } else {
                        0
                    };

                    let row = OutputRow {
                        playlist_id,
                        track_id: track_id.clone(),
                        label,
                        acousticness: feat.acousticness,
                        danceability: feat.danceability,
                        energy: feat.energy,
                        instrumentalness: feat.instrumentalness,
                        liveness: feat.liveness,
                        loudness: feat.loudness,
                        speechiness: feat.speechiness,
                        tempo: feat.tempo,
                        valence: feat.valence,
                        popularity: feat.popularity,
                        avg_acousticness,
                        avg_danceability,
                        avg_energy,
                        avg_instrumentalness,
                        avg_liveness,
                        avg_loudness,
                        avg_speechiness,
                        avg_tempo,
                        avg_valence,
                        avg_popularity,
                        norm_num_tracks,
                        norm_num_albums,
                        norm_num_followers,
                        norm_modified_at,
                    };

                    sender.send(row).unwrap();
                }
            }
        }
    }

    println!(
        "Num of matching tracks: {}",
        track_intersection.load(atomic::Ordering::Relaxed)
    );

    drop(sender);
    writer_handle.join().unwrap();

    println!("Done.");
    Ok(())
}

fn check_output_file() -> std::io::Result<()> {
    let output_file = "../data/processed_data/processed_dataset2.csv";
    let mut rdr = csv::Reader::from_path(output_file)?;
    let mut total_rows = 0;
    let mut true_label_count = 0;

    for result in rdr.records() {
        if result.is_err() {
            println!("Number of rows (excluding header): {}", total_rows);
            println!("Number of rows with label = 1: {}", true_label_count);
            remove_last_line(output_file)?;
        }
        let record = result?;
        total_rows += 1;
        if let Some(label_str) = record.get(2) {
            if label_str.trim() == "1" {
                true_label_count += 1;
            }
        }
    }

    println!("Number of rows (excluding header): {}", total_rows);
    println!("Number of rows with label = 1: {}", true_label_count);

    Ok(())
}
fn remove_last_line(path: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;

    let mut pos = file.seek(SeekFrom::End(0))?;

    if pos == 0 {
        // File is empty
        return Ok(());
    }

    // Walk backwards to find the second-to-last newline
    let mut buffer = [0u8; 1];
    while pos > 0 {
        pos -= 1;
        file.seek(SeekFrom::Start(pos))?;
        file.read_exact(&mut buffer)?;

        if buffer[0] == b'\n' && pos != file.seek(SeekFrom::End(0))? - 1 {
            break;
        }
    }

    file.set_len(pos)?;
    println!("Succesfully removed last line");
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "p" => process_dataset(
                "../data/spotify_songs2.csv",
                "../data/processed_data/processed_dataset2.csv",
            ),
            "c" => check_output_file(),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Invalid CLI command",
            )),
        }
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Missing CLI command",
        ))
    }
}
