
import tensorflow as tf

import train_config as cfg

MAX_PLAYLIST_LENGTH = cfg.MAX_PADDING # 375

def pad_up_to(t, max_in_dims=[1 ,MAX_PLAYLIST_LENGTH], constant_value=''):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_value)

def return_padded_tensors(data):
    
    a = pad_up_to(tf.reshape(data['track_name_pl'], shape=(1,-1)) , constant_value='')
    b = pad_up_to(tf.reshape(data['artist_name_pl'], shape=(1,-1)) , constant_value='')
    c = pad_up_to(tf.reshape(data['album_name_pl'], shape=(1,-1)) , constant_value='')
    d = pad_up_to(tf.reshape(data['track_uri_pl'], shape=(1, -1,)) , constant_value='')
    e = pad_up_to(tf.reshape(data['duration_ms_songs_pl'], shape=(1,-1)) , constant_value=-1.)
    f = pad_up_to(tf.reshape(data['artist_pop_pl'], shape=(1,-1)) , constant_value=-1.)
    g = pad_up_to(tf.reshape(data['artists_followers_pl'], shape=(1,-1)) , constant_value=-1.)
    h = pad_up_to(tf.reshape(data['track_pop_pl'], shape=(1,-1)) , constant_value=-1.)
    i = pad_up_to(tf.reshape(data['artist_genres_pl'], shape=(1,-1)) , constant_value='')
        
    padded_data = data.copy()
    padded_data['track_name_pl'] = a
    padded_data['artist_name_pl'] = b
    padded_data['album_name_pl'] = c
    padded_data['track_uri_pl'] = d
    padded_data['duration_ms_songs_pl'] = e
    padded_data['artist_pop_pl'] = f
    padded_data['artists_followers_pl'] = g
    padded_data['track_pop_pl'] = h
    padded_data['artist_genres_pl'] = i
        
    return padded_data

all_features = {
    'track_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'track_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'duration_ms_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'track_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_genres_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_followers_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'pos_seed_track': tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
    'track_name_seed_track': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_name_seed_track': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_name_seed_track': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'track_uri_seed_track': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_uri_seed_track': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_uri_seed_track': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'duration_seed_track': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'track_pop_seed_track': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_pop_seed_track': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_genres_seed_track': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_followers_seed_track': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'pid': tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
    'name': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'collaborative': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'duration_ms_seed_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'n_songs_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_artists_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_albums_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'description_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    ###ragged
    'track_name_pl': tf.io.RaggedFeature(tf.string),
    'artist_name_pl': tf.io.RaggedFeature(tf.string),
    'album_name_pl': tf.io.RaggedFeature(tf.string),
    'track_uri_pl': tf.io.RaggedFeature(tf.string),
    'duration_ms_songs_pl': tf.io.RaggedFeature(tf.float32),
    'artist_pop_pl': tf.io.RaggedFeature(tf.float32),
    'artists_followers_pl': tf.io.RaggedFeature(tf.float32),
    'track_pop_pl': tf.io.RaggedFeature(tf.float32),
    'artist_genres_pl': tf.io.RaggedFeature(tf.string),
}

def parse_tfrecord_fn(example, feature_dict=all_features): # =all_features
    example = tf.io.parse_single_example(
        example, 
        features=feature_dict
    )
    return example

candidate_features = {
    'track_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'track_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'duration_ms_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'track_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_genres_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_followers_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
}

def parse_candidate_tfrecord_fn(example, feature_dict=candidate_features):
    example = tf.io.parse_single_example(
        example, 
        features=feature_dict
    )
    return example
