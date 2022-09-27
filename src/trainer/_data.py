
import tensorflow as tf

import train_config as cfg

MAX_PLAYLIST_LENGTH = cfg.MAX_PADDING # 375

def pad_up_to(t, max_in_dims=[1 ,MAX_PLAYLIST_LENGTH], constant_value=''):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_value)

def return_padded_tensors(context, data):
    
    a = data['track_name_pl'].to_tensor(default_value='', shape=[None, MAX_PLAYLIST_LENGTH]), 
    b = data['artist_name_pl'].to_tensor(default_value='', shape=[None, MAX_PLAYLIST_LENGTH]), 
    c = data['album_name_pl'].to_tensor(default_value='', shape=[None, MAX_PLAYLIST_LENGTH]), 
    d = data['track_uri_pl'].to_tensor(default_value='', shape=[None, MAX_PLAYLIST_LENGTH]), 
    e = data['duration_ms_songs_pl'].to_tensor(default_value=-1., shape=[None, MAX_PLAYLIST_LENGTH]), 
    f = data['artist_pop_pl'].to_tensor(default_value=-1., shape=[None, MAX_PLAYLIST_LENGTH]), 
    g = data['artists_followers_pl'].to_tensor(default_value=-1., shape=[None, MAX_PLAYLIST_LENGTH]), 
    h = data['track_pop_pl'].to_tensor(default_value=-1., shape=[None, MAX_PLAYLIST_LENGTH]), 
    i = data['artist_genres_pl'].to_tensor(default_value='', shape=[None, MAX_PLAYLIST_LENGTH]), 
        
    padded_data = context.copy()
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

cont_feats = {
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
    'name': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'collaborative': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'n_songs_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_artists_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_albums_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'description_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
}
    ###ragged

seq_feats = {
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

def parse_tfrecord(example):
    example = tf.io.parse_single_sequence_example(
        example, 
        context_features=cont_feats,
        sequence_features=seq_feats
    )
    return example

def parse_candidate_tfrecord_fn(example):
    example = tf.io.parse_single_example(
        example, 
        features=candidate_features
    )
    return example
