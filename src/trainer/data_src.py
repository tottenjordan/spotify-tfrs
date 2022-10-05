
import tensorflow as tf

import train_config as cfg

MAX_PLAYLIST_LENGTH = cfg.MAX_PADDING # 375

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

feats = {
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
    'name': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'collaborative': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'n_songs_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_artists_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_albums_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'description_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'track_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    'artist_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    'album_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    'track_uri_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH)),
    'duration_ms_songs_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'artist_pop_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'artists_followers_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'track_pop_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'artist_genres_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
}

def parse_tfrecord(example):
    example = tf.io.parse_example(
        example, 
        feats
    )
    return example


def parse_candidate_tfrecord_fn(example):
    example = tf.io.parse_single_example(
        example, 
        features=candidate_features
    )
    return example
