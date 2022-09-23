
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)
@kfp.v2.dsl.component(
    base_image='python:3.9',
    packages_to_install=[
        'google-cloud-aiplatform==1.17.0',
        'tensorflow==2.9.2',
        'tensorflow-recommenders==0.7.0',
        'numpy',
        'google-cloud-storage',
    ],
    # output_component_file="./pipelines/build_vocabs_stats.yaml",
)
def build_vocabs_string_lookups(
    project: str,
    location: str,
    # version: str,
    model_version: str,
    pipeline_version: str,
    train_output_gcs_bucket: str,
    train_dir: str,
    train_dir_prefix: str,
    valid_dir: str,
    valid_dir_prefix: str,
    candidate_file_dir: str,
    candidate_files_prefix: str,
    experiment_name: str,
    experiment_run: str, 
    # gcs_bucket_name: str,
    # path_to_train_dir: str,
    # path_to_valid_dir: str,
    # path_to_candidate_dir: str,
    path_to_vocab_file: str,
    vocab_file_name: str,
    generate_vocabs_stats: bool,
    max_padding: int = 375,
    split_names: list = ['train', 'valid'],
) -> NamedTuple('Outputs', [
                            ('vocab_dict', Artifact),
                            ('vocab_gcs_filename', str),
                            ('vocab_gcs_sub_dir', str),
                            ('vocab_gcs_uri', str),
]):
    
    from google.cloud import aiplatform as vertex_ai
    import json
    import logging
    import json
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    import numpy as np
    import pickle as pkl
    from google.cloud import storage
    from datetime import datetime
    import numpy as np
    import string

    # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    # logging.info(f'TIMESTAMP: {TIMESTAMP}')
    
    MAX_PLAYLIST_LENGTH = max_padding       # 375
    logging.info(f'MAX_PLAYLIST_LENGTH: {MAX_PLAYLIST_LENGTH}')

    vertex_ai.init(
        project=project,
        location=location,
    )
    
    ################################################################################
    # Helper Functions for feature parsing
    ################################################################################
    
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
    
    def parse_candidate_tfrecord_fn(example, feature_dict=candidate_features):
        example = tf.io.parse_single_example(
            example, 
            features=feature_dict
        )
        return example
    
    def parse_tfrecord_fn(example, feature_dict=all_features): # =all_features
        example = tf.io.parse_single_example(
            example, 
            features=feature_dict
        )
        return example

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
    
    storage_client = storage.Client()
    
    if generate_vocabs_stats == False:
        VOCAB_FILENAME = f'{vocab_file_name}'
        VOCAB_GCS_SUB_DIR = f'{path_to_vocab_file}'
        VOCAB_GCS_URI = f'gs://{train_output_gcs_bucket}/{VOCAB_GCS_SUB_DIR}/{VOCAB_FILENAME}'
        
        logging.info(f"Loading Vocab File: {VOCAB_FILENAME} from {VOCAB_GCS_URI}")
        
        with open(f'{VOCAB_FILENAME}', 'wb') as file_obj:
            storage_client.download_blob_to_file(
                f'gs://{VOCAB_GCS_URI}', file_obj)


        with open(f'{VOCAB_FILENAME}', 'rb') as pickle_file:
            vocab_dict = pkl.load(pickle_file)
        
    else:
        # ========================================================================
        # Candidate dataset
        # ========================================================================

        candidate_files = []
        for blob in storage_client.list_blobs(f'{candidate_file_dir}', prefix=f'{candidate_files_prefix}', delimiter="/"):
            candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

        raw_candidate_dataset = tf.data.TFRecordDataset(candidate_files)
        parsed_candidate_dataset = raw_candidate_dataset.map(parse_candidate_tfrecord_fn)

        '''
        Get vocabularies of unique values for strings
        '''
        logging.info(f"Getting unqiue values for `artist_name_can`...")
        v_artist_name_can = np.unique(
          np.concatenate(
              list(
                  parsed_candidate_dataset.map(
                      lambda x: x['artist_name_can']
                  )
                  .batch(1000)
              )
          )
        )
        v_artist_name_can_cleaned = [str(z).replace("b'","").translate(str.maketrans('', '', string.punctuation)) for z in v_artist_name_can]
    
        # v_track_uri_can = np.unique(
        #   np.concatenate(
        #       list(
        #           parsed_candidate_dataset.map(
        #               lambda x: x['track_uri_can']
        #           )
        #           .batch(1000)
        #       )
        #   )
        # )
        # v_track_uri_can_cleaned = [str(z).replace("b'","").translate(str.maketrans('', '', string.punctuation)) for z in v_track_uri_can]
        
        logging.info(f"Getting unqiue values for `track_name_can`...") 
        v_track_name_can = np.unique(
          np.concatenate(
              list(
                  parsed_candidate_dataset.map(
                      lambda x: x['track_name_can']
                  )
                  .batch(1000)
              )
          )
        )
        v_track_name_can_cleaned = [str(z).replace("b'","").translate(str.maketrans('', '', string.punctuation)) for z in v_track_name_can]
    
        logging.info(f"Getting unqiue values for `album_name_can`...")
        v_album_name_can = np.unique(
          np.concatenate(
              list(
                  parsed_candidate_dataset.map(
                      lambda x: x['album_name_can']
                  )
                  .batch(1000)
              )
          )
        )
        v_album_name_can_cleaned = [str(z).replace("b'","").translate(str.maketrans('', '', string.punctuation)) for z in v_album_name_can]

        logging.info(f"Getting unqiue values for `artist_genres_can`...")
        v_artist_genres_can = np.unique(
          np.concatenate(
              list(
                  parsed_candidate_dataset.map(
                      lambda x: x['artist_genres_can']
                  )
                  .batch(1000)
              )
          )
        )
        v_artist_genres_can_cleaned = [str(z).replace("b'","").translate(str.maketrans('', '', string.punctuation)) for z in v_artist_genres_can]

        # ========================================================================
        # TODO: parameterize / automate
        # ========================================================================
        avg_duration_ms_seed_pl = 13000151.68
        var_duration_ms_seed_pl = 133092900971233.58

        avg_n_songs_pl = 55.21
        var_n_songs_pl = 2317.54

        avg_n_artists_pl = 30.56
        var_n_artists_pl = 769.26

        avg_n_albums_pl = 40.25
        var_n_albums_pl = 1305.54

        avg_artist_pop = 16.08
        var_artist_pop = 300.64

        avg_duration_ms_songs_pl = 234823.14
        var_duration_ms_songs_pl = 5558806228.41

        avg_artist_followers = 43337.77
        var_artist_followers = 377777790193.57

        avg_track_pop = 10.85
        var_track_pop = 202.18

        # ========================================================================
        # Create Vocab Dict
        # ========================================================================
    
        vocab_dict = {
            'artist_name_can': v_artist_name_can_cleaned,
            'track_name_can': v_track_name_can_cleaned,
            'album_name_can': v_album_name_can_cleaned,
            'artist_genres_can': v_artist_genres_can_cleaned,
            'avg_duration_ms_seed_pl': avg_duration_ms_seed_pl,
            'var_duration_ms_seed_pl': var_duration_ms_seed_pl,
            'avg_n_songs_pl': avg_n_songs_pl,
            'var_n_songs_pl': var_n_songs_pl,
            'avg_n_artists_pl': avg_n_artists_pl,
            'var_n_artists_pl': var_n_artists_pl,
            'avg_n_albums_pl': avg_n_albums_pl,
            'var_n_albums_pl': var_n_albums_pl,
            'avg_artist_pop': avg_artist_pop,
            'var_artist_pop': var_artist_pop,
            'avg_duration_ms_songs_pl': avg_duration_ms_songs_pl,
            'var_duration_ms_songs_pl': var_duration_ms_songs_pl,
            'avg_artist_followers': avg_artist_followers,
            'var_artist_followers': var_artist_followers,
            'avg_track_pop': avg_track_pop,
            'var_track_pop': var_track_pop,
        }
        VOCAB_FILENAME = f'vocab_dict_{experiment_run}.pkl'
        VOCAB_GCS_SUB_DIR = f'{experiment_name}/{experiment_run}/vocabs_stats'
        VOCAB_GCS_URI = f'gs://{train_output_gcs_bucket}/{VOCAB_GCS_SUB_DIR}/{VOCAB_FILENAME}'
        
        logging.info(f"Saving Vocab File: {VOCAB_FILENAME} to {VOCAB_GCS_URI}")
        
        # Upload vocab_dict to GCS
        bucket = storage_client.bucket(train_output_gcs_bucket)
        blob = bucket.blob(f'{VOCAB_GCS_SUB_DIR}/{VOCAB_FILENAME}')
        pickle_out = pkl.dumps(vocab_dict)
        blob.upload_from_string(pickle_out)
        
        return (
            vocab_dict,
            f'{VOCAB_FILENAME}',
            f'{VOCAB_GCS_SUB_DIR}',
            f'{VOCAB_GCS_URI}',
        )
