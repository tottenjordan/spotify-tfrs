
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)
@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.17.0',
        'tensorflow==2.9.2',
        'google-cloud-storage',
    ],
    # output_component_file="./pipelines/generate_candidate_embedding_index.yaml",
)
def generate_candidate_embedding_index(
    project: str,
    location: str,
    # version: str,
    model_version: str,
    pipeline_version: str,
    train_output_gcs_bucket: str,
    # data_bucket_name: str,
    model_dir: str,
    candidate_file_dir: str,
    candidate_files_prefix: str,
    # candidate_items_prefix: str,
    experiment_name: str,
    experiment_run: str,
    embedding_index_destination_gcs_uri: str,
    uploaded_candidate_model_resources: str,
) -> NamedTuple('Outputs', [('candidate_embedding_index_file_uri', str),
                            ('embedding_index_gcs_bucket', str),
]):
    
    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob
    
    from google.cloud import aiplatform as vertex_ai
    import tensorflow as tf
    from datetime import datetime
    import logging
    import os
    import numpy as np
    
    # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # initialize clients
    storage_client = storage.Client(project=project)
    vertex_ai.init(project=project,location=location)
    
    # ========================================================================
    # Helper Functions
    # ========================================================================
    
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

    def parse_candidate_tfrecord_fn(example):
        example = tf.io.parse_single_example(
            example, 
            features=candidate_features
        )
        return example
    
    # ========================================================================
    # Load Saved Model
    # ========================================================================
    # TODO: use model uploaded to Vertex Registry
    
    logging.info(f"loaded_candidate_tower from model_dir: {model_dir}")
    loaded_candidate_tower = tf.saved_model.load(model_dir)
    
    candidate_predictor = loaded_candidate_tower.signatures["serving_default"]
    
    # ========================================================================
    # Candidate Dataset
    # ========================================================================
    
    # data_bucket_name = 'spotify-beam-v3'
    # candidate_items_prefix = 'v3/candidates-jt-tmp/'

    candidate_files = []
    for blob in storage_client.list_blobs(f'{candidate_file_dir}', prefix=f'{candidate_files_prefix}', delimiter="/"):
        if blob.name[-9:] == 'tfrecords':
            candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
        else:
            pass
        
    raw_dataset = tf.data.TFRecordDataset(candidate_files)
    parsed_candidate_dataset = raw_dataset.map(parse_candidate_tfrecord_fn)
    
    # ========================================================================
    # Convert candidates to embeddings
    # ========================================================================
    
    embs_iter = parsed_candidate_dataset.batch(1).map(
        lambda data: candidate_predictor(
            artist_name_can = data["artist_name_can"],
            track_name_can = data['track_name_can'],
            album_name_can = data['album_name_can'],
            track_uri_can = data['track_uri_can'],
            artist_uri_can = data['artist_uri_can'],
            album_uri_can = data['album_uri_can'],
            duration_ms_can = data['duration_ms_can'],
            track_pop_can = data['track_pop_can'],
            artist_pop_can = data['artist_pop_can'],
            artist_followers_can = data['artist_followers_can'],
            artist_genres_can = data['artist_genres_can']
        )
    )

    embs = []
    for emb in embs_iter:
        embs.append(emb)

    print(f"Length of embs: {len(embs)}")
    
    cleaned_embs = [x['output_1'].numpy()[0] for x in embs] #clean up the output

    print(f"Length of cleaned_embs: {len(cleaned_embs)}")
    
    # ========================================================================
    # candidate track uri (IDs)
    # ========================================================================
    
    # clean product IDs
    track_uris = [x['track_uri_can'].numpy() for x in parsed_candidate_dataset]
    track_uris_cleaned = [str(z).replace("b'","").replace("'","") for z in track_uris]
    
    logging.info(f"Length of track_uris: {len(track_uris)}")
    logging.info(f"Length of track_uris_cleaned: {len(track_uris_cleaned)}")

    # ========================================================================
    # Remove bad records (e.g., nans)
    # ========================================================================
    
    bad_records = []

    for i, emb in enumerate(cleaned_embs):
        bool_emb = np.isnan(emb)
        for val in bool_emb:
            if val:
                bad_records.append(i)

    bad_record_filter = np.unique(bad_records)

    logging.info(f"bad_records: {len(bad_records)}")
    logging.info(f"bad_record_filter: {len(bad_record_filter)}")
    
    # ========================================================================
    # zip good and bad records seperately
    # ========================================================================
    
    track_uris_valid = []
    emb_valid = []

    bad_record_ids = []
    bad_record_embs = []

    for i, pair in enumerate(zip(track_uris_cleaned, cleaned_embs)):
        if i in bad_record_filter:
            t_uri, embed = pair
            bad_record_ids.append(t_uri)
            bad_record_embs.append(embed)
            # pass
        else:
            t_uri, embed = pair
            track_uris_valid.append(t_uri)
            emb_valid.append(embed)
            
    logging.info(f"Length of emb_valid             : {len(emb_valid)}")
    logging.info(f"Length of track_uris_valid      : {len(track_uris_valid)}")
    
    logging.info(f"Length of bad_record_ids        : {len(bad_record_ids)}")
    logging.info(f"Length of bad_record_embs       : {len(bad_record_embs)}")

    # ========================================================================
    # Write json file
    # ========================================================================

    embeddings_index_filename = f'candidate_embeddings_{model_version}_{experiment_run}.json'
    
    logging.info(f"Writting embedding vectors to file: {embeddings_index_filename}")

    with open(f'{embeddings_index_filename}', 'w') as f:
        for prod, emb in zip(track_uris_valid, emb_valid):
            f.write('{"id":"' + str(prod) + '",')
            f.write('"embedding":[' + ",".join(str(x) for x in list(emb)) + "]}")
            f.write("\n")
        

    # DESTINATION_BLOB_NAME = embeddings_index_filename
    # SOURCE_FILE_NAME = embeddings_index_filename

    # ========================================================================
    # Upload vocab_dict to GCS
    # ========================================================================
    # spotify-tfrs/candidate_embeddings_local_v2_092122.json
    # bucket = storage_client.bucket(gcs_bucket_name)
    # candidate_tower_dir_uri = f"gs://{output_dir_gcs_bucket_name}/{experiment_name}/{experiment_run}/model-dir/candidate_tower/"
    
    EMBEDDING_INDEX_DIR_GCS_URI = f'gs://{train_output_gcs_bucket}/{experiment_name}/{experiment_run}/candidate-embeddings/corpus-index-dir'
    
    logging.info(f"Uploading emebdding vectors to : {EMBEDDING_INDEX_DIR_GCS_URI}/{embeddings_index_filename}")
    blob = Blob.from_string(os.path.join(EMBEDDING_INDEX_DIR_GCS_URI, embeddings_index_filename))
    blob.bucket._client = storage_client
    blob.upload_from_filename(embeddings_index_filename)
    
    embedding_index_file_uri = f'{embedding_index_destination_gcs_uri}/{embeddings_index_filename}'
    logging.info(f"Saved embedding vectors for candidate corpus: {embedding_index_file_uri}")
    
    # Note: TODO: considerations for storing files used to create index and index updates
    
    return (
        f'{embedding_index_file_uri}',
        f'{EMBEDDING_INDEX_DIR_GCS_URI}',
    )
