
import json
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.python.client import device_lib

import argparse
import os
import sys
import logging
import pickle as pkl

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage
import hypertune

import time
import numpy as np

# ====================================================
# Helper functions
# ====================================================

def _is_chief(task_type, task_id): 
    ''' Check for primary if multiworker training
    '''
    return (task_type == 'chief') or (task_type == 'worker' and task_id == 0) or task_type is None

def get_arch_from_string(arch_string):
    q = arch_string.replace(']', '')
    q = q.replace('[', '')
    q = q.replace(" ", "")
    return [int(x) for x in q.split(',')]

# ====================================================
# Main
# ====================================================
import _data
import _model
import train_config as config
import time 

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

def main(args):
    
    logging.info("Starting training...")
    logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
    
    storage_client = storage.Client(
        project=args.project
    )
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    # AIP_TB_LOGS = args.aip_tb_logs # os.environ.get('AIP_TENSORBOARD_LOG_DIR', 'NA')
    # logging.info(f'AIP TENSORBOARD LOG DIR: {AIP_TB_LOGS}')
    
    # ====================================================
    # Set Device / GPU/TPU Strategy
    # ====================================================
    logging.info("Detecting devices....")
    logging.info(f'Detected Devices {str(device_lib.list_local_devices())}')
    logging.info("Setting device strategy...")
    
    # Single Machine, single compute device
    if args.distribute == 'single':
        if tf.config.list_physical_devices('GPU'): # TODO: replace with - tf.config.list_physical_devices('GPU') | tf.test.is_gpu_available()
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        logging.info("Single device training")
    
    # Single Machine, multiple compute device
    elif args.distribute == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Mirrored Strategy distributed training")

    # Multi Machine, multiple compute device
    elif args.distribute == 'multiworker':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logging.info("Multi-worker Strategy distributed training")
        logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
    
    # Single Machine, multiple TPU devices
    elif args.distribute == 'tpu':
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

    
    logging.info('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))
    NUM_REPLICAS = strategy.num_replicas_in_sync
    
    # ====================================================
    # Vocab Files
    # ====================================================

    # TODO: parameterize & configure for adapts vs vocab files

    BUCKET_NAME = 'spotify-v1'
    FILE_PATH = 'vocabs/v1_string_vocabs'
    FILE_NAME = 'string_vocabs_v1_20220705-202905.txt'
    DESTINATION_FILE = 'downloaded_vocabs.txt'

    with open(f'{DESTINATION_FILE}', 'wb') as file_obj:
        storage_client.download_blob_to_file(
            f'gs://{BUCKET_NAME}/{FILE_PATH}/{FILE_NAME}', file_obj)


    with open(f'{DESTINATION_FILE}', 'rb') as pickle_file:
        vocab_dict_load = pkl.load(pickle_file)


    # TODO: include as a preprocessing step 
    avg_duration_ms_seed_pl = 13000151.68
    var_duration_ms_seed_pl = 133092900971233.58
    vocab_dict_load['avg_duration_ms_seed_pl']=avg_duration_ms_seed_pl
    vocab_dict_load['var_duration_ms_seed_pl']=var_duration_ms_seed_pl

    avg_n_songs_pl = 55.21
    var_n_songs_pl = 2317.54
    vocab_dict_load['avg_n_songs_pl']=avg_n_songs_pl
    vocab_dict_load['var_n_songs_pl']=var_n_songs_pl

    avg_n_artists_pl = 30.56
    var_n_artists_pl = 769.26
    vocab_dict_load['avg_n_artists_pl']=avg_n_artists_pl
    vocab_dict_load['var_n_artists_pl']=var_n_artists_pl

    avg_n_albums_pl = 40.25
    var_n_albums_pl = 1305.54
    vocab_dict_load['avg_n_albums_pl']=avg_n_albums_pl
    vocab_dict_load['var_n_albums_pl']=var_n_albums_pl

    avg_artist_pop = 16.08
    var_artist_pop = 300.64
    vocab_dict_load['avg_artist_pop']=avg_artist_pop
    vocab_dict_load['var_artist_pop']=var_artist_pop

    avg_duration_ms_songs_pl = 234823.14
    var_duration_ms_songs_pl = 5558806228.41
    vocab_dict_load['avg_duration_ms_songs_pl']=avg_duration_ms_songs_pl
    vocab_dict_load['var_duration_ms_songs_pl']=var_duration_ms_songs_pl

    avg_artist_followers = 43337.77
    var_artist_followers = 377777790193.57
    vocab_dict_load['avg_artist_followers']=avg_artist_followers
    vocab_dict_load['var_artist_followers']=var_artist_followers

    avg_track_pop = 10.85
    var_track_pop = 202.18
    vocab_dict_load['avg_track_pop']=avg_track_pop
    vocab_dict_load['var_track_pop']=var_track_pop

    # ====================================================
    # Parse & Pad train dataset
    # ====================================================

    logging.info(f'args.train_dir: {args.train_dir}')
    logging.info(f'args.train_dir_prefix: {args.train_dir_prefix}')
    
    train_files = []
    for blob in storage_client.list_blobs(f'{args.train_dir}', prefix=f'{args.train_dir_prefix}', delimiter="/"):
        train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
        
    # Parse train dataset
    raw_train_dataset = tf.data.TFRecordDataset(train_files)
    parsed_dataset = raw_train_dataset.map(_data.parse_tfrecord_fn)
    parsed_dataset_padded = parsed_dataset.map(_data.return_padded_tensors)  
    
    # ====================================================
    # Parse candidates dataset
    # ====================================================

    logging.info(f'args.candidate_file_dir: {args.candidate_file_dir}')
    logging.info(f'args.candidate_files_prefix: {args.candidate_files_prefix}')

    candidate_files = []
    for blob in storage_client.list_blobs(f'{args.candidate_file_dir}', prefix=f'{args.candidate_files_prefix}', delimiter="/"):
        candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
        
    raw_candidate_dataset = tf.data.TFRecordDataset(candidate_files)
    parsed_candidate_dataset = raw_candidate_dataset.map(_data.parse_candidate_tfrecord_fn)
    
    # ====================================================
    # Prepare Train and Valid Data
    # ====================================================
    logging.info(f'preparing train and valid splits...')
    tf.random.set_seed(42)
    shuffled_parsed_ds = parsed_dataset_padded.shuffle(10_000, seed=42, reshuffle_each_iteration=False)

    # train_data = shuffled_parsed_ds.take(80_000).batch(128)
    # valid_data = shuffled_parsed_ds.skip(80_000).take(20_000).batch(128)
    
    valid_size = 20_000 # config.VALID_SIZE # 20_000 # args.valid_size
    valid = shuffled_parsed_ds.take(valid_size)
    train = shuffled_parsed_ds.skip(valid_size)
    cached_train = train.batch(args.batch_size * strategy.num_replicas_in_sync).prefetch(tf.data.AUTOTUNE)
    cached_valid = valid.batch(args.batch_size * strategy.num_replicas_in_sync).cache().prefetch(tf.data.AUTOTUNE)
    
    # ====================================================
    # metaparams for Vertex Ai Experiments
    # ====================================================
    logging.info('Logging metaparams & hyperparams for Vertex Experiments')
    
    EXPERIMENT_NAME = f"{args.experiment_name}"
    RUN_NAME = f"{args.experiment_run}"
    logging.info(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}\n RUN_NAME: {RUN_NAME}")
    
    metaparams = {}
    metaparams["experiment_name"] = f'{EXPERIMENT_NAME}'
    metaparams["experiment_run"] = f"{RUN_NAME}"
    metaparams["distribute"] = f'{args.distribute}'
    
    hyperparams = {}
    hyperparams["epochs"] = int(args.num_epochs)
    hyperparams["batch_size"] = int(args.batch_size)
    hyperparams["embedding_dim"] = args.embedding_dim
    hyperparams["projection_dim"] = args.projection_dim
    hyperparams["use_cross_layer"] = config.USE_CROSS_LAYER
    hyperparams["use_dropout"] = config.USE_DROPOUT
    hyperparams["dropout_rate"] = args.dropout_rate
    hyperparams['layer_sizes'] = args.layer_sizes
    
    logging.info(f"Creating run: {RUN_NAME}; for experiment: {EXPERIMENT_NAME}")
    
    # Create experiment
    vertex_ai.init(experiment=EXPERIMENT_NAME)
    # vertex_ai.start_run(RUN_NAME,resume=True) # RUN_NAME
    
    with vertex_ai.start_run(RUN_NAME) as my_run:
        logging.info(f"logging metaparams")
        my_run.log_params(metaparams)
        
        logging.info(f"logging hyperparams")
        my_run.log_params(hyperparams)
        
    # ====================================================
    # Compile, Adapt, and Train model
    # ====================================================
    logging.info('Setting model adapts and compiling the model')
    
    LAYER_SIZES = get_arch_from_string(args.layer_sizes)
    logging.info(f'LAYER_SIZES: {LAYER_SIZES}')
    
    logging.info(f'adapting layers: {config.NEW_ADAPTS}')
    # Wrap variable creation within strategy scope
    with strategy.scope():

        model = _model.TheTwoTowers(LAYER_SIZES, vocab_dict_load, parsed_candidate_dataset)
        
        if config.NEW_ADAPTS:
            model.query_tower.pl_name_text_embedding.layers[0].adapt(parsed_dataset_padded.map(lambda x: x['name']).batch(args.batch_size)) # TODO: adapts on full dataset or train only?
            # vocab_dict_load['name'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
            
#             model.candidate_tower.artist_name_can_text_embedding.layers[0].adapt(parsed_dataset_padded.map(lambda x: x['artist_name_can']).batch(args.batch_size))
#             # vocab_dict_load['artist_name_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
            
#             model.candidate_tower.track_name_can_text_embedding.layers[0].adapt(parsed_dataset_padded.map(lambda x: x['track_name_can']).batch(args.batch_size))
#             # vocab_dict_load['track_name_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
            
#             model.candidate_tower.album_name_can_text_embedding.layers[0].adapt(parsed_dataset_padded.map(lambda x: x['album_name_can']).batch(args.batch_size))
#             # vocab_dict_load['album_name_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
            
#             model.candidate_tower.artist_genres_can_text_embedding.layers[0].adapt(parsed_dataset_padded.map(lambda x: x['artist_genres_can']).batch(args.batch_size))
#             # vocab_dict_load['artist_genres_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
            
        model.compile(optimizer=tf.keras.optimizers.Adagrad(args.learning_rate))
        
    if config.NEW_ADAPTS:
        vocab_dict_load['name'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
    #     vocab_dict_load['artist_name_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
    #     vocab_dict_load['track_name_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
    #     vocab_dict_load['album_name_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
    #     vocab_dict_load['artist_genres_can'] = model.query_tower.pl_name_text_embedding.layers[0].get_vocabulary()
        bucket = storage_client.bucket(args.model_dir)
        blob = bucket.blob(f'vocabs_stats/vocab_dict_{args.version}.txt')
        pickle_out = pkl.dumps(vocab_dict_load)
        blob.upload_from_string(pickle_out)
    
    logging.info('Adapts finish - training next')
        
    tf.random.set_seed(args.seed)
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"gs://{args.model_dir}/logs-{RUN_NAME}",
        histogram_freq=0, 
        write_graph=True, 
        profile_batch = '500,520'
    )
    # if os.environ.get('AIP_TENSORBOARD_LOG_DIR', 'NA') is not 'NA':
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #         log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'],
    #         histogram_freq=0, write_graph=True, profile_batch = '500,520')
    # else:
    #     os.mkdir('/tb_logs')
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #         log_dir='/tb_logs',
    #         histogram_freq=0)
        
    logging.info('Training starting')
    layer_history = model.fit(
        cached_train,
        validation_data=cached_valid,
        validation_freq=args.valid_frequency,
        callbacks=tensorboard_callback,
        epochs=args.num_epochs,
        verbose=2
    )
    
    # Determine type and task of the machine from the strategy cluster resolver
    if args.distribute == 'multiworker':
        task_type, task_id = (strategy.cluster_resolver.task_type,
                              strategy.cluster_resolver.task_id)
    else:
        task_type, task_id = None, None
    
    # ====================================================
    # Eval Metrics
    # ====================================================
    logging.info('Getting evaluation metrics')

    val_metrics = model.evaluate(cached_valid, return_dict=True) #check performance
    
    logging.info('Validation metrics below:')
    logging.info(val_metrics)
    
    with vertex_ai.start_run(RUN_NAME,resume=True) as my_run:
        logging.info(f"logging metrics to experiment run {RUN_NAME}")
        my_run.log_metrics(val_metrics)
    
    # logging.info(f"Ending experiment run: {RUN_NAME}")
    # vertex_ai.end_run()
    
    # ====================================================
    # Save Towers
    # ====================================================
    
    logging.info(f'Saving models to {args.model_dir}')

    query_dir_save = f"gs://{args.model_dir}/{args.version}/{RUN_NAME}/query_tower/" #+ FLAGS.TS 
    candidate_dir_save = f"gs://{args.model_dir}/{args.version}/{RUN_NAME}/candidate_tower/" #+ FLAGS.TS 
    logging.info(f'Saving chief query model to {query_dir_save}')
    
    # save model from primary node in multiworker
    if _is_chief(task_type, task_id):
        tf.saved_model.save(model.query_tower, query_dir_save)
        logging.info(f'Saved chief query model to {query_dir_save}')
        tf.saved_model.save(model.candidate_tower, candidate_dir_save)
        logging.info(f'Saved chief candidate model to {candidate_dir_save}')
    else:
        worker_dir_query = query_dir_save + '/workertemp_query_/' + str(task_id)
        tf.io.gfile.makedirs(worker_dir_query)
        tf.saved_model.save(model.query_tower, worker_dir_query)
        logging.info(f'Saved worker: {task_id} query model to {worker_dir_query}')

        worker_dir_can = candidate_dir_save + '/workertemp_can_/' + str(task_id)
        tf.io.gfile.makedirs(worker_dir_can)
        tf.saved_model.save(model.candidate_tower, worker_dir_can)
        logging.info(f'Saved worker: {task_id} candidate model to {worker_dir_can}')

    if not _is_chief(task_type, task_id):
        tf.io.gfile.rmtree(worker_dir_can)
        tf.io.gfile.rmtree(worker_dir_query)

    logging.info('All done - model saved') #all done
    
def parse_args():
    """
    Parses command line arguments
    
    type: int, float, str
          bool() converts empty strings to `False` and non-empty strings to `True`
          see more details here: https://docs.python.org/3/library/argparse.html#type
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir', required=False)
    
    parser.add_argument('--train_dir', 
                        type=str, help='dir of training files', required=True)

    parser.add_argument('--candidate_file_dir', 
                        type=str, help='dir of candidate files', required=True)

    parser.add_argument('--project', 
                        type=str, help='project', required=True)

    parser.add_argument('--max_padding', 
                        default=375, type=int, help='max_padding', required=False)

    parser.add_argument('--experiment_name', 
                        type=str, help='#TODO', required=True)

    parser.add_argument('--experiment_run', 
                        type=str, help='#TODO', required=True)

    parser.add_argument('--num_epochs', 
                        default=1, type=int, help='#TODO', required=False)

    parser.add_argument('--batch_size', 
                        default=128, type=int, help='#TODO', required=False)

    parser.add_argument('--embedding_dim', 
                        default=32, type=int, help='#TODO', required=False)

    parser.add_argument('--projection_dim', 
                        default=5, type=int, help='#TODO', required=False)

    parser.add_argument('--seed', 
                        default=1234, type=str, help='#TODO', required=False)

#     parser.add_argument('--use_cross_layer', 
#                         default=True, type=bool, help='#TODO', required=False)

#     parser.add_argument('--use_dropout', 
#                         default=False, type=bool, help='#TODO', required=False)

    parser.add_argument('--dropout_rate', 
                        default=0.4, type=float, help='#TODO', required=False)

    parser.add_argument('--layer_sizes', 
                        default='[64,32]', type=str, help='#TODO', required=False)

    # parser.add_argument('--aip_tb_logs', 
    #                     default=os.getenv('AIP_TENSORBOARD_LOG_DIR'), type=str, help='#TODO', required=False)

    # parser.add_argument('--new_adapts', 
    #                     default=False, type=bool, help='#TODO', required=False)

    parser.add_argument('--learning_rate', 
                        default=0.01, type=float, help='learning rate', required=False)

    parser.add_argument('--valid_size', 
                        default='#TODO', type=str, help='number of records in valid split', required=False)

    parser.add_argument('--valid_frequency', 
                        default=10, type=int, help='number of epochs per metrics val calculation', required=False)

    parser.add_argument('--distribute', 
                        default='single', type=str, help='TF strategy: single, mirrored, multiworker, tpu', required=False)

    parser.add_argument('--version', 
                        type=str, help='version of train code; for tracking', required=True)

    parser.add_argument('--train_dir_prefix', 
                        type=str, help='file path under GCS bucket', required=True)

    parser.add_argument('--candidate_files_prefix', 
                        type=str, help='file path under GCS bucket', required=True)

    # args = parser.parse_args()
    return parser.parse_args()
    
if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout
    )

    parsed_args = parse_args()

    logging.info('Args: %s', parsed_args)
    start_time = time.time()
    logging.info('Starting jobs main() script')

    main(parsed_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info('Training completed. Elapsed time: %s', elapsed_time )
