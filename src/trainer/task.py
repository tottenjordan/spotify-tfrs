
import json
import tensorflow as tf
# from tensorflow.keras import mixed_precision
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
from google.cloud.aiplatform.training_utils import cloud_profiler

import numpy as np

# ====================================================
# Helper functions
# ====================================================

def _is_chief(task_type, task_id): 
    ''' Check for primary if multiworker training
    '''
    if task_type == 'chief':
        results = 'chief'
    else:
        results = None
    return results
    # return (task_type == 'chief') or (task_type == 'worker' and task_id == 0) or task_type is None
    # return ((task_type == 'chief' and task_id == 0) or task_type is None)

def get_arch_from_string(arch_string):
    q = arch_string.replace(']', '')
    q = q.replace('[', '')
    q = q.replace(" ", "")
    return [int(x) for x in q.split(',')]

# ====================================================
# Main
# ====================================================
import data_src as trainer_data
import model_src as trainer_model
import train_config as cfg
import time

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

def main(args):
    
    # limiting GPU growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f'detected: {len(gpus)} GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)
                
    # tf.debugging.set_log_device_placement(True) # logs all tf ops and their device placement;
    # TF_GPU_THREAD_MODE='gpu_private'
    os.environ['TF_GPU_THREAD_MODE']='gpu_private'
    os.environ['TF_GPU_THREAD_COUNT']='1'
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    
    storage_client = storage.Client(
        project=args.project
    )
    
    # ====================================================
    # Set Device Strategy
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
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
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

    logging.info(f'TF training strategy = {strategy}')
    
    NUM_REPLICAS = strategy.num_replicas_in_sync
    logging.info(f'num_replicas_in_sync = {NUM_REPLICAS}')
    
    # Here the batch size scales up by number of workers since
    # `tf.data.Dataset.batch` expects the global batch size.
    GLOBAL_BATCH_SIZE = args.batch_size * NUM_REPLICAS
    logging.info(f'GLOBAL_BATCH_SIZE = {GLOBAL_BATCH_SIZE}')

    # TODO: Determine type and task of the machine from the strategy cluster resolver
    logging.info(f'Setting task_type and task_id...')
    # task_type, task_id = (
    #     strategy.cluster_resolver.task_type,
    #     strategy.cluster_resolver.task_id
    # )
    if args.distribute == 'multiworker':
        task_type, task_id = (
            strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id
        )
    else:
        task_type, task_id = 'chief', None
    
    logging.info(f'task_type = {task_type}')
    logging.info(f'task_id = {task_id}')
    # ====================================================
    # Vocab Files
    # ====================================================

    # TODO: parameterize & configure for adapts vs vocab files

    BUCKET_NAME = 'spotify-v1'                             # args.vocab_gcs_bucket
    FILE_PATH = 'vocabs/v2_string_vocabs'                  # args.vocab_gcs_file_path
    FILE_NAME = 'string_vocabs_v1_20220924-tokens22.pkl'   # args.vocab_filename
    DESTINATION_FILE = 'downloaded_vocabs.txt'     

    with open(f'{DESTINATION_FILE}', 'wb') as file_obj:
        storage_client.download_blob_to_file(
            f'gs://{BUCKET_NAME}/{FILE_PATH}/{FILE_NAME}', file_obj)


    with open(f'{DESTINATION_FILE}', 'rb') as pickle_file:
        vocab_dict_load = pkl.load(pickle_file)

    # ====================================================
    # TRAIN dataset - Parse & Pad
    # ====================================================
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO # FILE | DATA | AUTO
    # options.experimental_optimization.map_and_batch_fusion = False
    
    logging.info(f'Path to TRAIN files: gs://{args.train_dir}/{args.train_dir_prefix}')
    
    train_files = []
    for blob in storage_client.list_blobs(f'{args.train_dir}', prefix=f'{args.train_dir_prefix}', delimiter="/"):
        train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
    
    # OPTIMIZE DATA INPUT PIPELINE
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    # train_dataset = tf.data.Dataset.list_files(train_files)
    # train_dataset = train_dataset.shard(num_shards=NUM_REPLICAS, index=REPLICA_ID).repeat(args.num_epochs)
    train_dataset = train_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    ).map(
        trainer_data.parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE,
          ).map(
        trainer_data.return_padded_tensors, #(max_playlist_len=args.max_padding),
        num_parallel_calls=tf.data.AUTOTUNE,
          ).batch(
        args.batch_size * strategy.num_replicas_in_sync   # GLOBAL_BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE,
    ).with_options(options)
    
    # ====================================================
    # VALID dataset - Parse & Pad 
    # ====================================================
    
    logging.info(f'Path to VALID files: gs://{args.valid_dir}/{args.valid_dir_prefix}')
    
    valid_files = []
    for blob in storage_client.list_blobs(f'{args.valid_dir}', prefix=f'{args.valid_dir_prefix}', delimiter="/"):
        valid_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
    
    # OPTIMIZE DATA INPUT PIPELINE
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files)
    # valid_dataset = tf.data.Dataset.list_files(valid_files)
    valid_dataset = valid_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    ).map(
        trainer_data.parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE,
          ).map(
        trainer_data.return_padded_tensors, #(max_playlist_len=args.max_padding),
        num_parallel_calls=tf.data.AUTOTUNE,
          ).batch(
        args.batch_size * strategy.num_replicas_in_sync   # GLOBAL_BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE,
    ).with_options(options)
    
    # ====================================================
    # Parse candidates dataset
    # ====================================================
    
    logging.info(f'Path to CANDIDATE files: gs://{args.candidate_file_dir}/{args.candidate_files_prefix}')

    candidate_files = []
    for blob in storage_client.list_blobs(f'{args.candidate_file_dir}', prefix=f'{args.candidate_files_prefix}', delimiter="/"):
        candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
    
    #generate the candidate dataset
    candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)
    parsed_candidate_dataset = candidate_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    ).map(
        trainer_data.parse_candidate_tfrecord_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(
        tf.data.AUTOTUNE,
    ).with_options(options)
    
    # ====================================================
    # metaparams for Vertex Ai Experiments
    # ====================================================
#     logging.info('Logging metaparams & hyperparams for Vertex Experiments')
    
    EXPERIMENT_NAME = f"{args.experiment_name}"
    # EXPERIMENT_DESCRIPTION = f"vertex ai experiment runs for model_version: {args.model_version}"
    EXPERIMENT_TENSORBOARD = f"{args.experiment_tb}"
    RUN_NAME = f"{args.experiment_run}-{TIMESTAMP}"
    
    # BASE_OUTPUT_DIR = f'gs://{OUTPUT_BUCKET}/{EXPERIMENT_NAME}/{RUN_NAME_PREFIX}' # gcs path submitted to vertex job
    WORKING_DIR_GCS_URI = f'gs://{args.train_output_gcs_bucket}/{EXPERIMENT_NAME}/{RUN_NAME}' 
    
    logging.info(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")
    # logging.info(f"EXPERIMENT_DESCRIPTION: {EXPERIMENT_DESCRIPTION}")
    logging.info(f"EXPERIMENT_TENSORBOARD: {EXPERIMENT_TENSORBOARD}")
    logging.info(f"RUN_NAME: {RUN_NAME}")
    logging.info(f'Train job output directory: {WORKING_DIR_GCS_URI}') 

    # # Create experiment
    vertex_ai.init(
        experiment=EXPERIMENT_NAME,
        # experiment_description=EXPERIMENT_DESCRIPTION,
        # experiment_tensorboard=EXPERIMENT_TENSORBOARD,
        # project=args.project,
    )
        
    # ====================================================
    # Compile, Adapt, and Train model
    # ====================================================
    logging.info('Setting model adapts and compiling the model')
    
    LAYER_SIZES = get_arch_from_string(args.layer_sizes)
    logging.info(f'LAYER_SIZES: {LAYER_SIZES}')

    # if task_type == 'chief':
    #     logging.info(f" initializing experiment run: {RUN_NAME}")
    #     with vertex_ai.start_run(RUN_NAME) as my_run:
    #         vertex_ai.start_run(run=f"{RUN_NAME}", tensorboard=EXPERIMENT_TENSORBOARD)
    
    # Wrap variable creation within strategy scope
    with strategy.scope():

        model = trainer_model.TheTwoTowers(LAYER_SIZES, vocab_dict_load, parsed_candidate_dataset) #, max_padding_len=args.max_padding)
            
        model.compile(optimizer=tf.keras.optimizers.Adagrad(args.learning_rate))
    
    logging.info('model compiled...')
        
    tf.random.set_seed(args.seed)
    
    # logs_dir = f'gs://{args.train_output_gcs_bucket}/{EXPERIMENT_NAME}/{RUN_NAME}/tb-logs'
    logs_dir = f'{WORKING_DIR_GCS_URI}/tb-logs'
    # AIP_LOGS = os.environ.get('AIP_TENSORBOARD_LOG_DIR', f'{logs_dir}')
    logging.info(f'TensorBoard logdir: {logs_dir}')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir,
        histogram_freq=0, 
        write_graph=True, 
        # profile_batch = '500,520'
    )
    
    cloud_profiler.init()
    
    logging.info('Starting training loop...')
    start_model_fit = time.time()
    
    layer_history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        validation_freq=args.valid_frequency,
        callbacks=tensorboard_callback,
        # steps_per_epoch=10, #for debugging purposes
        epochs=args.num_epochs,
        verbose=1
    )
    
    # capture elapsed time
    end_model_fit = time.time()
    elapsed_model_fit = end_model_fit - start_model_fit
    elapsed_model_fit = round(elapsed_model_fit, 2)
    logging.info(f'Elapsed model_fit: {elapsed_model_fit} seconds')
    
    # Log additional parameters
    # if task_type == 'chief':
    #     logging.info(f" logging layer_history.params: {layer_history.params}")
    #     vertex_ai.log_params(layer_history.params)
            
        # # TODO: Log metrics per epochs with `layer_history.params`
        # for idx in range(0, history.params["epochs"]):
        #     vertex_ai.log_time_series_metrics(
        #         {
        #             "train_mae": history.history["mae"][idx],
        #             "train_mse": history.history["mse"][idx],
        #         }
        #     )
        
#     # ====================================================
#     # Aprroximate Validation with ScaNN
#     # ====================================================
#     # Get candidate item (songs/tracks) embeddings
#     song_embeddings = parsed_candidate_dataset.batch(2048).map(            # 2048 | GLOBAL_BATCH_SIZE
#         model.candidate_tower, 
#         num_parallel_calls=tf.data.AUTOTUNE
#     ).prefetch(
#         tf.data.AUTOTUNE
#     ).with_options(options)
    
#     logging.info("Creating ScaNN layer for approximate validation metrics")
#     start_scann_layer = time.time()
    
#     # Compute predictions # TODO: parameterize
#     # with strategy.scope():
#     scann = tfrs.layers.factorized_top_k.ScaNN(
#         num_reordering_candidates=500,
#         num_leaves_to_search=30
#     )
#     scann.index_from_dataset(song_embeddings)

#     model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
#         candidates=scann
#     )

#     with strategy.scope():
#         model.compile()
        
#     logging.info('model compiled...')
    
#     # capture elapsed time
#     end_scann_layer = time.time()
#     elapsed_scann_layer = end_scann_layer - start_scann_layer
#     elapsed_scann_layer = round(elapsed_scann_layer, 2)
#     logging.info(f'Elapsed ScaNN Layer: {elapsed_scann_layer} seconds')
    
#     logging.info("custom scann layer generation for validation complete")
    #TODO - perhaps output the scann layer for indexing use if needed

    # ====================================================
    # Eval Metrics
    # ====================================================
    logging.info('Getting evaluation metrics...')
    start_evaluation = time.time()
    
    val_metrics = model.evaluate(
        valid_dataset,
        verbose="auto",
        return_dict=True,
        callbacks=tensorboard_callback,
    ) #check performance
    
    # capture elapsed time
    end_evaluation = time.time()
    elapsed_evaluation = end_evaluation - start_evaluation
    elapsed_evaluation = round(elapsed_evaluation, 2)
    logging.info(f'Elapsed model Evaluation: {elapsed_evaluation} seconds')

    logging.info('Validation metrics:')
    logging.info(val_metrics)
    
    # ====================================================
    # metaparams for Vertex Ai Experiments
    # ====================================================
    logging.info('Logging metaparams, hyperparams, and metrics to Experiments')
    
    time_metrics = {}
    time_metrics["elapsed_model_fit"] = elapsed_model_fit
    # time_metrics["elapsed_scann_layer"] = elapsed_scann_layer
    time_metrics["elapsed_evaluation"] = elapsed_evaluation
    
    metaparams = {}
    # metaparams["experiment_name"] = f'{EXPERIMENT_NAME}'
    # metaparams["experiment_run"] = f"{RUN_NAME}"
    metaparams["model_version"] = f"{args.model_version}"
    metaparams["pipe_version"] = f"{args.pipeline_version}"
    metaparams["data_regime"] = f"{args.data_regime}"
    metaparams["distribute"] = f'{args.distribute}'
    
    hyperparams = {}
    hyperparams["epochs"] = int(args.num_epochs)
    hyperparams["batch_size"] = int(args.batch_size)
    hyperparams["embedding_dim"] = args.embedding_dim
    hyperparams["projection_dim"] = args.projection_dim
    hyperparams["use_cross_layer"] = cfg.USE_CROSS_LAYER # args.use_cross_layer
    hyperparams["use_dropout"] = cfg.USE_DROPOUT # args.use_dropout
    hyperparams["dropout_rate"] = args.dropout_rate
    hyperparams['layer_sizes'] = args.layer_sizes
    hyperparams['max_padding'] = args.max_padding
    
    # IF CHIEF, LOG to EXPERIMENT
    # if _is_chief(task_type, task_id):
    if task_type == 'chief':
        logging.info(f" task_type logging experiments: {task_type}")
        logging.info(f" task_id logging experiments: {task_id}")

        with vertex_ai.start_run(RUN_NAME, tensorboard=EXPERIMENT_TENSORBOARD) as my_run:
            logging.info(f"logging metrics...")
            my_run.log_metrics(val_metrics)
            my_run.log_metrics(time_metrics)

            logging.info(f"logging metaparams...")
            my_run.log_params(metaparams)

            logging.info(f"logging hyperparams...")
            my_run.log_params(hyperparams)

            vertex_ai.end_run()
            logging.info(f"EXPERIMENT RUN: {RUN_NAME} has ended")

    
    # # if _is_chief(task_type, task_id):
    # if task_type == 'chief':
    #     with vertex_ai.start_run(RUN_NAME,resume=True) as my_run:
    #         logging.info(f"logging metrics to experiment run {RUN_NAME}")
    #         my_run.log_metrics(val_metrics)
    #         my_run.log_metrics(time_metrics)
    
    # logging.info(f"Ending experiment run: {RUN_NAME}")
    # vertex_ai.end_run()
    
    # ====================================================
    # Save Towers
    # ====================================================
    
    # MODEL_DIR_GCS_URI = f'gs://{args.train_output_gcs_bucket}/{EXPERIMENT_NAME}/{RUN_NAME}/model-dir'
    MODEL_DIR_GCS_URI = f'{WORKING_DIR_GCS_URI}/model-dir'
    
    logging.info(f'Saving models to {MODEL_DIR_GCS_URI}')

    query_dir_save = f"{MODEL_DIR_GCS_URI}/query_tower/"                                      # replaced: f"gs://{args.model_dir}/{args.version}/{RUN_NAME}/query_tower/" 
    candidate_dir_save = f"{MODEL_DIR_GCS_URI}/candidate_tower/"                              # replaced: f"gs://{args.model_dir}/{args.version}/{RUN_NAME}/candidate_tower/"
    logging.info(f'Saving chief query model to {query_dir_save}')
    
    # save model from primary node in multiworker
    # if _is_chief(task_type, task_id):
    if task_type == 'chief':
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

    # if not _is_chief(task_type, task_id):
    if task_type != 'chief':
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
                        default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir', required=False) # TODO: sunset this arg
    
    parser.add_argument('--train_output_gcs_bucket',
                        default=os.getenv('AIP_MODEL_DIR'), type=str, help='bucket for train job output', required=False) # TODO: use this
    
    parser.add_argument('--train_dir', 
                        type=str, help='bucket holding training files', required=True)
    
    parser.add_argument('--train_dir_prefix', 
                        type=str, help='file path under GCS bucket', required=True)
    
    parser.add_argument('--valid_dir', 
                        type=str, help='bucket holding valid files', required=True)
    
    parser.add_argument('--valid_dir_prefix', 
                        type=str, help='file path under GCS bucket', required=True)
    
    parser.add_argument('--candidate_file_dir', 
                        type=str, help='bucket holding candidate files', required=True)

    parser.add_argument('--candidate_files_prefix', 
                        type=str, help='file path under GCS bucket', required=True)

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

    # parser.add_argument('--valid_size', 
    #                     default='#TODO', type=str, help='number of records in valid split', required=False)

    parser.add_argument('--valid_frequency', 
                        default=10, type=int, help='number of epochs per metrics val calculation', required=False)

    parser.add_argument('--distribute', 
                        default='single', type=str, help='TF strategy: single, mirrored, multiworker, tpu', required=False)

    # parser.add_argument('--version', 
    #                     type=str, help='version of train code; for tracking', required=True)
    
    parser.add_argument('--model_version', 
                        type=str, help='version of model train code', required=True)
    
    parser.add_argument('--pipeline_version', 
                        type=str, help='version of pipeline code; v0 for non-pipeline execution', required=True)
    
    parser.add_argument('--data_regime', 
                        type=str, help='id for tracking different datasets', required=True)
    
    parser.add_argument(
        '--experiment_tb', 
        type=str, 
        help='''
        The Vertex AI TensorBoard instance, 
        Tensorboard resource name,
        or Tensorboard resource ID
        ''',
        required=False
    )


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
