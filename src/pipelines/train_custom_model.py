
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
    # output_component_file="./pipelines/train_custom_model.yaml",
)
def train_custom_model(
    project: str,
    # version: str,
    model_version: str,
    pipeline_version: str,
    model_name: str, 
    worker_pool_specs: dict,
    vocab_dict_uri: str, 
    train_output_gcs_bucket: str,                         # change to workdir?
    training_image_uri: str,
    tensorboard_resource_name: str,
    service_account: str,
    experiment_name: str,
    experiment_run: str,
) -> NamedTuple('Outputs', [
    ('query_tower_dir_uri', str),
    ('candidate_tower_dir_uri', str),
    # ('candidate_index_dir_uri', str),
]):
    
    from google.cloud import aiplatform as vertex_ai
    import logging
    import numpy as np
    
    vertex_ai.init(
        project=project,
        location='us-central1',
    )
    
    JOB_NAME = f'train-{model_name}'
    logging.info(f'JOB_NAME: {JOB_NAME}')
    
    BASE_OUTPUT_DIR = f'gs://{train_output_gcs_bucket}/{experiment_name}/{experiment_run}'
    logging.info(f'BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}')
    
    logging.info(f'vocab_dict_uri: {vocab_dict_uri}')
    
    logging.info(f'tensorboard_resource_name: {tensorboard_resource_name}')
    logging.info(f'service_account: {service_account}')
    logging.info(f'worker_pool_specs: {worker_pool_specs}')
  
    job = vertex_ai.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=BASE_OUTPUT_DIR,
    )
    
    logging.info(f'Submitting train job to Vertex AI...')

    job.run(
        tensorboard=tensorboard_resource_name,
        service_account=f'{service_account}',
        restart_job_on_worker_restart=False,
        enable_web_access=True,
        sync=False,
    )
    
    # TODO: this is hardcoded, but created in train job
    query_tower_dir_uri = f"gs://{output_dir_gcs_bucket_name}/{experiment_name}/{experiment_run}/model-dir/query_tower" 
    candidate_tower_dir_uri = f"gs://{output_dir_gcs_bucket_name}/{experiment_name}/{experiment_run}/model-dir/candidate_tower"
    # candidate_index_dir_uri = f"gs://{output_dir_gcs_bucket_name}/{experiment_name}/{experiment_run}/candidate-index"
    
    return (
        f'{query_tower_dir_uri}',
        f'{candidate_tower_dir_uri}',
        # f'{candidate_index_dir_uri}',
    )
