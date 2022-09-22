
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
    packages_to_install=["google-cloud-build"],
    # output_component_file="./pipelines/build_custom_train_image.yaml",
)
def build_custom_train_image(
    project: str, 
    gcs_train_script_path: str,   # TRAIN_APP_CODE_PATH = f"{BUCKET_URI}/{VERSION}/src/" # jt-tfrs-test/pipev1/src
    training_image_uri: str,      # TRAIN_IMAGE_URI = f"gcr.io/{PROJECT_ID}/multiworker:2tower-pipe-{VERSION}"
    train_dockerfile_name: str,   # Dockerfile.tfrs
) -> NamedTuple("Outputs", [("training_image_uri", str)]):

    # TODO: make output Artifact for image_uri
    """
    custom pipeline component to build custom training image using
    Cloud Build and the training application code and dependencies
    defined in the Dockerfile
    """

    import logging
    import os
    import time

    from google.cloud.devtools import cloudbuild_v1 as cloudbuild
    from google.protobuf.duration_pb2 import Duration

    # initialize client for cloud build
    logging.getLogger().setLevel(logging.INFO)
    build_client = cloudbuild.services.cloud_build.CloudBuildClient()
    
    logging.info(f"train_dockerfile_name: {train_dockerfile_name}")

    # parse step inputs to get path to Dockerfile and training application code
    gcs_dockerfile_path = os.path.join(gcs_train_script_path, f"{train_dockerfile_name}")   # two-tower-pipes/2tower-recsys/vertex_train
    # gcs_cloudbuild_path = os.path.join(gcs_train_script_path, f"cloudbuild.yaml")
    gcs_train_script_dir = os.path.join(gcs_train_script_path, "trainer/")  # TRAIN_APP_CODE_PATH = f"{BUCKET_URI}/{APP_NAME}/{VERSION}/vertex_train/"
    
    logging.info(f"gcs_dockerfile_path: {gcs_dockerfile_path}")
    # logging.info(f"gcs_cloudbuild_path: {gcs_cloudbuild_path}")
    logging.info(f"gcs_train_script_dir: {gcs_train_script_dir}")
    
    logging.info(f"training_image_uri: {training_image_uri}") 
     

    start_time = time.time()

    # define build steps to pull the training code and Dockerfile
    # and build/push the custom training container image
    build = cloudbuild.Build()
    build.steps = [
        {
            "name": "gcr.io/cloud-builders/gsutil",
            "args": ["cp", "-r", gcs_train_script_dir, "."],
        },
        {
            "name": "gcr.io/cloud-builders/gsutil",
            "args": ["cp", gcs_dockerfile_path, f"{train_dockerfile_name}"],
        },
        # enabling Kaniko cache in a Docker build that caches intermediate
        # layers and pushes image automatically to Container Registry
        # https://cloud.google.com/build/docs/kaniko-cache
        # {
        #     "name": "gcr.io/kaniko-project/executor:latest",
        #     # "name": "gcr.io/kaniko-project/executor:v1.8.0",        # TODO; downgraded to avoid error in build
        #     # "args": [f"--destination={training_image_uri}", "--cache=true"],
        #     "args": [f"--destination={training_image_uri}", "--cache=false"],
        # },
        {
            "name": "gcr.io/cloud-builders/docker",
            "args": ['build','-t', f'{training_image_uri}', '.'],
        },
        {
            "name": "gcr.io/cloud-builders/docker",
            "args": ['push', f'{training_image_uri}'], 
        },
    ]
    # override default timeout of 10min
    timeout = Duration()
    timeout.seconds = 7200
    build.timeout = timeout

    # create build
    operation = build_client.create_build(project_id=project, build=build)
    logging.info("IN PROGRESS:")
    logging.info(operation.metadata)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    logging.info(f"Elapsed timefor build: {elapsed_time}")

    # get build status
    result = operation.result()
    logging.info("RESULT:", result.status)
    
    logging.info(f"training_image_uri: {training_image_uri}")

    # return step outputs
    return (
        training_image_uri,
    )
