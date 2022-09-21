
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)
@component(
    base_image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
    packages_to_install=["google-cloud-build"],
    output_component_file="./pipelines/build_custom_train_image.yaml",
)
def build_custom_train_image(
    project: str, 
    gcs_train_script_path: str,                                                 # TRAIN_APP_CODE_PATH = f"{BUCKET_URI}/{APP_NAME}/{VERSION}/vertex_train/"
    training_image_uri: str,                                                   # TRAIN_IMAGE_URI = f"gcr.io/{PROJECT_ID}/multiworker:2tower-pipe-{VERSION}" 
) -> NamedTuple("Outputs", [("training_image_uri", str)]):

    # TODO: make output Artifact for image_uri
    """
    custom pipeline component to build custom training image using
    Cloud Build and the training application code and dependencies
    defined in the Dockerfile
    """

    import logging
    import os

    from google.cloud.devtools import cloudbuild_v1 as cloudbuild
    from google.protobuf.duration_pb2 import Duration

    # initialize client for cloud build
    logging.getLogger().setLevel(logging.INFO)
    build_client = cloudbuild.services.cloud_build.CloudBuildClient()

    # parse step inputs to get path to Dockerfile and training application code
    gs_dockerfile_path = os.path.join(gcs_train_script_path, "Dockerfile")   # two-tower-pipes/2tower-recsys/vertex_train
    _gcs_train_script_path = os.path.join(gcs_train_script_path, "trainer/")  # TRAIN_APP_CODE_PATH = f"{BUCKET_URI}/{APP_NAME}/{VERSION}/vertex_train/"

    logging.info(f"training_image_uri: {training_image_uri}") 

    # define build steps to pull the training code and Dockerfile
    # and build/push the custom training container image
    build = cloudbuild.Build()
    build.steps = [
        {
            "name": "gcr.io/cloud-builders/gsutil",
            "args": ["cp", "-r", _gcs_train_script_path, "."],
        },
        {
            "name": "gcr.io/cloud-builders/gsutil",
            "args": ["cp", gs_dockerfile_path, "Dockerfile"],
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

    # get build status
    result = operation.result()
    logging.info("RESULT:", result.status)

    # return step outputs
    return (
        training_image_uri,
    )
