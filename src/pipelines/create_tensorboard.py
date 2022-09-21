
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=[
                       'google-cloud-aiplatform==1.17.0',
  ],
  output_component_file="./pipelines/create_tensorboard.yaml",
)
def create_tensorboard(
    project: str,
    location: str,
    version: str,
    gcs_bucket_name: str,
    app_name: str,
    create_tb_resource: bool,
) -> NamedTuple('Outputs', [
                            ('tensorboard', Artifact),
                            ('tensorboard_resource_name', str),
]):

    import google.cloud.aiplatform as aiplatform
    from datetime import datetime
    import logging

    # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    aiplatform.init(
        project=project,
        location=location,
    )

    TENSORBOARD_DISPLAY_NAME = f"{app_name}-{version}"

    if create_tb_resource:
        logging.info(f"TENSORBOARD_DISPLAY_NAME: {TENSORBOARD_DISPLAY_NAME}")

        tensorboard = aiplatform.Tensorboard.create(display_name=TENSORBOARD_DISPLAY_NAME)

        tensorboard_resource_name = tensorboard.resource_name # projects/934903580331/locations/us-central1/tensorboards/6275818857298919424

        logging.info(f"Created tensorboard_resource_name: {tensorboard_resource_name}")

    else:
        logging.info(f"Searching for Existing TB: {TENSORBOARD_DISPLAY_NAME}")

        _tb_resource = aiplatform.TensorboardExperiment.list(
            filter=f'display_name="{TENSORBOARD_DISPLAY_NAME}"'
        )[0]

        # retrieve endpoint uri
        tensorboard_resource_name = _tb_resource.resource_name
        logging.info(f"Found existing TB resource: {tensorboard_resource_name}")

        tensorboard = aiplatform.Tensorboard(f'{tensorboard_resource_name}')

    return (
        tensorboard,
        f'{tensorboard_resource_name}',
    )
