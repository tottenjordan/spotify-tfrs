
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-aiplatform==1.17.0'],
    # output_component_file="./pipelines/find_model_endpoint.yaml",
)
def find_model_endpoint_test(
    project: str,
    location: str,
    endpoint_name: str,
) -> NamedTuple('Outputs', [
                            ('create_new_endpoint', str),
                            ('existing_endpoint_uri', str),
                            ('deployed_models_count', int),
                            ('undeploy_model_needed', str),
                            ('deployed_model_list', list),
                            ('endpoint_traffic_split', str),
]):

    from google.cloud import aiplatform as vertex_ai
    import json
    import logging

    vertex_ai.init(
        project=project,
        location=location,
    )

    deployed_model_list = []

    logging.info(f"Searching for model endpoint: {endpoint_name}")

    if vertex_ai.Endpoint.list(
        filter=f'display_name="{endpoint_name}"'):
        '''
        Because existing Endpoint found: 
            (1) will not create new
            (2) Need the endpoint uri
            (3) Need list of deployed models on this endpoint;
            (4) If more than 1 deployed model exists, trigger subsequent conditional step
                to undeploy all but 1 (latest) model 

        '''
        logging.info(f"Model endpoint, {endpoint_name}, already exists")
        create_new_endpoint="False"
    
        # create endpoint list resource in memory
        _endpoint = vertex_ai.Endpoint.list(
            filter=f'display_name="{endpoint_name}"'
        )[0]
        
        logging.info(f"Parsing details for _endpoint: {_endpoint}")

        # retrieve endpoint uri
        existing_endpoint_uri = _endpoint.resource_name
        logging.info(f"existing_endpoint_uri: {existing_endpoint_uri}")
        _traffic_split = _endpoint.traffic_split

        # retrieve deployed model IDs
        deployed_models = _endpoint.gca_resource.deployed_models
        deployed_models_count = len(deployed_models)
        logging.info(f"deployed_models_count: {deployed_models_count}")

        if deployed_models_count > 1:
            # deployed_model_id_0 = _endpoint.gca_resource.deployed_models[0].id
            # deployed_model_id_1 = _endpoint.gca_resource.deployed_models[1].id
            undeploy_model_needed = "True"                               # arbitrary assumption: no more than 2 (3) models per model_endpoint
            for model in deployed_models:
                deployed_model_list.append(model.id)
        elif deployed_models_count == 0:
            undeploy_model_needed = "False"
        else:
            undeploy_model_needed = "False"
            deployed_model_list.append(_endpoint.gca_resource.deployed_models[0].id)

        # deployed_model_id = _endpoint.gca_resource.deployed_models[0].id
        logging.info(f"Currently deployed_model_list {deployed_model_list}")

    else:
        logging.info(f"Model endpoint, {endpoint_name}, does not exist")

        create_new_endpoint="True"
        deployed_models_count=0
        existing_endpoint_uri="N/A"
        undeploy_model_needed = "N/A"
        _traffic_split = "N/A"
        # deployed_model_list = []

        logging.info(f"create_new_endpoint {create_new_endpoint}")
        logging.info(f"existing_endpoint_uri {existing_endpoint_uri}")
        logging.info(f"deployed_models_count {deployed_models_count}")
        logging.info(f"undeploy_model_needed {undeploy_model_needed}")
        logging.info(f"deployed_model_list {deployed_model_list}")
        logging.info(f"_traffic_split {_traffic_split}")


    return (
        f'{create_new_endpoint}',
        f'{existing_endpoint_uri}',
        deployed_models_count,
        f'{undeploy_model_needed}',
        deployed_model_list,
        f'{_traffic_split}',
    )
