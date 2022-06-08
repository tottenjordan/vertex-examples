@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-aiplatform==1.13.0'],
  output_component_file="./pipelines/undeploy_model_from_endpoint.yaml",
)
def undeploy_model_from_endpoint(
    project: str,
    location: str,
    existing_endpoint_uri: str,
    deployed_models_count: int,
) -> NamedTuple('Outputs', [
                            ('existing_endpoint_uri', str),
                            ('endpoint_traffic_split', str),
                            ('deployed_models_count', int),
                            ('deployed_model_list', list),
]):

  from google.cloud import aiplatform
  from heapq import nsmallest, nlargest
  import json
  import logging
  from google.protobuf.field_mask_pb2 import FieldMask
  '''
  Undeploys all but the latest 1 (edit for 2) deployed models per Model Endpoint
  Endpoint's traffic_split needs to reflect 0% for each model_id to be undeployed
  '''

  aiplatform.init(
      project=project,
      location=location,
  )

  # API service endpoint
  API_ENDPOINT = f"{location}-aiplatform.googleapis.com"

  # Vertex location root path for your dataset, model and endpoint resources
  PARENT = "projects/" + project + "/locations/" + location

  # client options same for all services
  client_options = {"api_endpoint": API_ENDPOINT}

  def create_endpoint_client():
    client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)
    return client

  clients = {}
  clients["endpoint"] = create_endpoint_client()

  _model_endpoint_resource = aiplatform.Endpoint(existing_endpoint_uri)
  endpoint_traffic_split = _model_endpoint_resource.traffic_split
  logging.info(f"Original endpoint_traffic_split: {endpoint_traffic_split}")

  # Models being undeployed should have 0 traffic
  gapic_endpoint = clients["endpoint"].get_endpoint(name=_model_endpoint_resource.resource_name)

  # Get depoyed model_ids and their create_time
  deployed_models = _model_endpoint_resource.gca_resource.deployed_models
  deployed_models_count = len(deployed_models)
  logging.info(f"deployed_models_count: {deployed_models_count}")

  KEEPS = 1 # 2
  REMOVE = deployed_models_count- KEEPS

  deployed_model_dict = {}

  for model in deployed_models:
    deployed_model_dict[model.id] = model.create_time

  latest_deployed_models = nlargest(KEEPS, deployed_model_dict, key = deployed_model_dict.get)
  oldest_deployed_models = nsmallest(REMOVE, deployed_model_dict, key = deployed_model_dict.get)

  logging.info(f"deployed_model_dict: {deployed_model_dict}")
  logging.info(f"latest_deployed_models: {latest_deployed_models}")
  logging.info(f"oldest_deployed_models: {oldest_deployed_models}")

  new_traffic_split = {}
  new_traffic_split[latest_deployed_models[0]] = 100
  # new_traffic_split[latest_deployed_models[1]] = 80

  for i in range(len(oldest_deployed_models)):
    new_traffic_split[oldest_deployed_models[i]] = 0

  logging.info(f"new_traffic_split: {new_traffic_split}")

  # assign new traffic split to in-memory GAPIC endpoint
  gapic_endpoint.traffic_split = new_traffic_split
  gapic_endpoint.deployed_models = []

  # push new traffic split to Endpoint load balancer
  clients["endpoint"].update_endpoint(
      endpoint=gapic_endpoint, update_mask=FieldMask(paths=["traffic_split"])
  )

  # Undeploy older deployments
  for model_id in oldest_deployed_models:
    _model_endpoint_resource.undeploy(model_id)
    logging.info(f"Undeployed deployed_model ID: {model_id}")


  # retrieve remaining deployed model IDs
  _deployed_models = _model_endpoint_resource.gca_resource.deployed_models
  
  deployed_model_list = []
  for model in _deployed_models:
    deployed_model_list.append(model.id)
  
  deployed_models_count = len(deployed_model_list)

  endpoint_traffic_split = _model_endpoint_resource.traffic_split
  logging.info(f"Final endpoint_traffic_split ID: {endpoint_traffic_split}")


  return (
      f'{existing_endpoint_uri}',
      f'{endpoint_traffic_split}',
      deployed_models_count,
      deployed_model_list,
  )
