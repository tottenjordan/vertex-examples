@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=['google-cloud-aiplatform==1.13.0'],
  output_component_file="./pipelines/reconfigure_traffic_split.yaml",
)
def reconfigure_traffic_split(
    project: str,
    location: str,
    endpoint: str,
    existing_endpoint_uri: str,
    endpoint_traffic_split: str,
) -> NamedTuple('Outputs', [
                            ('existing_endpoint_uri', str),
                            ('original_traffic_split', str),
                            ('existing_traffic_split', str),
                            ('updated_traffic_split', str),
]):

  from google.cloud import aiplatform
  from heapq import nsmallest, nlargest
  import json
  import logging

  '''
  Reconfigure's traffic_Split for 2 deployed models on a single endpoint
  If edited to accomadate more than 2 models, also edit pipeline step: "Undeploy Models"
  '''

  aiplatform.init(
      project=project,
      location=location,
  )

  _model_endpoint_resource = aiplatform.Endpoint(existing_endpoint_uri)

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

  logging.info(f"Input traffic split: {endpoint_traffic_split}")

  _existing_traffic_split = _model_endpoint_resource.traffic_split
  logging.info(f"Endpoint's current traffic split: {_existing_traffic_split}")

  deployed_models = _model_endpoint_resource.gca_resource.deployed_models
  logging.info(f"deployed_models: {deployed_models}")
  
  # deployed_model_count = len(deployed_models)
  # KEEPS = 2
  # N = max(KEEPS, deployed_model_count)
  N = 2

  deployed_model_dict = {}

  for model in deployed_models:
    deployed_model_dict[model.id] = model.create_time

  latest_deployed_models = nlargest(N, deployed_model_dict, key = deployed_model_dict.get)

  logging.info(f"deployed_model_dict: {deployed_model_dict}")
  logging.info(f"latest_deployed_models: {latest_deployed_models}")

  new_traffic_split = {}
  new_traffic_split[latest_deployed_models[0]] = 20                             # latest deployment
  new_traffic_split[latest_deployed_models[1]] = 80

  logging.info(f"new_traffic_split: {new_traffic_split}")
  if sum(new_traffic_split.values()) != 100:
    raise ValueError(
        "Sum of all traffic within traffic split needs to be 100."
    )
    
  ## Update Endpoint
  from google.protobuf.field_mask_pb2 import FieldMask

  # Create in-memory GAPIC endpoint and set new_traffic_split
  gapic_endpoint = clients["endpoint"].get_endpoint(
      name=_model_endpoint_resource.resource_name
  )
  gapic_endpoint.traffic_split = new_traffic_split
  gapic_endpoint.deployed_models = []

  clients["endpoint"].update_endpoint(
      endpoint=gapic_endpoint, update_mask=FieldMask(paths=["traffic_split"])
  )

  # refetch the endpoint
  gapic_endpoint = clients["endpoint"].get_endpoint(
      name=_model_endpoint_resource.resource_name
  )
  updated_traffic_split = gapic_endpoint.traffic_split
  logging.info(f"updated_traffic_split: {updated_traffic_split}")

  return (
      f'{existing_endpoint_uri}',
      f'{endpoint_traffic_split}',
      f'{_existing_traffic_split}',
      f'{updated_traffic_split}',
  )
