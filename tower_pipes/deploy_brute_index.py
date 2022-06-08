@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',    # TODO: update once merged
    ],
    output_component_file="./pipelines/deploy_brute_index.yaml",
)
def deploy_brute_index(
    project: str,
    location: str,
    version: str,
    deployed_brute_force_index_name: str,
    brute_force_index_resource_uri: str,
    brute_index_endpoint_resource_uri: str,
) -> NamedTuple('Outputs', [
                            ('brute_index_endpoint_resource_uri', str),
                            ('brute_force_index_resource_uri', str),
                            ('deployed_brute_force_index_name', str),
                            ('deployed_brute_force_index', Artifact),
]):
  
  from google.cloud import aiplatform
  from datetime import datetime
  import logging
  # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

  aiplatform.init(project=project, location=location)

  brute_index = aiplatform.MatchingEngineIndex(
      index_name=brute_force_index_resource_uri
  )
  brute_force_index_resource_uri = brute_index.resource_name

  index_endpoint = aiplatform.MatchingEngineIndexEndpoint(brute_index_endpoint_resource_uri)

  logging.info(f"Deploying Brute Force index to {brute_index_endpoint_resource_uri}")

  index_endpoint = index_endpoint.deploy_index(
      index=brute_index, 
      deployed_index_id=f'{deployed_brute_force_index_name}'
  )

  logging.info(f"Index deployed: {index_endpoint.deployed_indexes}")

  return (
      f'{brute_index_endpoint_resource_uri}',
      f'{brute_force_index_resource_uri}',
      f'{deployed_brute_force_index_name}',
      brute_index,
  )
