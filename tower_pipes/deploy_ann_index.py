@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',    # TODO: update once merged
    ],
    output_component_file="./pipelines/deploy_ann_index.yaml",
)
def deploy_ann_index(
    project: str,
    location: str,
    version: str,
    deployed_ann_index_name: str,
    ann_index_resource_uri: str,
    ann_index_endpoint_resource_uri: str,
) -> NamedTuple('Outputs', [
                            ('ann_index_endpoint_resource_uri', str),
                            ('ann_index_resource_uri', str),
                            ('deployed_ann_index_name', str),
                            ('deployed_ann_index', Artifact),
]):
  
  from google.cloud import aiplatform
  from datetime import datetime
  import logging
  # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

  aiplatform.init(project=project, location=location)

  ann_index = aiplatform.MatchingEngineIndex(
      index_name=ann_index_resource_uri
  )
  ann_index_resource_uri = ann_index.resource_name

  index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
      ann_index_endpoint_resource_uri
  )

  logging.info(f"Deploying ANN index to {ann_index_endpoint_resource_uri}")
  
  index_endpoint = index_endpoint.deploy_index(
      index=ann_index, 
      deployed_index_id=f'{deployed_ann_index_name}'
  )

  logging.info(f"Index deployed: {index_endpoint.deployed_indexes}")

  return (
      f'{ann_index_endpoint_resource_uri}',
      f'{ann_index_resource_uri}',
      f'{deployed_ann_index_name}',
      ann_index,
  )
