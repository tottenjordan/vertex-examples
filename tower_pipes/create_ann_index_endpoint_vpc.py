@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',    # TODO: update once merged
    ],
    output_component_file="./pipelines/create_ann_index_endpoint_vpc.yaml",
)
def create_ann_index_endpoint_vpc(
    project: str,
    project_number: str,
    location: str,
    version: str,
    vpc_network_name: str,
    ann_index_endpoint_display_name: str,
    ann_index_endpoint_description: str,
) -> NamedTuple('Outputs', [
                            ('vpc_network_resource_uri', str),
                            ('ann_index_endpoint_resource_uri', str),
                            ('ann_index_endpoint', Artifact),
                            ('ann_index_endpoint_display_name', str),
]):

  from google.cloud import aiplatform
  from datetime import datetime
  import logging

  aiplatform.init(project=project, location=location)
  # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

  vpc_network_resource_uri = f'projects/{project_number}/global/networks/{vpc_network_name}'
  logging.info(f"vpc_network_resource_uri: {vpc_network_resource_uri}")

  ann_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
      display_name=f'{ann_index_endpoint_display_name}',
      description=ann_index_endpoint_description,
      network=vpc_network_resource_uri,
  )
  ann_index_endpoint_resource_uri = ann_index_endpoint.resource_name
  logging.info(f"ann_index_endpoint_resource_uri: {ann_index_endpoint_resource_uri}")

  return (
      f'{vpc_network_resource_uri}',
      f'{ann_index_endpoint_resource_uri}',
      ann_index_endpoint,
      f'{ann_index_endpoint_display_name}'
  )
