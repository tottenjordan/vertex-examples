@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',    # TODO: update once merged
    ],
    output_component_file="./pipelines/create_brute_force_index.yaml",
)
def create_brute_force_index(
    project: str,
    location: str,
    version: str,
    vpc_network_name: str,
    emb_index_gcs_bucket_uri: str,
    dimensions: int,
    brute_force_index_display_name: str,
    approximate_neighbors_count: int,
    distance_measure_type: str,
    brute_force_index_description: str,
    brute_force_index_labels: Dict,
) -> NamedTuple('Outputs', [('brute_force_index_resource_uri', str),
                            ('brute_force_index', Artifact),]):


  from google.cloud import aiplatform
  from datetime import datetime
  import logging

  aiplatform.init(project=project, location=location)
  # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

  ENDPOINT = "{}-aiplatform.googleapis.com".format(location)
  PARENT = "projects/{}/locations/{}".format(project, location)

  logging.info(f"ENDPOINT: {ENDPOINT}")
  logging.info(f"PROJECT_ID: {project}")
  logging.info(f"REGION: {location}")
  logging.info(f"NETWORK_NAME: {vpc_network_name}")
  logging.info(f"PARENT: {PARENT}")

  logging.info(f"Creating Brute Force index")
  brute_force_index = aiplatform.MatchingEngineIndex.create_brute_force_index(
      display_name=f'{brute_force_index_display_name}',
      contents_delta_uri=emb_index_gcs_bucket_uri,
      dimensions=dimensions,
      # approximate_neighbors_count=approximate_neighbors_count,
      distance_measure_type=distance_measure_type,
      description=brute_force_index_description,
      labels=brute_force_index_labels,
  )
  brute_force_index_resource_uri = brute_force_index.resource_name
  logging.info(f"brute_force_index_resource_uri: {brute_force_index_resource_uri}")

  return (
      f'{brute_force_index_resource_uri}',
      brute_force_index,
  )
