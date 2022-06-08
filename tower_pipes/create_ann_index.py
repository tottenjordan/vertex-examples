@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',
    ],
    output_component_file="./pipelines/create_ann_index.yaml",
)
def create_ann_index(
    project: str,
    location: str,
    version: str,
    vpc_network_name: str,
    emb_index_gcs_bucket_uri: str,
    dimensions: int,
    ann_index_display_name: str,
    approximate_neighbors_count: int,
    distance_measure_type: str,
    leaf_node_embedding_count: int,
    leaf_nodes_to_search_percent: int, 
    ann_index_description: str,
    ann_index_labels: Dict, 
) -> NamedTuple('Outputs', [('ann_index_resource_uri', str),
                            ('ann_index', Artifact),]):


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

  logging.info(f"Creating ANN index")
  ann_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
      display_name=f'{ann_index_display_name}',
      contents_delta_uri=emb_index_gcs_bucket_uri,
      dimensions=dimensions,
      approximate_neighbors_count=approximate_neighbors_count,
      distance_measure_type=distance_measure_type,
      leaf_node_embedding_count=leaf_node_embedding_count,
      leaf_nodes_to_search_percent=leaf_nodes_to_search_percent,
      description=ann_index_description,
      labels=ann_index_labels,
  )

  ann_index_resource_uri = ann_index.resource_name
  logging.info(f"ann_index_resource_uri: {ann_index_resource_uri}")

  return (
      f'{ann_index_resource_uri}',
      ann_index,
  )
