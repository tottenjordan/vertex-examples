@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',
                         'google-cloud-storage',
                         'google-cloud-pipeline-components'
    ],
    output_component_file="./pipelines/test_serving_ann_index.yaml",
)
def test_serving_ann_index(
    project: str,
    location: str,
    version: str,
    model_endpoint_uri: str,
    num_neighbors: int,
    ann_index_endpoint_resource_uri: str,
    brute_index_endpoint_resource_uri: str,
    deployed_ann_index_name: str,
    deployed_brute_force_index_name: str,
    deployed_test_destination_gcs_uri: str,
    metrics: Output[Metrics],
) -> NamedTuple('Outputs', [('deployed_test_gcs_uri', str),]):
  
    import json
    import os
    import time
    from datetime import datetime
    from typing import Dict, List, Union
    import logging

    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    # from google.protobuf.json_format import Parse

    # from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob

    logging.getLogger().setLevel(logging.INFO)
    aiplatform.init(project=project, location=location)
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    ############################################################################
    # Helper functions 
    ############################################################################

    def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
      """Uploads a file to GCS bucket"""
      client = storage.Client(project=project)
      blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
      blob.bucket._client = client
      blob.upload_from_filename(source_file_name)

    ############################################################################
    # Define Model and Index Endpoint Resources
    ############################################################################

    logging.info(f"Endpoint URI = {model_endpoint_uri}")

    # Define Model Endpoint Resource in component
    _model_endpoint_resource = aiplatform.Endpoint(model_endpoint_uri)

    
    # Define ANN Index Endpoint Resources
    logging.info(f"ANN Index Endpoint uri = {ann_index_endpoint_resource_uri}")

    _ann_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(ann_index_endpoint_resource_uri)

    logging.info(f"ANN Index Endpoint initialized = {_ann_index_endpoint}")

    
    # Define Brute Force Index Endpoint Resources
    logging.info(f"Brute Force Index Endpoint uri = {brute_index_endpoint_resource_uri}")

    _brute_force_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(brute_index_endpoint_resource_uri)

    logging.info(f"Brute Force Index Endpoint initialized = {_brute_force_index_endpoint}")

    ############################################################################
    # Define test instance(s)
    ############################################################################

    from google.cloud import storage

    client = storage.Client()
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    files = []
    for blob in client.list_blobs('limestone-recsys', prefix='tf-record-all-features/2022-05-05174504/', delimiter="/"):
        files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    # Query Features
    sequence_features_ = {
        "last_viewed": tf.io.RaggedFeature(tf.string),
          "productTypeCombo_ss": tf.io.RaggedFeature(tf.string),
          # "Searchable_t": tf.io.RaggedFeature(tf.string),
          # "spellcheck": tf.io.RaggedFeature(tf.string)
    }

    context_features_ = {
            "query": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "IVM_s": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            # "description": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "total_ratings_i": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1)), 
            "overall_ratings": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1)), 
            "avg_rating_td": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1)), 
            # "parent_description": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "Brand_s": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "item_type": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "prc_rdc_amt": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1)), 
            "quantity_sold": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1)), 
            "sales_dollar_f": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1)),  
            "freight_term": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "is_energy_star_s": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "price_td": tf.io.FixedLenFeature(dtype=tf.float32, shape=(1)), 
            "PriceRange_s": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            # "clean_Brand_s": tf.io.FixedLenFeature(dtype=tf.string, shape=(1)), 
            "visual": tf.io.FixedLenFeature(dtype=tf.float32, shape=(2048)), 
            "month": tf.io.FixedLenFeature(dtype=tf.int64, shape=(1)),
            "hour": tf.io.FixedLenFeature(dtype=tf.int64, shape=(1))
        }

    @tf.function()
    def parse_tfrecord_fn(example):
      example = tf.io.parse_single_sequence_example(
          example, 
          sequence_features=sequence_features_, 
          context_features=context_features_
      )
      return example

    raw_dataset = tf.data.TFRecordDataset(files[:3])

    parsed_dataset = raw_dataset.map(
            parse_tfrecord_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        ).with_options(options)

    ################################################################################
    # Prediction instances # TODO: more robust test
    ################################################################################

    test_prediction_instance =  {
        "month" : [2], 
        "query" : ["lightbulbs vintage"], 
        "last_viewed" : [['1500755-371-93121158', '1500779-337-LT560WH6930R-6PK', '']], 
        "hour" : [12], 
        "Brand_s" : 'x', 
        "IVM_s" : 'x', 
        "PriceRange_s" : 'x',
        "avg_rating_td" : .0, 
        # "clean_Brand_s" : 'x',
        # "description" : 'x',
        "freight_term" : 'x',
        "is_energy_star_s" : 'x',
        "item_type" : 'x',
        "overall_ratings" : .0, 
        # "parent_description" : 'x',
        "prc_rdc_amt" : .0, 
        "price_td" : .0,
        "productTypeCombo_ss" : 'x', 
        "quantity_sold" : .0,
        "sales_dollar_f" : .0, 
        "total_ratings_i" : .0, 
        "visual" : .0,
    }

    # prediction_test = predict_custom_trained_model_sample(
    #     project=project,                     
    #     endpoint_id=_endpoint_id,
    #     location="us-central1",
    #     instances=test_prediction_instance
    # )

    ############################################################################
    # Query Tower Prediction -> Query Index -> Retrieve Neighbors
    ############################################################################

    start = time.time()
    query_embedding_vector = _model_endpoint_resource.predict(instances=[test_prediction_instance])
    model_predict_delta = time.time() - start

    logging.info(f'query_embedding_vector: {query_embedding_vector}')
    logging.info(f'query_embedding_vector.predictions: {query_embedding_vector.predictions}')

    # ANN lookup
    ann_start = time.time()
    ann_response = _ann_index_endpoint.match(
        deployed_index_id=deployed_ann_index_name,
        queries=query_embedding_vector.predictions,
        num_neighbors=num_neighbors,
    )
    ann_end = time.time()
    ann_retrieval_delta = ann_start - ann_end

    # Brute force lookup
    brute_start = time.time()
    brute_force_response = _brute_force_index_endpoint.match(
        deployed_index_id=deployed_brute_force_index_name, 
        queries=query_embedding_vector.predictions, 
        num_neighbors=num_neighbors,
    )
    brute_end = time.time()
    brute_force_retrieval_delta = brute_start - brute_end

    # Calculate recall by determining how many neighbors were correctly retrieved as compared to the brute-force option.
    correct_neighbors = 0
    for tree_ah_neighbors, brute_force_neighbors in zip(ann_response, brute_force_response):
      tree_ah_neighbor_ids = [neighbor.id for neighbor in tree_ah_neighbors]
      brute_force_neighbor_ids = [neighbor.id for neighbor in brute_force_neighbors]

      correct_neighbors += len(
          set(tree_ah_neighbor_ids).intersection(brute_force_neighbor_ids)
      )

    recall = correct_neighbors / (len(test_prediction_instance) * num_neighbors)
    
    logging.info(f'Model Prediction latency: {round(model_predict_delta, 4)}')
    logging.info(f'ANN retrieval latency: {round(ann_retrieval_delta, 4)}')
    logging.info(f'Brute Force retrieval latency: {round(brute_force_retrieval_delta, 4)}')
    
    logging.info(f'Recall: {recall}')
    metrics.log_metric("Recall", recall)
    metrics.log_metric("Index", "ANN index") 
    metrics.log_metric("Index Endpoint", f"{ann_index_endpoint_resource_uri}")
    metrics.log_metric("Num Nieghbors", num_neighbors)

    logging.info(f'ANN retrieved neighbors: {ann_response}')
    logging.info(f'Brute Force retrieved neighbors: {brute_force_response}')

    test_filename = f'test_deployed_model_indexes_{TIMESTAMP}.json'
    with open(f'{test_filename}', 'w') as f:
      for neighbor in ann_response:
        f.write(f"ANN neighbors, retieved in {round(ann_retrieval_delta, 4)}:")
        f.write(f'{neighbor}')
        f.write("\n")

      for neighbor in brute_force_response:
        f.write(f"Brute Force neighbors, retieved in {round(brute_force_retrieval_delta, 4)}:")
        f.write(f'{neighbor}')
        f.write("\n")

    deployed_test_gcs_bucket_uri = deployed_test_destination_gcs_uri + f'/{version}-{TIMESTAMP}' #+ f"/{test_filename}"

    logging.info(f"Saving {test_filename} to {deployed_test_gcs_bucket_uri}")

    _upload_blob_gcs(
        deployed_test_gcs_bucket_uri,
        f"{test_filename}", 
        f"{test_filename}",
    )
    deployed_test_gcs_uri = deployed_test_gcs_bucket_uri + f'/{test_filename}'
    logging.info(f"Deployment test complete")

    return (
        deployed_test_gcs_uri,
    )
