@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',
                         'tensorflow',
                         'google-cloud-storage',
    ],
    output_component_file="./pipelines/generate_candidate_embedding_index.yaml",     
)
def generate_candidate_embedding_index(
    project: str,
    location: str,
    version: str,
    gcs_bucket_name: str,
    model_dir: str,
    candidate_items_prefix: str,
    embedding_index_destination_gcs_uri: str,
    uploaded_candidate_model_resources: str,
) -> NamedTuple('Outputs', [('candidate_embedding_index_file_uri', str),
                            ('embedding_index_gcs_bucket', str),
]):

    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob
    
    from google.cloud import aiplatform
    import tensorflow as tf
    from datetime import datetime
    import logging
    import os
    import numpy as np
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # initialize clients
    client = storage.Client(project=project)
    aiplatform.init(project=project,location=location)

    ############################################################################
    # Helper Functions
    ############################################################################

    def _upload_blob_gcs(gcs_uri, source_file_name, destination_blob_name):
      """Uploads a file to GCS bucket"""
      client = storage.Client(project=project)
      blob = Blob.from_string(os.path.join(gcs_uri, destination_blob_name))
      blob.bucket._client = client
      blob.upload_from_filename(source_file_name)

    sequence_features_cat = {
          "productTypeCombo_ss": tf.io.RaggedFeature(tf.string),
          # "Searchable_t": tf.io.RaggedFeature(tf.string),
          # "spellcheck": tf.io.RaggedFeature(tf.string)
    }

    context_features_cat = {
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
            "visual": tf.io.FixedLenFeature(dtype=tf.float32, shape=(2048))
        }

    @tf.function()
    def parse_tfrecord_fn_cat(example):
        example = tf.io.parse_single_sequence_example(
            example, 
            sequence_features=sequence_features_cat, 
            context_features=context_features_cat
        )
        return example
    
    def return_tensors2(context, sequence):
        a = sequence['productTypeCombo_ss'].to_tensor(default_value='', shape=[None, 8])
        context2 = context.copy()
        context2['productTypeCombo_ss'] = a
        return context2

    ################################################################################
    # Candidate Dataset
    ################################################################################

    logging.info(f'Creating parsed candidate dataset from gs://{gcs_bucket_name}/{candidate_items_prefix}')

    files_cat = []
    # for blob in client.list_blobs('limestone-recsys', prefix='prod-catalog-full/2022-05-05204110/', delimiter="/"):
    for blob in client.list_blobs(f'{gcs_bucket_name}', prefix=f'{candidate_items_prefix}', delimiter="/"):  # 'prod-catalog-central'
      files_cat.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    candidate_dataset = tf.data.TFRecordDataset(files_cat)

    parsed_dataset_candidates = candidate_dataset.map(
        parse_tfrecord_fn_cat,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    parsed_dataset_candidates = parsed_dataset_candidates.map(return_tensors2, 
                                                              num_parallel_calls=tf.data.AUTOTUNE).cache()

    logging.info(f'Candidate dataset created in memory')

    ################################################################################
    # Candidate Model
    ################################################################################

    # load candidate model
    loaded_candidate_model = tf.saved_model.load(model_dir)
    logging.info(f'Candidate Model loaded from {model_dir}')


    # SERVING SIGNATURE
    # Note the output shape is how we want it - the shape of the embedding
    predictor = loaded_candidate_model.signatures['serving_default']
    logging.info(f'Candidate Model Serving Function Output Shapes: {predictor.output_shapes}')

    # Embedding iterator
    embs_iter = parsed_dataset_candidates.batch(1).map(
        lambda data: predictor(
            price_td = data["price_td"],
            PriceRange_s = data['PriceRange_s'],
            # description = data['description'],
            IVM_s = data['IVM_s'],
            is_energy_star_s = data['is_energy_star_s'],
            #dense features
            overall_ratings = data['overall_ratings'],
            avg_rating_td = data['avg_rating_td'],
            quantity_sold = data['quantity_sold'],
            sales_dollar_f = data['sales_dollar_f'],
            prc_rdc_amt = data['prc_rdc_amt'],
            # self.brand_clean_embedding(data['clean_Brand_s']),
            item_type = data['item_type'],
            visual = data['visual'],
            total_ratings_i = data['total_ratings_i'],
            productTypeCombo_ss = data['productTypeCombo_ss'],
            # parent_description = data['parent_description'],
            freight_term = data['freight_term'],
            # clean_Brand_s = data['clean_Brand_s'],
            Brand_s = data['Brand_s']
        )
    )
    
    embs = []
    for emb in embs_iter:
        embs.append(emb)

    embs = [x['output_1'].numpy()[0] for x in embs] #clean up the output

    # clean product IDs
    ivm_s = [x['IVM_s'] for x in parsed_dataset_candidates]
    ivm_s = [str(z.numpy()[0]).replace("b'","").replace("'","") for z in ivm_s]

    ################################################################################
    # Check for NaNs
    ################################################################################
    bad_records = []

    for i, emb in enumerate(embs):
      bool_emb = np.isnan(emb)
      for val in bool_emb:
        if val:
          bad_records.append(i)

    bad_record_filter = np.unique(bad_records)
    bad_record_filter

    ivm_s_clean = []
    emb_clean = []

    for i, pair in enumerate(zip(ivm_s, embs)):
        if i in bad_record_filter:
            pass
        else:
            ivm, emb = pair
            ivm_s_clean.append(ivm)
            emb_clean.append(emb)

    logging.info(f"After cleaning there are {len(ivm_s_clean)} IVM records and {len(emb_clean)} embeddings")
    
    logging.info(f'Example of cleaned Product ID: {ivm_s_clean[0]}')

    logging.info(f'Example of cleaned Embedding: {emb_clean[0]}')
      
    ################################################################################
    # Write Index json file
    ################################################################################

    embeddings_index_filename = f'candidate_embeddings_{version}_{TIMESTAMP}.json'

    logging.info(f"Writing index file: {embeddings_index_filename}") # matching-engine-for-towers/dist_indexes

    with open(f'{embeddings_index_filename}', 'w') as f:
      for prod, emb in zip(ivm_s_clean, emb_clean):
        f.write('{"id":"' + str(prod) + '",')
        f.write('"embedding":[' + ",".join(str(x) for x in list(emb)) + "]}")
        f.write("\n")

    _embedding_index_destination_gcs_uri = embedding_index_destination_gcs_uri + f"/{version}-{TIMESTAMP}"

    logging.info(f"Saving {embeddings_index_filename} to {embedding_index_destination_gcs_uri}")

    _upload_blob_gcs(
        _embedding_index_destination_gcs_uri, 
        f"{embeddings_index_filename}", 
        f"{embeddings_index_filename}",
    )
    
    embedding_index_file_uri = f'{_embedding_index_destination_gcs_uri}/{embeddings_index_filename}'
    logging.info(f"{embeddings_index_filename} saved to {embedding_index_file_uri}")

    return (
        f'{embedding_index_file_uri}',
        f'{_embedding_index_destination_gcs_uri}/',
    )
