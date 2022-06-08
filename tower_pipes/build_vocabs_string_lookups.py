@kfp.v2.dsl.component(
  base_image='python:3.9',
  packages_to_install=[
                       'google-cloud-aiplatform==1.13.0',
                       'tensorflow',
                       'tensorflow-recommenders==0.6.0',
                       'numpy',
                       'google-cloud-storage',
  ],
  output_component_file="./pipelines/build_vocabs_string_lookups.yaml",
)
def build_vocabs_string_lookups(
    project: str,
    location: str,
    version: str,
    gcs_bucket_name: str,
    app_name: str,
) -> NamedTuple('Outputs', [
                            ('vocab_dict', Artifact),
                            ('vocab_gcs_filename', str),
                            ('vocab_gcs_sub_dir', str),
                            ('vocab_gcs_uri', str),
]):

  from google.cloud import aiplatform
  import json
  import logging
  import json
  import tensorflow as tf
  import tensorflow_recommenders as tfrs
  import numpy as np
  import pickle as pkl
  from google.cloud import storage
  from datetime import datetime

  TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

  aiplatform.init(
      project=project,
      location=location,
  )

  ################################################################################
  # Helper Functions for feature parsing
  ################################################################################

  sequence_features_ = {
      "last_viewed": tf.io.RaggedFeature(tf.string),
        "productTypeCombo_ss": tf.io.RaggedFeature(tf.string),
        "Searchable_t": tf.io.RaggedFeature(tf.string),
        "spellcheck": tf.io.RaggedFeature(tf.string)
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


  sequence_features_cat = {
        "productTypeCombo_ss": tf.io.RaggedFeature(tf.string),
        "Searchable_t": tf.io.RaggedFeature(tf.string),
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

  def parse_tfrecord_fn_cat(example):
      example = tf.io.parse_single_sequence_example(example, sequence_features=sequence_features_cat, context_features=context_features_cat)
      return example

  def parse_tfrecord_fn(example):
      example = tf.io.parse_single_sequence_example(example, sequence_features=sequence_features_, context_features=context_features_)
      return example

  client = storage.Client()

  files = []
  for blob in client.list_blobs('limestone-recsys', prefix='tf-record-all-features/2022-05-05174504/', delimiter="/"):
      files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))


  files_cat = []
  for blob in client.list_blobs('limestone-recsys', prefix='prod-catalog-full/2022-05-05204110/', delimiter="/"):
      files_cat.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

  raw_dataset = tf.data.TFRecordDataset(files) #local machine training wheels - using smaller data set for starters
  cat_dataset = tf.data.TFRecordDataset(files_cat)


  parsed_dataset = raw_dataset.map(
          parse_tfrecord_fn)

  ################################################################################
  # Get vocabularies of unique values for strings
  ################################################################################

  import numpy as np

  unique_IVM_s = np.unique(
      np.concatenate(
          list(
              parsed_dataset.map(
                  lambda x, y: x['IVM_s']
              )
              .batch(1000)
          )
      )
  )
  
  unique_clean_Brand_s = np.unique(
      np.concatenate(
          list(
              parsed_dataset.map(
                  lambda x, y: x['clean_Brand_s']
              )
              .batch(1000)
          )
      )
  )
  
  unique_freight_term = np.unique(
      np.concatenate(
          list(
              parsed_dataset.map(
                  lambda x, y: x['freight_term']
              )
              .batch(1000)
          )
      )
  )
  
  unique_item_type = np.unique(
      np.concatenate(
          list(
              parsed_dataset.map(
                  lambda x, y: x['item_type']
              )
              .batch(1000)
          )
      )
  )

  ################################################################################
  # Iterate over nested and ragged data stored in the sequence features
  ################################################################################

  def ragged_unique_collection(field):
      data = np.array([b''])
      for x in parsed_dataset.map(lambda x, y: y[field]).batch(1000):
          y = np.unique(np.concatenate(np.concatenate(x.numpy())))
          data = np.concatenate([data, y])
      data = np.unique(data)
      return(data)

  unique_productTypeCombo_ss = ragged_unique_collection('productTypeCombo_ss')
  # unique_Searchable_t = ragged_unique_collection('Searchable_t')
  # unique_spellcheck = ragged_unique_collection('spellcheck')
  unique_last_viewed = ragged_unique_collection('last_viewed')

  ################################################################################
  # Save Vocab
  ################################################################################

  import pickle as pkl

  vocab_dict = {
      'IVM_s_vocab': unique_IVM_s,
      'clean_Brand_s_vocab': unique_clean_Brand_s,
      'freight_term_vocab': unique_freight_term,
      'item_type_vocab': unique_item_type,
      'productTypeCombo_ss_vocab': unique_productTypeCombo_ss,
      'last_viewed_vocab': unique_last_viewed,
  }
  
  VOCAB_FILENAME = f'string_vocabs_{version}_{TIMESTAMP}'
  VOCAB_GCS_SUB_DIR = f'{app_name}/{version}/vocabs'
  VOCAB_GCS_URI = f'gs://{gcs_bucket_name}/{VOCAB_GCS_SUB_DIR}/{VOCAB_FILENAME}'

  with open(f'{VOCAB_FILENAME}', 'wb') as handle:
          pkl.dump(vocab_dict, handle)
              
  # JOB_DIR = gcs_bucket_name # 'trfs-tf-bucket' # TODO: paramaterize 
  
  bucket = client.bucket(gcs_bucket_name)
  bucket= client.get_bucket(gcs_bucket_name);  
  blob = bucket.blob(f'{VOCAB_GCS_SUB_DIR}/{VOCAB_FILENAME}')
  blob.upload_from_filename(f'{VOCAB_FILENAME}')

  return (
      vocab_dict,
      f'{VOCAB_FILENAME}',
      f'{VOCAB_GCS_SUB_DIR}',
      f'{VOCAB_GCS_URI}',
  )
