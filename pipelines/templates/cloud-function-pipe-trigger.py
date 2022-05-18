# Copyright 2021 Google LLC.
# SPDX-License-Identifier: Apache-2.0
import kfp
import json
import time
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from kfp.v2.google.client import AIPlatformClient

client = bigquery.Client()
RETRAIN_THRESHOLD = 1000 # Change this based on your use case

def insert_bq_data(table_id, num_rows):
    rows_to_insert = [
        {u"num_rows_last_retraining": num_rows, u"last_retrain_time": time.time()}
    ]

    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors == []:
        print("New rows have been added.")
    else:
        print(f"Encountered errors while inserting rows: {errors}")

def create_count_table(table_id, num_rows):
    schema = [
        bigquery.SchemaField("num_rows_last_retraining", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("last_retrain_time", "TIMESTAMP", mode="REQUIRED")
    ]
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)
    print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

    insert_bq_data(table_id, num_rows)
    
def create_pipeline_run():
    print('Kicking off a pipeline run...')
    
    REGION = "us-central1" # Change this to the region you want to run in
    api_client = AIPlatformClient(
        project_id=client.project,
        region=REGION,
    )
    try:
        response = api_client.create_run_from_job_spec(
            "compiled_pipeline.json",                              # TODO
            pipeline_root="gs://your-gcs-bucket/pipeline_root/",   # TODO
            parameter_values={"project": client.project, "display_name": "pipeline_gcf_trigger"}
        )
        return response
    except:
        print("Error trying to run the pipeline")
        raise

# This should be the entrypoint for your Cloud Function
def check_table_size(request):
    request = request.get_data()

    try: 
        request_json = json.loads(request.decode())
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
        return "JSON Error", 400
    
    if request_json and 'bq_dataset' in request_json:
        dataset = request_json['bq_dataset']
        table = request_json['bq_table']

        data_table = client.get_table(f"{client.project}.{dataset}.{table}")
        current_rows = data_table.num_rows
        print(f"{table} table has {current_rows} rows")

        # See if `count` table exists in dataset
        try:
            count_table = client.get_table(f"{client.project}.{dataset}.count")
            print("Count table exists, querying to see how many rows at last pipeline run")

        except NotFound:
            print("No count table found, creating one...")
            create_count_table(f"{client.project}.{dataset}.count", current_rows)
    
        query_job = client.query(
            """
            SELECT num_rows_last_retraining FROM `your-project.your-dataset.count`            
            ORDER BY last_retrain_time DESC
            LIMIT 1"""
        )                               # TODO ^

        results = query_job.result()
        for i in results:
            last_retrain_count = i[0]

        rows_added_since_last_pipeline_run = current_rows - last_retrain_count
        print(f"{rows_added_since_last_pipeline_run} rows have been added since we last ran the pipeline")

        if (rows_added_since_last_pipeline_run >= RETRAIN_THRESHOLD):
            pipeline_result = create_pipeline_run()
            insert_bq_data(f"{client.project}.{dataset}.count", current_rows)
    else:
        return f"No BigQuery data given"
