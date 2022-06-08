@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
                         'google-cloud-aiplatform==1.13.0',
                         'google-cloud-pipeline-components',
    ],
    output_component_file="./pipelines/test_deployed_existing_endpoint.yaml",
                      
)
def test_deployed_existing_endpoint(
    project: str,
    location: str,
    endpoint_uri: str,
) -> NamedTuple('Outputs', [
                            ('endpoint_uri', str),
]):

    import base64
    import logging

    from typing import Dict, List, Union

    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Value

    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    logging.getLogger().setLevel(logging.INFO)
    aiplatform.init(project=project, location=location)

    # define endpoint resource in component
    logging.info(f"endpoint_uri = {endpoint_uri}")
    _endpoint = aiplatform.Endpoint(endpoint_uri)

    ################################################################################
    # Helper function for returning endpoint predictions via required json format
    ################################################################################

    def predict_custom_trained_model_sample(
        project: str,
        endpoint_id: str,
        instances: Dict,
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
        """
        `instances` can be either single instance of type dict or a list
        of instances.
        """

        ########################################################################
        # Initialize Vertex Endpoint
        ########################################################################

        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": api_endpoint}
        
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        
        # The format of each instance should conform to the deployed model's prediction input schema.
        instances = instances if type(instances) == list else [instances]
        instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]
        
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        
        endpoint = client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )
        
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        logging.info(f'Response: {response}')
        
        logging.info(f'Deployed Model ID(s): {response.deployed_model_id}')

        # The predictions are a google.protobuf.Value representation of the model's predictions.
        _predictions = response.predictions
        logging.info(f'Response Predictions: {_predictions}')
        
        return _predictions

    ################################################################################
    # Request Prediction
    ################################################################################
    
    # IVM_s,PriceRange_s,description,hour,last_viewed,month,price_td,productTypeCombo_ss,query,visual"

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

    prediction_test = predict_custom_trained_model_sample(
        project=project,                     
        endpoint_id=endpoint_uri,
        location="us-central1",
        instances=test_prediction_instance
    )

    return (
        f'{endpoint_uri}',
    )
