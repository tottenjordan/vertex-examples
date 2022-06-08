@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform==1.13.0"],
    output_component_file="./pipelines/return_unmanaged_model.yaml",
)
def return_unmanaged_model(
    serving_image: str, 
    artifact_uri: str, 
    resource_name: str, 
    model: Output[Artifact]):
  
    model.metadata["containerSpec"] = {"imageUri": serving_image}

    model.metadata["resourceName"] = resource_name

    model.uri = artifact_uri
