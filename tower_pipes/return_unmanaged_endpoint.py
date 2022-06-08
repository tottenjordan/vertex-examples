@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform==1.13.0"],
    output_component_file="./pipelines/return_unmanaged_endpoint.yaml",
)
def return_unmanaged_endpoint(
    resource_name: str, 
    endpoint: Output[Artifact]):

    endpoint.metadata["resourceName"] = resource_name 
