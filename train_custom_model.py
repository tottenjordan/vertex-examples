@component(
base_image='python:3.9',
packages_to_install=[
                       'google-cloud-aiplatform==1.13.0',
                       'tensorflow',
                       'tensorflow-recommenders==0.6.0',
                       'numpy',
                       'google-cloud-storage',
  ])
def train_custom_model(
    project: str,
    version: str,
    job_name: str, 
    worker_pool_specs: dict,
    vocab_dict_uri: str, 
    base_output_dir: str,
    training_image_uri: str,
    tensorboard_resource_name: str,
    service_account: str, 
) -> str:

    from google.cloud import aiplatform as vertex_ai

  # #new_wps = []
  # for i, a in enumerate(worker_pool_specs):
  #   if a['container_spec']['args'] is None:
  #     pass 
  #   else:
  #     arg = a['container_spec']['args']
  #     worker_pool_specs[i]['container_spec']['args'] = arg.append(f'--VOCAB_DICT={vocab_dict_uri}')


    vertex_ai.init(
        project=project,
        location='us-central1',
        staging_bucket=base_output_dir,
    )
  
    job = vertex_ai.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=base_output_dir,
    )

    job.run(
        tensorboard=tensorboard_resource_name,
        service_account=f'{service_account}',
    )

    return("finished training")
