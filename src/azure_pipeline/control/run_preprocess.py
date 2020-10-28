from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig

ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name='pre-process')

config = ScriptRunConfig(source_directory='src/azure_pipeline/src/', script='preprocess.py', compute_target='cpu-cluster')

# set up  environment
env = Environment.from_conda_specification(name='env', file_path='.azureml/env.yml')
config.run_config.environment = env

run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)
