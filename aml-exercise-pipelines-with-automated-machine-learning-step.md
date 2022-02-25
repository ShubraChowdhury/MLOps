Copyright (c) Microsoft Corporation. All rights reserved.  
Licensed under the MIT License.

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/NotebookVM/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-with-automated-machine-learning-step.png)

# Azure Machine Learning Pipeline with AutoMLStep (Udacity Course 2)
This notebook demonstrates the use of AutoMLStep in Azure Machine Learning Pipeline.

## Introduction
In this example we showcase how you can use AzureML Dataset to load data for AutoML via AML Pipeline. 

If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, make sure you have executed the [configuration](https://aka.ms/pl-config) before running this notebook.

In this notebook you will learn how to:
1. Create an `Experiment` in an existing `Workspace`.
2. Create or Attach existing AmlCompute to a workspace.
3. Define data loading in a `TabularDataset`.
4. Configure AutoML using `AutoMLConfig`.
5. Use AutoMLStep
6. Train the model using AmlCompute
7. Explore the results.
8. Test the best fitted model.

## Azure Machine Learning and Pipeline SDK-specific imports


```
import logging
import os
import csv

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import pkg_resources

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset

from azureml.pipeline.steps import AutoMLStep

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)
```

    SDK version: 1.38.0


## Initialize Workspace
Initialize a workspace object from persisted configuration. Make sure the config file is present at .\config.json


```
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
```

    quick-starts-ws-187534
    aml-quickstarts-187534
    southcentralus
    5a4ab2ba-6c51-4805-8155-58759ad589d8


## Create an Azure ML experiment
Let's create an experiment named "automlstep-classification" and a folder to hold the training scripts. The script runs will be recorded under the experiment in Azure.

The best practice is to use separate folders for scripts and its dependent files for each step and specify that folder as the `source_directory` for the step. This helps reduce the size of the snapshot created for the step (only the specific folder is snapshotted). Since changes in any files in the `source_directory` would trigger a re-upload of the snapshot, this helps keep the reuse of the step when there are no changes in the `source_directory` of the step.

*Udacity Note:* There is no need to create an Azure ML experiment, this needs to re-use the experiment that was already created



```
# Choose a name for the run history container in the workspace.
# NOTE: update these to match your existing experiment name
experiment_name = 'ml-bike-experiment-1'
project_folder = './pipeline-bike-project'

experiment = Experiment(ws, experiment_name)
experiment
```




<table style="width:100%"><tr><th>Name</th><th>Workspace</th><th>Report Page</th><th>Docs Page</th></tr><tr><td>ml-bike-experiment-1</td><td>quick-starts-ws-187534</td><td><a href="https://ml.azure.com/experiments/id/ed7ab917-fa3b-4166-aaa1-d043922d58ac?wsid=/subscriptions/5a4ab2ba-6c51-4805-8155-58759ad589d8/resourcegroups/aml-quickstarts-187534/workspaces/quick-starts-ws-187534&amp;tid=660b3398-b80e-49d2-bc5b-ac1dc93b5254" target="_blank" rel="noopener">Link to Azure Machine Learning studio</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment.Experiment?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>



### Create or Attach an AmlCompute cluster
You will need to create a [compute target](https://docs.microsoft.com/azure/machine-learning/service/concept-azure-machine-learning-architecture#compute-target) for your AutoML run. In this tutorial, you get the default `AmlCompute` as your training compute resource.

**Udacity Note** There is no need to create a new compute target, it can re-use the previous cluster


```
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# NOTE: update the cluster name to match the existing cluster
# Choose a name for your CPU cluster
#amlcompute_cluster_name = "auto-ml"
amlcompute_cluster_name = "aml-compute"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',# for GPU, use "STANDARD_NC6"
                                                           #vm_priority = 'lowpriority', # optional
                                                           max_nodes=4)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True, min_node_count = 1, timeout_in_minutes = 10)
# For a more detailed view of current AmlCompute status, use get_status().
```

    Found existing cluster, use it.
    Succeeded.....................................................................................................................
    AmlCompute wait for completion finished
    
    Wait timeout has been reached
    Current provisioning state of AmlCompute is "Succeeded" and current node count is "0"


## Data

**Udacity note:** Make sure the `key` is the same name as the dataset that is uploaded, and that the description matches. If it is hard to find or unknown, loop over the `ws.datasets.keys()` and `print()` them.
If it *isn't* found because it was deleted, it can be recreated with the link that has the CSV 


```
# Try to load the dataset from the Workspace. Otherwise, create it from the file
# NOTE: update the key to match the dataset name
found = False
key = "Bikesharing Dataset"
description_text = "Bike Sharing DataSet for Udacity Course 2"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://raw.githubusercontent.com/Azure/MachineLearningNotebooks/master/how-to-use-azureml/automated-machine-learning/forecasting-bike-share/bike-no.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)


df = dataset.to_pandas_dataframe()
#df.describe()
```

### Review the Dataset Result

You can peek the result of a TabularDataset at any range using `skip(i)` and `take(j).to_pandas_dataframe()`. Doing so evaluates only `j` records for all the steps in the TabularDataset, which makes it fast even against large datasets.

`TabularDataset` objects are composed of a list of transformation steps (optional).


```
dataset.take(5).to_pandas_dataframe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>date</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>weekday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>0.344167</td>
      <td>0.363625</td>
      <td>0.805833</td>
      <td>0.160446</td>
      <td>331</td>
      <td>654</td>
      <td>985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-02</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0.363478</td>
      <td>0.353739</td>
      <td>0.696087</td>
      <td>0.248539</td>
      <td>131</td>
      <td>670</td>
      <td>801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-03</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.196364</td>
      <td>0.189405</td>
      <td>0.437273</td>
      <td>0.248309</td>
      <td>120</td>
      <td>1229</td>
      <td>1349</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-04</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.200000</td>
      <td>0.212122</td>
      <td>0.590435</td>
      <td>0.160296</td>
      <td>108</td>
      <td>1454</td>
      <td>1562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-05</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0.226957</td>
      <td>0.229270</td>
      <td>0.436957</td>
      <td>0.186900</td>
      <td>82</td>
      <td>1518</td>
      <td>1600</td>
    </tr>
  </tbody>
</table>
</div>



## Train
This creates a general AutoML settings object.
**Udacity notes:** These inputs must match what was used when training in the portal. `time_column_name` has to be `cnt` for example.


```
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'normalized_root_mean_squared_error',
    "n_cross_validations": 3
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "forecasting",
                             training_data=dataset,
                             time_column_name="date", 
                             label_column_name="cnt",  
                             path = project_folder,
                             enable_early_stopping= True,
                             #featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
```

#### Create Pipeline and AutoMLStep

You can define outputs for the AutoMLStep using TrainingOutput.


```
from azureml.pipeline.core import PipelineData, TrainingOutput

ds = ws.get_default_datastore()
metrics_output_name = 'metrics_output'
best_model_output_name = 'best_model_output'

metrics_data = PipelineData(name='metrics_data',
                           datastore=ds,
                           pipeline_output_name=metrics_output_name,
                           training_output=TrainingOutput(type='Metrics'))
model_data = PipelineData(name='model_data',
                           datastore=ds,
                           pipeline_output_name=best_model_output_name,
                           training_output=TrainingOutput(type='Model'))
```

Create an AutoMLStep.


```
automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)
```


```
from azureml.pipeline.core import Pipeline
pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,    
    steps=[automl_step])
```


```
pipeline_run = experiment.submit(pipeline)
```

    Created step automl_module [e92ca11c][3543584e-1a16-4270-bfa8-d262c74bbb91], (This step will run and generate new outputs)
    Submitted PipelineRun 5dd4f320-7a3a-45a5-b874-3b1d403bd3ae
    Link to Azure Machine Learning Portal: https://ml.azure.com/runs/5dd4f320-7a3a-45a5-b874-3b1d403bd3ae?wsid=/subscriptions/5a4ab2ba-6c51-4805-8155-58759ad589d8/resourcegroups/aml-quickstarts-187534/workspaces/quick-starts-ws-187534&tid=660b3398-b80e-49d2-bc5b-ac1dc93b5254



```
from azureml.widgets import RunDetails
RunDetails(pipeline_run).show()
```


    _PipelineWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO', …





```
pipeline_run.wait_for_completion()
```

    PipelineRunId: 5dd4f320-7a3a-45a5-b874-3b1d403bd3ae
    Link to Azure Machine Learning Portal: https://ml.azure.com/runs/5dd4f320-7a3a-45a5-b874-3b1d403bd3ae?wsid=/subscriptions/5a4ab2ba-6c51-4805-8155-58759ad589d8/resourcegroups/aml-quickstarts-187534/workspaces/quick-starts-ws-187534&tid=660b3398-b80e-49d2-bc5b-ac1dc93b5254
    PipelineRun Status: Running
    
    
    StepRunId: 4d85652b-e60d-4510-abed-8246d05fa168
    Link to Azure Machine Learning Portal: https://ml.azure.com/runs/4d85652b-e60d-4510-abed-8246d05fa168?wsid=/subscriptions/5a4ab2ba-6c51-4805-8155-58759ad589d8/resourcegroups/aml-quickstarts-187534/workspaces/quick-starts-ws-187534&tid=660b3398-b80e-49d2-bc5b-ac1dc93b5254
    StepRun( automl_module ) Status: Running
    
    StepRun(automl_module) Execution Summary
    =========================================
    StepRun( automl_module ) Status: Finished
    
    Warnings:
    No scores improved over last 20 iterations, so experiment stopped early. This early stopping behavior can be disabled by setting enable_early_stopping = False in AutoMLConfig for notebook/python SDK runs.
    {'runId': '4d85652b-e60d-4510-abed-8246d05fa168', 'target': 'aml-compute', 'status': 'Completed', 'startTimeUtc': '2022-02-25T19:57:36.315879Z', 'endTimeUtc': '2022-02-25T20:09:16.7118Z', 'services': {}, 'warnings': [{'source': 'JasmineService', 'message': 'No scores improved over last 20 iterations, so experiment stopped early. This early stopping behavior can be disabled by setting enable_early_stopping = False in AutoMLConfig for notebook/python SDK runs.'}], 'properties': {'ContentSnapshotId': '00000000-0000-0000-0000-000000000000', 'StepType': 'AutoMLStep', 'azureml.moduleid': '3543584e-1a16-4270-bfa8-d262c74bbb91', 'azureml.moduleName': 'automl_module', 'azureml.runsource': 'azureml.StepRun', 'azureml.nodeid': 'e92ca11c', 'azureml.pipelinerunid': '5dd4f320-7a3a-45a5-b874-3b1d403bd3ae', 'azureml.pipeline': '5dd4f320-7a3a-45a5-b874-3b1d403bd3ae', 'azureml.pipelineComponent': 'masterautomlcloud', 'num_iterations': '1000', 'training_type': 'TrainFull', 'acquisition_function': 'EI', 'metrics': 'accuracy', 'primary_metric': 'normalized_root_mean_squared_error', 'train_split': '0', 'MaxTimeSeconds': None, 'acquisition_parameter': '0', 'num_cross_validation': '3', 'target': 'aml-compute', 'RawAMLSettingsString': None, 'AMLSettingsJsonString': '{"path": null, "name": "placeholder", "subscription_id": "5a4ab2ba-6c51-4805-8155-58759ad589d8", "resource_group": "aml-quickstarts-187534", "workspace_name": "quick-starts-ws-187534", "region": "southcentralus", "compute_target": "aml-compute", "spark_service": null, "azure_service": null, "many_models": false, "pipeline_fetch_max_batch_size": 1, "enable_batch_run": false, "enable_run_restructure": false, "start_auxiliary_runs_before_parent_complete": false, "enable_code_generation": false, "iterations": 1000, "primary_metric": "normalized_root_mean_squared_error", "task_type": "regression", "positive_label": null, "data_script": null, "test_size": 0.0, "test_include_predictions_only": false, "validation_size": 0.0, "n_cross_validations": 3, "y_min": null, "y_max": null, "num_classes": null, "featurization": "auto", "_ignore_package_version_incompatibilities": false, "is_timeseries": true, "max_cores_per_iteration": 1, "max_concurrent_iterations": 4, "iteration_timeout_minutes": null, "mem_in_mb": null, "enforce_time_on_windows": false, "experiment_timeout_minutes": 20, "experiment_exit_score": null, "whitelist_models": null, "blacklist_algos": null, "supported_models": ["TensorFlowLinearRegressor", "XGBoostRegressor", "AutoArima", "ElasticNet", "SGD", "TensorFlowDNN", "LassoLars", "SeasonalAverage", "TCNForecaster", "SeasonalNaive", "ExtremeRandomTrees", "GradientBoosting", "RandomForest", "Naive", "Arimax", "LightGBM", "Prophet", "KNN", "TabnetRegressor", "Average", "DecisionTree", "ExponentialSmoothing"], "private_models": [], "auto_blacklist": true, "blacklist_samples_reached": false, "exclude_nan_labels": true, "verbosity": 20, "_debug_log": "automl_errors.log", "show_warnings": false, "model_explainability": true, "service_url": null, "sdk_url": null, "sdk_packages": null, "enable_onnx_compatible_models": false, "enable_split_onnx_featurizer_estimator_models": false, "vm_type": "STANDARD_DS3_V2", "telemetry_verbosity": 20, "send_telemetry": true, "enable_dnn": false, "scenario": "SDK-1.13.0", "environment_label": null, "save_mlflow": false, "enable_categorical_indicators": false, "force_text_dnn": false, "enable_feature_sweeping": false, "time_column_name": "date", "grain_column_names": null, "drop_column_names": [], "max_horizon": 1, "dropna": false, "overwrite_columns": true, "transform_dictionary": {"min": "_automl_target_col", "max": "_automl_target_col", "mean": "_automl_target_col"}, "window_size": null, "country_or_region": null, "lags": null, "feature_lags": null, "seasonality": "auto", "use_stl": null, "short_series_handling": true, "freq": null, "short_series_handling_configuration": "auto", "target_aggregation_function": null, "cv_step_size": null, "enable_early_stopping": true, "early_stopping_n_iters": 10, "arguments": null, "dataset_id": null, "hyperdrive_config": null, "validation_dataset_id": null, "run_source": null, "metrics": null, "enable_metric_confidence": false, "enable_ensembling": true, "enable_stack_ensembling": false, "ensemble_iterations": 15, "enable_tf": false, "enable_subsampling": false, "subsample_seed": null, "enable_nimbusml": false, "enable_streaming": false, "force_streaming": false, "track_child_runs": true, "allowed_private_models": [], "label_column_name": "cnt", "weight_column_name": null, "cv_split_column_names": null, "enable_local_managed": false, "_local_managed_run_id": null, "cost_mode": 1, "lag_length": 0, "metric_operation": "minimize", "preprocess": true}', 'DataPrepJsonString': '{\\"training_data\\": {\\"datasetId\\": \\"340931f5-8590-4f38-b18b-58ff4d26037c\\"}, \\"datasets\\": 0}', 'EnableSubsampling': 'False', 'runTemplate': 'AutoML', 'Orchestrator': 'automl', 'ClientType': 'Others', '_aml_system_scenario_identification': 'Remote.Parent', 'root_attribution': 'azureml.StepRun', 'snapshotId': '00000000-0000-0000-0000-000000000000', 'SetupRunId': '4d85652b-e60d-4510-abed-8246d05fa168_setup', 'SetupRunContainerId': 'dcid.4d85652b-e60d-4510-abed-8246d05fa168_setup', 'forecasting_target_lags': '[0]', 'forecasting_target_rolling_window_size': '0', 'forecasting_max_horizon': '1', 'forecasting_freq': 'D', 'ProblemInfoJsonString': '{"dataset_num_categorical": 0, "is_sparse": false, "subsampling": false, "has_extra_col": true, "dataset_classes": 696, "dataset_features": 34, "dataset_samples": 731, "single_frequency_class_detected": false, "series_column_count": 1, "series_count": 1, "series_len_min": 731, "series_len_max": 731, "series_len_avg": 731.0, "series_len_perc_25": 731.0, "series_len_perc_50": 731.0, "series_len_perc_75": 731.0}', 'ModelExplainRunId': '4d85652b-e60d-4510-abed-8246d05fa168_ModelExplain'}, 'inputDatasets': [], 'outputDatasets': [], 'logFiles': {'logs/azureml/executionlogs.txt': 'https://mlstrg187534.blob.core.windows.net/azureml/ExperimentRun/dcid.4d85652b-e60d-4510-abed-8246d05fa168/logs/azureml/executionlogs.txt?sv=2019-07-07&sr=b&sig=El7jbMDyUUAddM4l5Ziy%2BTUVq3PQkuWZ00CE0WZ5mwg%3D&skoid=4b09ab65-7bf6-4a41-9dfd-2e75aa22461a&sktid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&skt=2022-02-25T19%3A47%3A20Z&ske=2022-02-27T03%3A57%3A20Z&sks=b&skv=2019-07-07&st=2022-02-25T19%3A58%3A00Z&se=2022-02-26T04%3A08%3A00Z&sp=r', 'logs/azureml/stderrlogs.txt': 'https://mlstrg187534.blob.core.windows.net/azureml/ExperimentRun/dcid.4d85652b-e60d-4510-abed-8246d05fa168/logs/azureml/stderrlogs.txt?sv=2019-07-07&sr=b&sig=ormD%2BZRFUVANAyJ8G%2FU0NXW0FQJe3iDNxf4FCw%2FMWQk%3D&skoid=4b09ab65-7bf6-4a41-9dfd-2e75aa22461a&sktid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&skt=2022-02-25T19%3A47%3A20Z&ske=2022-02-27T03%3A57%3A20Z&sks=b&skv=2019-07-07&st=2022-02-25T19%3A58%3A00Z&se=2022-02-26T04%3A08%3A00Z&sp=r', 'logs/azureml/stdoutlogs.txt': 'https://mlstrg187534.blob.core.windows.net/azureml/ExperimentRun/dcid.4d85652b-e60d-4510-abed-8246d05fa168/logs/azureml/stdoutlogs.txt?sv=2019-07-07&sr=b&sig=gaI%2BE7%2BIGwynn0p9YxXiWi%2BF7IG8MHOomi0Umkh%2BEd8%3D&skoid=4b09ab65-7bf6-4a41-9dfd-2e75aa22461a&sktid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&skt=2022-02-25T19%3A47%3A20Z&ske=2022-02-27T03%3A57%3A20Z&sks=b&skv=2019-07-07&st=2022-02-25T19%3A58%3A00Z&se=2022-02-26T04%3A08%3A00Z&sp=r'}, 'submittedBy': 'ODL_User 187534'}
    
    
    
    PipelineRun Execution Summary
    ==============================
    PipelineRun Status: Finished
    {'runId': '5dd4f320-7a3a-45a5-b874-3b1d403bd3ae', 'status': 'Completed', 'startTimeUtc': '2022-02-25T19:57:19.235873Z', 'endTimeUtc': '2022-02-25T20:09:39.816045Z', 'services': {}, 'properties': {'azureml.runsource': 'azureml.PipelineRun', 'runSource': 'SDK', 'runType': 'SDK', 'azureml.parameters': '{}', 'azureml.continue_on_step_failure': 'False', 'azureml.pipelineComponent': 'pipelinerun'}, 'inputDatasets': [], 'outputDatasets': [], 'logFiles': {'logs/azureml/executionlogs.txt': 'https://mlstrg187534.blob.core.windows.net/azureml/ExperimentRun/dcid.5dd4f320-7a3a-45a5-b874-3b1d403bd3ae/logs/azureml/executionlogs.txt?sv=2019-07-07&sr=b&sig=WB5B8WYyrrX%2Bx4UsVj9BH4lVcVGPCzOAHYhvM%2BbcY1U%3D&skoid=4b09ab65-7bf6-4a41-9dfd-2e75aa22461a&sktid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&skt=2022-02-25T19%3A47%3A20Z&ske=2022-02-27T03%3A57%3A20Z&sks=b&skv=2019-07-07&st=2022-02-25T19%3A58%3A06Z&se=2022-02-26T04%3A08%3A06Z&sp=r', 'logs/azureml/stderrlogs.txt': 'https://mlstrg187534.blob.core.windows.net/azureml/ExperimentRun/dcid.5dd4f320-7a3a-45a5-b874-3b1d403bd3ae/logs/azureml/stderrlogs.txt?sv=2019-07-07&sr=b&sig=DP9QhRP%2F7m8HdYv7OFZr22vovfhJ8ZLmR8k4UUf%2FiVA%3D&skoid=4b09ab65-7bf6-4a41-9dfd-2e75aa22461a&sktid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&skt=2022-02-25T19%3A47%3A20Z&ske=2022-02-27T03%3A57%3A20Z&sks=b&skv=2019-07-07&st=2022-02-25T19%3A58%3A06Z&se=2022-02-26T04%3A08%3A06Z&sp=r', 'logs/azureml/stdoutlogs.txt': 'https://mlstrg187534.blob.core.windows.net/azureml/ExperimentRun/dcid.5dd4f320-7a3a-45a5-b874-3b1d403bd3ae/logs/azureml/stdoutlogs.txt?sv=2019-07-07&sr=b&sig=dBe8PC2RgfM6LeUtwAz%2FXDu1zQHeh%2FHR1fARVV7dz88%3D&skoid=4b09ab65-7bf6-4a41-9dfd-2e75aa22461a&sktid=660b3398-b80e-49d2-bc5b-ac1dc93b5254&skt=2022-02-25T19%3A47%3A20Z&ske=2022-02-27T03%3A57%3A20Z&sks=b&skv=2019-07-07&st=2022-02-25T19%3A58%3A06Z&se=2022-02-26T04%3A08%3A06Z&sp=r'}, 'submittedBy': 'ODL_User 187534'}
    





    'Finished'



## Examine Results

### Retrieve the metrics of all child runs
Outputs of above run can be used as inputs of other steps in pipeline. In this tutorial, we will examine the outputs by retrieve output data and running some tests.


```
metrics_output = pipeline_run.get_pipeline_output(metrics_output_name)
num_file_downloaded = metrics_output.download('.', show_progress=True)
```

    Downloading azureml/4d85652b-e60d-4510-abed-8246d05fa168/metrics_data
    Downloaded azureml/4d85652b-e60d-4510-abed-8246d05fa168/metrics_data, 1 files out of an estimated total of 1



```
import json
with open(metrics_output._path_on_datastore) as f:
    metrics_output_result = f.read()
    
deserialized_metrics_output = json.loads(metrics_output_result)
df = pd.DataFrame(deserialized_metrics_output)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_13</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_10</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_3</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_6</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_2</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_12</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_11</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_26</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_23</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_17</th>
      <th>...</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_9</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_14</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_16</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_18</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_15</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_22</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_20</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_19</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_24</th>
      <th>4d85652b-e60d-4510-abed-8246d05fa168_21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>root_mean_squared_log_error</th>
      <td>[0.2079557048025764]</td>
      <td>[0.001786973013283344]</td>
      <td>[0.30225518015642167]</td>
      <td>[1.7680386412699534e-05]</td>
      <td>[0.8784824532540254]</td>
      <td>[0.000516298968248563]</td>
      <td>[0.009090548237308779]</td>
      <td>[0.11374623326227991]</td>
      <td>[0.08104292067197161]</td>
      <td>[0.0005642251553285386]</td>
      <td>...</td>
      <td>[0.017214338880493802]</td>
      <td>[0.009804409289858226]</td>
      <td>[0.00024071746534826133]</td>
      <td>[0.06677051829219316]</td>
      <td>[0.0005242578975496551]</td>
      <td>[0.036680512687675325]</td>
      <td>[0.10383890130026459]</td>
      <td>[0.03999630908570998]</td>
      <td>[0.13827272940343605]</td>
      <td>[0.38526657130863295]</td>
    </tr>
    <tr>
      <th>explained_variance</th>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>...</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
      <td>[1.0]</td>
    </tr>
    <tr>
      <th>normalized_median_absolute_error</th>
      <td>[0.045204041573806863]</td>
      <td>[0.000529070260659238]</td>
      <td>[0.06561041351653409]</td>
      <td>[4.276184211675838e-06]</td>
      <td>[0.2939919143283009]</td>
      <td>[0.00012374572788181892]</td>
      <td>[0.0018129825361777978]</td>
      <td>[0.027746732992936406]</td>
      <td>[0.02064001478258373]</td>
      <td>[0.00013999331698355454]</td>
      <td>...</td>
      <td>[0.0033274033730144163]</td>
      <td>[0.001979044319190415]</td>
      <td>[6.33839750012797e-05]</td>
      <td>[0.01238966924827486]</td>
      <td>[9.019590243495753e-05]</td>
      <td>[0.008342920693357866]</td>
      <td>[0.02354961541246857]</td>
      <td>[0.009276481201469665]</td>
      <td>[0.02786251596028677]</td>
      <td>[0.09272827976047719]</td>
    </tr>
    <tr>
      <th>median_absolute_error</th>
      <td>[392.9135293595293]</td>
      <td>[4.598678705650097]</td>
      <td>[570.2857142857143]</td>
      <td>[0.03716859316788638]</td>
      <td>[2555.3777193415913]</td>
      <td>[1.07559786674877]</td>
      <td>[15.758444204457419]</td>
      <td>[241.17460317460322]</td>
      <td>[179.4030084902178]</td>
      <td>[1.216821911221056]</td>
      <td>...</td>
      <td>[28.921790118241308]</td>
      <td>[17.201853222403088]</td>
      <td>[0.5509335107111232]</td>
      <td>[107.69100510600508]</td>
      <td>[0.7839827839646508]</td>
      <td>[72.51666666666658]</td>
      <td>[204.6932571651768]</td>
      <td>[80.63117460317433]</td>
      <td>[242.18098872681261]</td>
      <td>[805.9942076780677]</td>
    </tr>
    <tr>
      <th>normalized_mean_absolute_error</th>
      <td>[0.045204041573806863]</td>
      <td>[0.000529070260659238]</td>
      <td>[0.06561041351653409]</td>
      <td>[4.276184211675838e-06]</td>
      <td>[0.2939919143283009]</td>
      <td>[0.00012374572788181892]</td>
      <td>[0.0018129825361777978]</td>
      <td>[0.027746732992936406]</td>
      <td>[0.02064001478258373]</td>
      <td>[0.00013999331698355454]</td>
      <td>...</td>
      <td>[0.0033274033730144163]</td>
      <td>[0.001979044319190415]</td>
      <td>[6.33839750012797e-05]</td>
      <td>[0.01238966924827486]</td>
      <td>[9.019590243495753e-05]</td>
      <td>[0.008342920693357866]</td>
      <td>[0.02354961541246857]</td>
      <td>[0.009276481201469665]</td>
      <td>[0.02786251596028677]</td>
      <td>[0.09272827976047719]</td>
    </tr>
    <tr>
      <th>r2_score</th>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>...</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
    </tr>
    <tr>
      <th>normalized_root_mean_squared_error</th>
      <td>[0.045204041573806863]</td>
      <td>[0.000529070260659238]</td>
      <td>[0.06561041351653409]</td>
      <td>[4.276184211675838e-06]</td>
      <td>[0.2939919143283009]</td>
      <td>[0.00012374572788181892]</td>
      <td>[0.0018129825361777978]</td>
      <td>[0.027746732992936406]</td>
      <td>[0.02064001478258373]</td>
      <td>[0.00013999331698355454]</td>
      <td>...</td>
      <td>[0.0033274033730144163]</td>
      <td>[0.001979044319190415]</td>
      <td>[6.33839750012797e-05]</td>
      <td>[0.01238966924827486]</td>
      <td>[9.019590243495753e-05]</td>
      <td>[0.008342920693357866]</td>
      <td>[0.02354961541246857]</td>
      <td>[0.009276481201469665]</td>
      <td>[0.02786251596028677]</td>
      <td>[0.09272827976047719]</td>
    </tr>
    <tr>
      <th>spearman_correlation</th>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>...</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
      <td>[-1.0]</td>
    </tr>
    <tr>
      <th>mean_absolute_percentage_error</th>
      <td>[23.216552658752136]</td>
      <td>[0.179136557516656]</td>
      <td>[25.712315091976155]</td>
      <td>[0.0017689130177037324]</td>
      <td>[150.98783460495096]</td>
      <td>[0.051649481615863684]</td>
      <td>[0.9149155256295919]</td>
      <td>[11.227195954072172]</td>
      <td>[7.668370426324709]</td>
      <td>[0.0564328736931512]</td>
      <td>...</td>
      <td>[1.7430326913281748]</td>
      <td>[0.9870780981972899]</td>
      <td>[0.024085343761376565]</td>
      <td>[6.977242993920168]</td>
      <td>[0.052438233229460944]</td>
      <td>[3.6334156450012607]</td>
      <td>[10.457396633643304]</td>
      <td>[4.070565361853174]</td>
      <td>[15.435853774573717]</td>
      <td>[51.082408623301966]</td>
    </tr>
    <tr>
      <th>normalized_root_mean_squared_log_error</th>
      <td>[0.035025258730636395]</td>
      <td>[0.00030097367222665694]</td>
      <td>[0.050907792588350397]</td>
      <td>[2.977846215617534e-06]</td>
      <td>[0.14795975539481918]</td>
      <td>[8.695844609040273e-05]</td>
      <td>[0.0015310895381174742]</td>
      <td>[0.01915788390334803]</td>
      <td>[0.013649778290607774]</td>
      <td>[9.503048770158448e-05]</td>
      <td>...</td>
      <td>[0.002899351444763439]</td>
      <td>[0.0016513226814544443]</td>
      <td>[4.054320852996093e-05]</td>
      <td>[0.011245927015962205]</td>
      <td>[8.829894097249427e-05]</td>
      <td>[0.006177971642941444]</td>
      <td>[0.017489226312880277]</td>
      <td>[0.00673643974002746]</td>
      <td>[0.02328879665669369]</td>
      <td>[0.0648891135406008]</td>
    </tr>
    <tr>
      <th>mean_absolute_error</th>
      <td>[392.9135293595293]</td>
      <td>[4.598678705650097]</td>
      <td>[570.2857142857143]</td>
      <td>[0.03716859316788638]</td>
      <td>[2555.3777193415913]</td>
      <td>[1.07559786674877]</td>
      <td>[15.758444204457419]</td>
      <td>[241.17460317460322]</td>
      <td>[179.4030084902178]</td>
      <td>[1.216821911221056]</td>
      <td>...</td>
      <td>[28.921790118241308]</td>
      <td>[17.201853222403088]</td>
      <td>[0.5509335107111232]</td>
      <td>[107.69100510600508]</td>
      <td>[0.7839827839646508]</td>
      <td>[72.51666666666658]</td>
      <td>[204.6932571651768]</td>
      <td>[80.63117460317433]</td>
      <td>[242.18098872681261]</td>
      <td>[805.9942076780677]</td>
    </tr>
    <tr>
      <th>root_mean_squared_error</th>
      <td>[392.9135293595293]</td>
      <td>[4.598678705650097]</td>
      <td>[570.2857142857143]</td>
      <td>[0.03716859316788638]</td>
      <td>[2555.3777193415913]</td>
      <td>[1.07559786674877]</td>
      <td>[15.758444204457419]</td>
      <td>[241.17460317460322]</td>
      <td>[179.4030084902178]</td>
      <td>[1.216821911221056]</td>
      <td>...</td>
      <td>[28.921790118241308]</td>
      <td>[17.201853222403088]</td>
      <td>[0.5509335107111232]</td>
      <td>[107.69100510600508]</td>
      <td>[0.7839827839646508]</td>
      <td>[72.51666666666658]</td>
      <td>[204.6932571651768]</td>
      <td>[80.63117460317433]</td>
      <td>[242.18098872681261]</td>
      <td>[805.9942076780677]</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 36 columns</p>
</div>



### Retrieve the Best Model


```
# Retrieve best model from Pipeline Run
best_model_output = pipeline_run.get_pipeline_output(best_model_output_name)
num_file_downloaded = best_model_output.download('.', show_progress=True)
```

    Downloading azureml/4d85652b-e60d-4510-abed-8246d05fa168/model_data
    Downloaded azureml/4d85652b-e60d-4510-abed-8246d05fa168/model_data, 1 files out of an estimated total of 1



```
!pip install azureml.automl.runtime
```

    Collecting azureml.automl.runtime
      Downloading azureml_automl_runtime-1.38.1-py3-none-any.whl (2.1 MB)
    [K     |████████████████████████████████| 2.1 MB 7.2 MB/s eta 0:00:01
    [?25hRequirement already satisfied: scipy<=1.5.2,>=1.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.5.2)
    Collecting azureml-automl-core~=1.38.1
      Downloading azureml_automl_core-1.38.1-py3-none-any.whl (228 kB)
    [K     |████████████████████████████████| 228 kB 71.7 MB/s eta 0:00:01
    [?25hRequirement already satisfied: onnxruntime<1.9.0,>=1.3.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.8.0)
    Collecting azureml-training-tabular~=1.38.1
      Downloading azureml_training_tabular-1.38.1-py3-none-any.whl (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 64.7 MB/s eta 0:00:01
    [?25hCollecting statsmodels<0.12,>=0.11.0
      Downloading statsmodels-0.11.1-cp36-cp36m-manylinux1_x86_64.whl (8.7 MB)
    [K     |████████████████████████████████| 8.7 MB 32.5 MB/s eta 0:00:01
    [?25hRequirement already satisfied: joblib==0.14.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (0.14.1)
    Requirement already satisfied: sklearn-pandas<=1.7.0,>=1.4.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.7.0)
    Requirement already satisfied: boto3<=1.20.19 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.20.19)
    Requirement already satisfied: dill<0.4.0,>=0.2.8 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (0.3.4)
    Requirement already satisfied: smart-open<=1.9.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.9.0)
    Requirement already satisfied: azureml-dataset-runtime[fuse,pandas]~=1.38.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.38.0)
    Collecting pmdarima==1.7.1
      Downloading pmdarima-1.7.1-cp36-cp36m-manylinux1_x86_64.whl (1.5 MB)
    [K     |████████████████████████████████| 1.5 MB 68.0 MB/s eta 0:00:01
    [?25hRequirement already satisfied: gensim<3.9.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (3.8.3)
    Requirement already satisfied: psutil<6.0.0,>=5.2.2 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (5.9.0)
    Requirement already satisfied: scikit-learn<0.23.0,>=0.19.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (0.22.2.post1)
    Collecting pandas<=1.3.4,>=1.1.5
      Downloading pandas-1.1.5-cp36-cp36m-manylinux1_x86_64.whl (9.5 MB)
    [K     |████████████████████████████████| 9.5 MB 58.3 MB/s eta 0:00:01
    [?25hRequirement already satisfied: skl2onnx==1.4.9 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.4.9)
    Requirement already satisfied: numpy<1.19.0,>=1.16.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.18.5)
    Requirement already satisfied: keras2onnx<=1.6.0,>=1.4.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.6.0)
    Requirement already satisfied: onnxconverter-common<=1.6.0,>=1.4.2 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.6.0)
    Requirement already satisfied: onnxmltools==1.4.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.4.1)
    Requirement already satisfied: botocore<=1.23.19 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.23.19)
    Requirement already satisfied: nimbusml<=1.8.0,>=1.7.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.8.0)
    Requirement already satisfied: onnx<=1.7.0,>=1.6.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (1.7.0)
    Requirement already satisfied: lightgbm<=3.2.1,>=2.0.11 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (2.3.0)
    Requirement already satisfied: dataclasses<=0.8,>=0.6 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml.automl.runtime) (0.6)
    Requirement already satisfied: azureml-telemetry~=1.38.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.38.0)
    Requirement already satisfied: protobuf in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from onnxruntime<1.9.0,>=1.3.0->azureml.automl.runtime) (3.19.3)
    Requirement already satisfied: flatbuffers in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from onnxruntime<1.9.0,>=1.3.0->azureml.automl.runtime) (2.0)
    Requirement already satisfied: patsy>=0.5 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from statsmodels<0.12,>=0.11.0->azureml.automl.runtime) (0.5.2)
    Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from boto3<=1.20.19->azureml.automl.runtime) (0.5.0)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from boto3<=1.20.19->azureml.automl.runtime) (0.10.0)
    Requirement already satisfied: requests in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from smart-open<=1.9.0->azureml.automl.runtime) (2.27.1)
    Requirement already satisfied: boto>=2.32 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from smart-open<=1.9.0->azureml.automl.runtime) (2.49.0)
    Requirement already satisfied: azureml-dataprep<2.27.0a,>=2.26.0a in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-dataset-runtime[fuse,pandas]~=1.38.0->azureml.automl.runtime) (2.26.0)
    Requirement already satisfied: pyarrow<4.0.0,>=0.17.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-dataset-runtime[fuse,pandas]~=1.38.0->azureml.automl.runtime) (3.0.0)
    Requirement already satisfied: fusepy<4.0.0,>=3.0.1; extra == "fuse" in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-dataset-runtime[fuse,pandas]~=1.38.0->azureml.automl.runtime) (3.0.1)
    Collecting Cython<0.29.18,>=0.29
      Downloading Cython-0.29.17-cp36-cp36m-manylinux1_x86_64.whl (2.1 MB)
    [K     |████████████████████████████████| 2.1 MB 45.7 MB/s eta 0:00:01    |█▋                              | 102 kB 45.7 MB/s eta 0:00:01
    [?25hRequirement already satisfied: urllib3 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from pmdarima==1.7.1->azureml.automl.runtime) (1.26.7)
    Collecting setuptools<50.0.0
      Downloading setuptools-49.6.0-py3-none-any.whl (803 kB)
    [K     |████████████████████████████████| 803 kB 50.8 MB/s eta 0:00:01
    [?25hRequirement already satisfied: six>=1.5.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from gensim<3.9.0->azureml.automl.runtime) (1.16.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from pandas<=1.3.4,>=1.1.5->azureml.automl.runtime) (2.8.2)
    Requirement already satisfied: pytz>=2017.2 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from pandas<=1.3.4,>=1.1.5->azureml.automl.runtime) (2021.3)
    Requirement already satisfied: fire in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from keras2onnx<=1.6.0,>=1.4.0->azureml.automl.runtime) (0.4.0)
    Requirement already satisfied: dotnetcore2>=2.1.2 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from nimbusml<=1.8.0,>=1.7.1->azureml.automl.runtime) (2.1.22)
    Requirement already satisfied: typing-extensions>=3.6.2.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from onnx<=1.7.0,>=1.6.0->azureml.automl.runtime) (4.0.1)
    Requirement already satisfied: applicationinsights in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.11.10)
    Requirement already satisfied: azureml-core~=1.38.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.38.0)
    Requirement already satisfied: idna<4,>=2.5; python_version >= "3" in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from requests->smart-open<=1.9.0->azureml.automl.runtime) (3.3)
    Requirement already satisfied: charset-normalizer~=2.0.0; python_version >= "3" in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from requests->smart-open<=1.9.0->azureml.automl.runtime) (2.0.10)
    Requirement already satisfied: certifi>=2017.4.17 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from requests->smart-open<=1.9.0->azureml.automl.runtime) (2021.10.8)
    Requirement already satisfied: azureml-dataprep-rslex~=2.2.0dev0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-dataprep<2.27.0a,>=2.26.0a->azureml-dataset-runtime[fuse,pandas]~=1.38.0->azureml.automl.runtime) (2.2.0)
    Requirement already satisfied: cloudpickle<3.0.0,>=1.1.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-dataprep<2.27.0a,>=2.26.0a->azureml-dataset-runtime[fuse,pandas]~=1.38.0->azureml.automl.runtime) (2.0.0)
    Requirement already satisfied: azureml-dataprep-native<39.0.0,>=38.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-dataprep<2.27.0a,>=2.26.0a->azureml-dataset-runtime[fuse,pandas]~=1.38.0->azureml.automl.runtime) (38.0.0)
    Requirement already satisfied: azure-identity==1.7.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-dataprep<2.27.0a,>=2.26.0a->azureml-dataset-runtime[fuse,pandas]~=1.38.0->azureml.automl.runtime) (1.7.0)
    Requirement already satisfied: termcolor in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from fire->keras2onnx<=1.6.0,>=1.4.0->azureml.automl.runtime) (1.1.0)
    Requirement already satisfied: distro>=1.2.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from dotnetcore2>=2.1.2->nimbusml<=1.8.0,>=1.7.1->azureml.automl.runtime) (1.6.0)
    Requirement already satisfied: msrestazure<=0.6.4,>=0.4.33 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.6.4)
    Requirement already satisfied: azure-mgmt-containerregistry<9.0.0,>=8.2.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (8.2.0)
    Requirement already satisfied: msal<2.0.0,>=1.15.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.16.0)
    Requirement already satisfied: azure-mgmt-storage<20.0.0,>=16.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (19.0.0)
    Requirement already satisfied: contextlib2<22.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (21.6.0)
    Requirement already satisfied: azure-mgmt-keyvault<10.0.0,>=0.40.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (9.3.0)
    Requirement already satisfied: humanfriendly<11.0,>=4.7 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (10.0)
    Requirement already satisfied: cryptography!=1.9,!=2.0.*,!=2.1.*,!=2.2.*,<37.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (36.0.1)
    Requirement already satisfied: azure-graphrbac<1.0.0,>=0.40.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.61.1)
    Requirement already satisfied: azure-core<1.22 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.21.1)
    Requirement already satisfied: msal-extensions<0.4,>=0.3.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.3.1)
    Requirement already satisfied: azure-mgmt-authorization<1.0.0,>=0.40.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.61.0)
    Requirement already satisfied: jsonpickle<3.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (2.1.0)
    Requirement already satisfied: pkginfo in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.8.2)
    Requirement already satisfied: backports.tempfile in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.0)
    Requirement already satisfied: PyJWT<3.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (2.3.0)
    Requirement already satisfied: paramiko<3.0.0,>=2.0.8 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (2.9.2)
    Requirement already satisfied: ndg-httpsclient<=0.5.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.5.1)
    Requirement already satisfied: pyopenssl<22.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (21.0.0)
    Requirement already satisfied: adal<=1.2.7,>=1.2.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.2.7)
    Requirement already satisfied: azure-mgmt-resource<21.0.0,>=15.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (20.0.0)
    Requirement already satisfied: SecretStorage<4.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (3.3.1)
    Requirement already satisfied: knack~=0.8.2 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.8.2)
    Requirement already satisfied: packaging<22.0,>=20.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (21.3)
    Requirement already satisfied: msrest<1.0.0,>=0.5.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.6.21)
    Requirement already satisfied: pathspec<1.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.9.0)
    Requirement already satisfied: docker<6.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (5.0.3)
    Requirement already satisfied: azure-common<2.0.0,>=1.1.12 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.1.27)
    Requirement already satisfied: argcomplete<2.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.12.3)
    Requirement already satisfied: azure-mgmt-core<2.0.0,>=1.2.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from azure-mgmt-containerregistry<9.0.0,>=8.2.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.3.0)
    Requirement already satisfied: cffi>=1.12 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from cryptography!=1.9,!=2.0.*,!=2.1.*,!=2.2.*,<37.0.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.15.0)
    Requirement already satisfied: portalocker<3,>=1.0; python_version >= "3.5" and platform_system != "Windows" in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from msal-extensions<0.4,>=0.3.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (2.3.2)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from jsonpickle<3.0.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (4.8.3)
    Requirement already satisfied: backports.weakref in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from backports.tempfile->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.0.post1)
    Requirement already satisfied: pynacl>=1.0.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from paramiko<3.0.0,>=2.0.8->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.5.0)
    Requirement already satisfied: bcrypt>=3.1.3 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from paramiko<3.0.0,>=2.0.8->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (3.2.0)
    Requirement already satisfied: pyasn1>=0.1.1 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from ndg-httpsclient<=0.5.1->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.4.8)
    Requirement already satisfied: jeepney>=0.6 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from SecretStorage<4.0.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.7.1)
    Requirement already satisfied: tabulate in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from knack~=0.8.2->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.8.9)
    Requirement already satisfied: pygments in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from knack~=0.8.2->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (2.11.2)
    Requirement already satisfied: colorama in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from knack~=0.8.2->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.4.4)
    Requirement already satisfied: pyyaml in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from knack~=0.8.2->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (6.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from packaging<22.0,>=20.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (3.0.6)
    Requirement already satisfied: requests-oauthlib>=0.5.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from msrest<1.0.0,>=0.5.1->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.3.0)
    Requirement already satisfied: isodate>=0.6.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from msrest<1.0.0,>=0.5.1->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (0.6.1)
    Requirement already satisfied: websocket-client>=0.32.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from docker<6.0.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (1.2.3)
    Requirement already satisfied: pycparser in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from cffi>=1.12->cryptography!=1.9,!=2.0.*,!=2.1.*,!=2.2.*,<37.0.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (2.21)
    Requirement already satisfied: zipp>=0.5 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from importlib-metadata; python_version < "3.8"->jsonpickle<3.0.0->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (3.6.0)
    Requirement already satisfied: oauthlib>=3.0.0 in /anaconda/envs/azureml_py36/lib/python3.6/site-packages (from requests-oauthlib>=0.5.0->msrest<1.0.0,>=0.5.1->azureml-core~=1.38.0->azureml-telemetry~=1.38.0->azureml-automl-core~=1.38.1->azureml.automl.runtime) (3.1.1)
    [31mERROR: pyldavis 3.3.1 requires sklearn, which is not installed.[0m
    [31mERROR: pandas-ml 0.6.1 requires enum34, which is not installed.[0m
    [31mERROR: raiwidgets 0.16.0 has requirement ipython==7.16.1, but you'll have ipython 7.16.3 which is incompatible.[0m
    [31mERROR: raiwidgets 0.16.0 has requirement jinja2==2.11.3, but you'll have jinja2 2.11.2 which is incompatible.[0m
    [31mERROR: pyldavis 3.3.1 has requirement numpy>=1.20.0, but you'll have numpy 1.18.5 which is incompatible.[0m
    [31mERROR: pyldavis 3.3.1 has requirement pandas>=1.2.0, but you'll have pandas 1.1.5 which is incompatible.[0m
    [31mERROR: pycaret 2.3.6 has requirement lightgbm>=2.3.1, but you'll have lightgbm 2.3.0 which is incompatible.[0m
    [31mERROR: pycaret 2.3.6 has requirement pyyaml<6.0.0, but you'll have pyyaml 6.0 which is incompatible.[0m
    [31mERROR: pycaret 2.3.6 has requirement scikit-learn==0.23.2, but you'll have scikit-learn 0.22.2.post1 which is incompatible.[0m
    [31mERROR: pandas-profiling 3.1.0 has requirement joblib~=1.0.1, but you'll have joblib 0.14.1 which is incompatible.[0m
    [31mERROR: ipython 7.16.3 has requirement jedi<=0.17.2,>=0.10, but you'll have jedi 0.18.0 which is incompatible.[0m
    [31mERROR: datasets 1.8.0 has requirement tqdm<4.50.0,>=4.27, but you'll have tqdm 4.62.3 which is incompatible.[0m
    [31mERROR: azureml-train-automl-runtime 1.38.0 has requirement pandas<1.0.0,>=0.21.0, but you'll have pandas 1.1.5 which is incompatible.[0m
    [31mERROR: azureml-train-automl-runtime 1.38.0 has requirement statsmodels<=0.10.2,>=0.9.0, but you'll have statsmodels 0.11.1 which is incompatible.[0m
    [31mERROR: autokeras 1.0.16 has requirement tensorflow<=2.5.0,>=2.3.0, but you'll have tensorflow 2.1.0 which is incompatible.[0m
    Installing collected packages: azureml-automl-core, pandas, statsmodels, Cython, setuptools, pmdarima, azureml-training-tabular, azureml.automl.runtime
      Attempting uninstall: azureml-automl-core
        Found existing installation: azureml-automl-core 1.38.0
        Uninstalling azureml-automl-core-1.38.0:
          Successfully uninstalled azureml-automl-core-1.38.0
      Attempting uninstall: pandas
        Found existing installation: pandas 0.25.3
        Uninstalling pandas-0.25.3:
          Successfully uninstalled pandas-0.25.3
      Attempting uninstall: statsmodels
        Found existing installation: statsmodels 0.10.2
        Uninstalling statsmodels-0.10.2:
          Successfully uninstalled statsmodels-0.10.2
      Attempting uninstall: Cython
        Found existing installation: Cython 0.29.26
        Uninstalling Cython-0.29.26:
          Successfully uninstalled Cython-0.29.26
      Attempting uninstall: setuptools
        Found existing installation: setuptools 50.3.0
        Uninstalling setuptools-50.3.0:
          Successfully uninstalled setuptools-50.3.0
      Attempting uninstall: pmdarima
        Found existing installation: pmdarima 1.1.1
        Uninstalling pmdarima-1.1.1:
          Successfully uninstalled pmdarima-1.1.1
    Successfully installed Cython-0.29.17 azureml-automl-core-1.38.1 azureml-training-tabular-1.38.1 azureml.automl.runtime pandas-1.1.5 pmdarima-1.7.1 setuptools-49.6.0 statsmodels-0.11.1



```
import pickle

with open(best_model_output._path_on_datastore, "rb" ) as f:
    best_model = pickle.load(f)
best_model
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    Input In [19], in <module>
          1 import pickle
          3 with open(best_model_output._path_on_datastore, "rb" ) as f:
    ----> 4     best_model = pickle.load(f)
          5 best_model


    ModuleNotFoundError: No module named 'azureml.automl.runtime'



```
best_model.steps
```

## Publish and run from REST endpoint

Run the following code to publish the pipeline to your workspace. In your workspace in the portal, you can see metadata for the pipeline including run history and durations. You can also run the pipeline manually from the portal.

Additionally, publishing the pipeline enables a REST endpoint to rerun the pipeline from any HTTP library on any platform.



```
published_pipeline = pipeline_run.publish_pipeline(
    name="Bikesharing Train", description="Training bikesharing pipeline", version="1.0")

published_pipeline

```




<table style="width:100%"><tr><th>Name</th><th>Id</th><th>Status</th><th>Endpoint</th></tr><tr><td>Bikesharing Train</td><td><a href="https://ml.azure.com/pipelines/f14de75d-d5d9-4dfc-aec1-dbf500ee306e?wsid=/subscriptions/5a4ab2ba-6c51-4805-8155-58759ad589d8/resourcegroups/aml-quickstarts-187534/workspaces/quick-starts-ws-187534" target="_blank" rel="noopener">f14de75d-d5d9-4dfc-aec1-dbf500ee306e</a></td><td>Active</td><td><a href="https://southcentralus.api.azureml.ms/pipelines/v1.0/subscriptions/5a4ab2ba-6c51-4805-8155-58759ad589d8/resourceGroups/aml-quickstarts-187534/providers/Microsoft.MachineLearningServices/workspaces/quick-starts-ws-187534/PipelineRuns/PipelineSubmit/f14de75d-d5d9-4dfc-aec1-dbf500ee306e" target="_blank" rel="noopener">REST Endpoint</a></td></tr></table>



Authenticate once again, to retrieve the `auth_header` so that the endpoint can be used


```
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()


```

Get the REST url from the endpoint property of the published pipeline object. You can also find the REST url in your workspace in the portal. Build an HTTP POST request to the endpoint, specifying your authentication header. Additionally, add a JSON payload object with the experiment name and the batch size parameter. As a reminder, the process_count_per_node is passed through to ParallelRunStep because you defined it is defined as a PipelineParameter object in the step configuration.

Make the request to trigger the run. Access the Id key from the response dict to get the value of the run id.



```
import requests

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "pipeline-bike-rest-endpoint"}
                        )
```


```
try:
    response.raise_for_status()
except Exception:    
    raise Exception("Received bad response from the endpoint: {}\n"
                    "Response Code: {}\n"
                    "Headers: {}\n"
                    "Content: {}".format(rest_endpoint, response.status_code, response.headers, response.content))

run_id = response.json().get('Id')
print('Submitted pipeline run: ', run_id)
```

    Submitted pipeline run:  2e370cde-d73c-469d-8d8f-6560fec67137


Use the run id to monitor the status of the new run. This will take another 10-15 min to run and will look similar to the previous pipeline run, so if you don't need to see another pipeline run, you can skip watching the full output.


```
from azureml.pipeline.core.run import PipelineRun
from azureml.widgets import RunDetails

published_pipeline_run = PipelineRun(ws.experiments["pipeline-bike-rest-endpoint"], run_id)
RunDetails(published_pipeline_run).show()
```


    _PipelineWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO', …





    _UserRunWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO', '…





```

```
