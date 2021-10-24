import argparse
import os
import shutil
import numpy as np
import azureml.core
from azureml.core import Run
from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

# PARSE ARGUMEMNTS
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, default="",
                    dest='input_data_dir', help='data folder')
args = parser.parse_args()

# RUN CONTEXT
# We need to get a reference to the current run context
run = Run.get_context()

# Create outputs folder
output_folder = os.path.join(os.getcwd(), 'outputs')
os.makedirs(output_folder, exist_ok=True)

# Copy model
model_path = os.path.join(args.input_data_dir,'mnist.h5')

# Upload the model file to the run so it can be registered
run.upload_file('outputs/mnist.h5', model_path)

# REGISTER MODEL
model = run.register_model(model_name='keras-mnist', 
                           model_path='outputs/mnist.h5',
                           model_framework=Model.Framework.TENSORFLOW,
                           model_framework_version='2.0',
                           resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5))
