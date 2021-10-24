import keras
import argparse
import os
import numpy as np
from utils import load_data
import azureml.core
from azureml.core import Workspace, Datastore
from azureml.opendatasets import MNIST
from keras.utils import to_categorical

print(f'Azure ML SDK Version: {azureml.core.VERSION}')

# Create data folder
data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

# Get Data set
mnist_file_dataset = MNIST.get_file_dataset()
mnist_file_dataset.download(data_folder, overwrite=True)

num_classes = 10

# Prepare image data
training_images = load_data(os.path.join(data_folder, "train-images-idx3-ubyte.gz"), False) / 255.0
training_images = np.reshape(training_images, (-1, 28,28)).astype('float32')
test_images = load_data(os.path.join(data_folder, "t10k-images-idx3-ubyte.gz"), False) / 255.0
test_images = np.reshape(test_images, (-1, 28,28)).astype('float32')

# Prepare label data
training_labels = load_data(os.path.join(data_folder, "train-labels-idx1-ubyte.gz"), True).reshape(-1)
training_labels = to_categorical(training_labels, num_classes)
test_labels = load_data(os.path.join(data_folder, "t10k-labels-idx1-ubyte.gz"), True).reshape(-1)
test_labels = to_categorical(test_labels, num_classes)

# Output shapes
print(f'Training Image: {training_images.shape}')
print(f'Training Labels: {training_labels.shape}')
print(f'Test Images: {test_images.shape}')
print(f'Test Labels: {test_labels.shape}')

# Save data as an numpy array
output = np.array([ training_images, training_labels, test_images, test_labels ])
print(f'Output Array: {output.shape}')
output_file_name = os.path.join(data_folder, 'mnist')
np.save(output_file_name, output)
