import argparse
import os
import numpy as np
from azureml.core import Run
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import to_categorical
import tensorflow as tf

print(f'TensorFlow Version: {tf.__version__}')

# PARSE ARGUMEMNTS
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type=str, default="",
                    dest='input_data_dir', help='data folder')
args = parser.parse_args()

# RUN CONTEXT
# We need to get a reference to the current run context
run = Run.get_context()

# Blob storage associated with the workspace
saved_data = np.load(os.path.join(args.input_data_dir,'mnist.npy'))
training_images = saved_data[0]
training_labels = saved_data[1]
test_images = saved_data[2]
test_labels = saved_data[3]

# MODEL CREATION & EVALUATION
# We next need to create, compile, fit, and evaluate our model

model = keras.Sequential([
    Flatten(input_shape=training_images[0].shape),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

input_shape = training_images.shape

model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adadelta(),
          metrics=['accuracy'])

model.fit(training_images, training_labels,
      batch_size=128,
      epochs=10,
      verbose=1,
      validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)

# METRICS
# We need to associate our accuracy metric with the Run context
print('Test accuracy:', score[1])

# MODEL EXPORT
# We will save out our model export so that we can register this within our workspace
os.makedirs('outputs', exist_ok=True)
model.save('outputs/mnist.h5')
