# Copyright (c) David Tucker. All rights reserved.
# Licensed under the MIT License.

import random
import math
import numpy as np
import matplotlib.pyplot as plt

# This function gets us a specific number of images for each of the labels (classes)
def find_sample_data(images, image_labels, max_num_per_class, randomize=False, sort=False, classes=None):
    if classes is None:
        classes = np.unique(image_labels)
    output = []
    for i in classes:
        index_positions = np.where(image_labels == i)
        indexes = index_positions[0][:max_num_per_class]
        for index in indexes:
            output.append((images[index], image_labels[index]))
    if randomize:
        random.shuffle(output)
    if sort:
        output.sort(key=lambda tup: tup[1])
    return output 

def plot_images(image_data):
    cols = min(8, len(image_data))
    rows = math.ceil(len(image_data)/cols)
    fig, ax = plt.subplots(rows, cols, figsize=(18,(2.5 * rows)))
    for i in range(cols*rows):
        col = i % cols
        row = i // cols
        if rows > 1:
            axis = ax[row, col]
        elif cols == 1:
            axis = ax
        else:
            axis = ax[col]
        if len(image_data) < (i + 1):
            axis.axis('off')
        else:
            data = image_data[i]
            axis.imshow(data[0], cmap='gray_r')
            axis.axis('on')
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            if len(data) == 2:
                axis.set_title(f'Number: {data[1]}')
    plt.subplots_adjust(hspace=0.5)
    plt.show()
