from utils import *
from matplotlib import pyplot as plt

classes = set([1, 4, 7, 8])
images = []
labels = []
extract_all_data('data/data_batch_1', images, labels, classes)
extract_all_data('data/data_batch_2', images, labels, classes)
extract_all_data('data/data_batch_3', images, labels, classes)
extract_all_data('data/data_batch_4', images, labels, classes)
extract_all_data('data/data_batch_5', images, labels, classes)
extract_all_data('data/test_batch', images, labels, classes)

images = np.array(images)
images = np.reshape(images, (images.shape[0], 3, 32, 32))

import os
from PIL import Image

# Ensure the directory exists
if not os.path.exists('all_imgs'):
    os.makedirs('all_imgs')

# Save all images
for i in range(images.shape[0]):
    img = Image.fromarray(images[i].transpose(1, 2, 0))
    img.save(f'all_imgs/img_{i}.png')