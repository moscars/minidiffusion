from utils import *
from PIL import Image
import numpy as np

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
labels = np.array(labels)
images = np.reshape(images, (images.shape[0], 3, 32, 32))

images = np.array([normalize(squeeze01(x)) for x in images])
labels = np.array(labels)

def create_image_grid(images, rows, cols):
    images = images.transpose(0, 2, 3, 1)
    images = [Image.fromarray((unnormalize(img) * 255).astype('uint8')) for img in images]
    w, h = images[0].size
    
    grid_img = Image.new('RGB', size=(cols * w, rows * h), color=(255, 255, 255))
    
    for i, img in enumerate(images):
        grid_img.paste(img, box=(i % cols * w, i // cols * h))
    
    return grid_img

grid_cols = 5
grid_rows = 2
num_images = grid_cols * grid_rows

horses = []
for i in range(len(labels)):
    if labels[i] == 2:
        horses.append(images[i])

ships = []
for i in range(len(labels)):
    if labels[i] == 3:
        ships.append(images[i])

relevant = []
for i in range(5):
    relevant.append(horses[i])
for i in range(5):
    relevant.append(ships[i])

grid_image = create_image_grid(np.array(relevant)[:num_images], grid_rows, grid_cols)

output_path = "grid_image.png"
grid_image.save(output_path)

grid_image.show()