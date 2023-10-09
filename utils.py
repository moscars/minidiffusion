import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def show_image(data_row):
    # Reshape the data to 3x32x32 and transpose it to 32x32x3
    image = data_row.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()

def show_images_grid(data):
    fig, axes = plt.subplots(2, 2)  # 2x2 grid
    for i, ax in enumerate(axes.flat):
        image = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        ax.imshow(image)
        ax.axis('off')  # Hide axes for better visualization
    plt.show()