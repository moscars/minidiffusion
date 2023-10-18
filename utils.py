import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def show_image(data_row, save=False, name=None):
    # Reshape the data to 3x32x32 and transpose it to 32x32x3
    data_row = np.array(data_row)
    data_row = np.clip(data_row, -1, 1)
    data_row = unnormalize(data_row)
    image = data_row.transpose(1, 2, 0)

    if save:
        plt.imsave(f'images/image_{name}.png', image)
        plt.close()
    else:
        plt.imshow(image)
        plt.show()

def show_images_grid(data):
    _, axes = plt.subplots(2, 2)  # 2x2 grid
    for i, ax in enumerate(axes.flat):
        image = data[i].transpose(1, 2, 0)
        ax.imshow(image)
        ax.axis('off')  # Hide axes for better visualization
    plt.show()

def squeeze01(image):
    return image / 255

def normalize(image):
    return (image * 2) - 1

def unnormalize(image):
    return (image + 1) / 2