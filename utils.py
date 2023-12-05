import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_horses(filename, horses, labels):
    data = unpickle(filename)
    for i in range(len(data[b'labels'])):
        if data[b'labels'][i] == 7:
            horses.append(data[b'data'][i])
            labels.append(data[b'labels'][i])

def extract_all_data(filename, images, labels, classes):
    data = unpickle(filename)
    for i in range(len(data[b'labels'])):
        if data[b'labels'][i] in classes:
            images.append(data[b'data'][i])
            labels.append(data[b'labels'][i])

def read_binary_data(filename, labelfile, images, labels, classes):
    with open(filename, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        img = np.reshape(everything, (-1, 3, 96, 96))
        img = [img[i] for i in range(len(img))]
    
    with open(labelfile, 'rb') as f:
        labs = np.fromfile(f, dtype=np.uint8)

    for i, pic in enumerate(img):
        if labs[i] in classes:
            images.append(pic)

    for lab in labs:
        if lab in classes:
            labels.append(lab)

def format_image_data(data_row):
    data_row = np.array(data_row)
    data_row = np.clip(data_row, -1, 1)
    data_row = unnormalize(data_row)
    # for small images
    #image = data_row.transpose(1, 2, 0)
    # for 96x96 dataset
    image = data_row.transpose(2, 1, 0)
    return image

def show_image(data_row, save=False, name=None):
    # Reshape the data to 3x32x32 and transpose it to 32x32x3
    image = format_image_data(data_row)

    if save:
        plt.imsave(f'images/image_{name}.png', image)
        plt.close()
    else:
        plt.imshow(image)
        plt.show()

def show_4_images(data, save=False, name=None):
    # data - 4 rows
    _, axes = plt.subplots(2, 2)  # 2x2 grid
    for i, ax in enumerate(axes.flat):
        image = format_image_data(data[i])
        ax.imshow(image)
        ax.axis('off')  # Hide axes for better visualization
    
    if save:
        plt.savefig(f'label_img/{name}.png')
        plt.close()
    else:
        plt.show()


def squeeze01(image):
    return image / 255

def normalize(image):
    return (image * 2) - 1

def unnormalize(image):
    return (image + 1) / 2