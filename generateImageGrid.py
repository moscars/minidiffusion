from utils import *
from PIL import Image
import numpy as np
import os

def read_png_images_from_directory(directory_path):
    images = []
    i = 0
    for file_name in os.listdir(directory_path):
        i += 1
        if file_name.endswith(".png"):
            image_path = os.path.join(directory_path, file_name)
            image = Image.open(image_path)
            images.append((image_path, image))
        if i > 50:
            break
    return images

# trueCars = read_png_images_from_directory("cifar10-64/test/class1")
# genCars = read_png_images_from_directory("cars64")
# trueDeer = read_png_images_from_directory("cifar10-64/test/class4")
# genDeer = read_png_images_from_directory("deer64")
# trueHorse = read_png_images_from_directory("cifar10-64/test/class7")
# genHorse = read_png_images_from_directory("horse64")
# trueShip = read_png_images_from_directory("cifar10-64/test/class8")
# genShip = read_png_images_from_directory("ship64")
latents = read_png_images_from_directory("latents")
latents.sort()
latents = [x[1] for x in latents]

def create_image_grid(images, rows, cols):
    # images = images.transpose(0, 2, 3, 1)
    # images = [Image.fromarray((unnormalize(img) * 255).astype('uint8')) for img in images]
    w, h = images[0].size
    
    grid_img = Image.new('RGB', size=(cols * w, rows * h), color=(255, 255, 255))
    
    for i, img in enumerate(images):
        grid_img.paste(img, box=(i % cols * w, i // cols * h))
    
    return grid_img

grid_cols = 10
grid_rows = 4
num_images = grid_cols * grid_rows

# myTot = []
# for i in range(10):
#     myTot.append(trueCars[i])
# # for i in range(10):
# #     myTot.append(genCars[i])
# for i in range(10):
#     myTot.append(trueDeer[i])
# # for i in range(10):
# #     myTot.append(genDeer[i])
# for i in range(10):
#     myTot.append(trueHorse[i])
# # for i in range(10):
# #     myTot.append(genHorse[i])
# for i in range(10):
#     myTot.append(trueShip[i])
# # for i in range(10):
#     myTot.append(genShip[i])

myTot = []
for i in range(40):
    myTot.append(latents[i])

grid_image = create_image_grid(myTot, grid_rows, grid_cols)

output_path = "grid_image.png"
grid_image.save(output_path)

grid_image.show()