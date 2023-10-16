import numpy as np
import matplotlib.pyplot as plt
from diffusion import Diffusion
from utils import *
from torch import optim, nn

def extract_horses(filename, horses):
    data = unpickle(filename)
    for i in range(len(data[b'labels'])):
        if data[b'labels'][i] == 7:
            horses.append(data[b'data'][i])

def squeeze01(image):
    return image / 255

def normalize(image):
    return (image * 2) - 1

def unnormalize(image):
    return (image + 1) / 2

def train(diffusion, lr, num_epochs, train_images):
    optimizer = optim.Adam(diffusion.model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        # start by training on one at a time
        for image in train_images:
            t = np.random.randint(1, diffusion.noising_steps)
            image_t, noise = diffusion.noise_image(image, t)
            pred_noise = diffusion.model(image_t, t)
            loss = criterion(noise, pred_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())

if __name__ == '__main__':
    horses = []
    extract_horses('data/data_batch_1', horses)
    extract_horses('data/data_batch_2', horses)
    extract_horses('data/data_batch_3', horses)
    extract_horses('data/data_batch_4', horses)
    extract_horses('data/data_batch_5', horses)
    extract_horses('data/test_batch', horses)

    horses = np.array([normalize(squeeze01(x)) for x in horses])

    diffusion = Diffusion()
    train(diffusion, 0.0001, 10, horses)
    print(horses.shape)

    # d = Diffusion()
    # noised2 = unnormalize(d.noise_image(normalize(horses[0]), 300))
    # noised3 = unnormalize(d.noise_image(normalize(horses[0]), 500))
    # noised4 = unnormalize(d.noise_image(normalize(horses[0]), 700))

    # show_images_grid([horses[0], noised2, noised3, noised4])



