import numpy as np
import matplotlib.pyplot as plt
from diffusion import Diffusion
from utils import *
from torch import optim, nn
import torch

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

    longLoss = None
    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        # start by training on one at a time
        for image in train_images:
            t = np.random.randint(1, diffusion.noising_steps)
            image_t, noise = diffusion.noise_image(image, t)

            image_t = torch.unsqueeze(image_t, 0)
            noise = torch.unsqueeze(noise, 0)

            pred_noise = diffusion.model(image_t, t)
            loss = criterion(noise, pred_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if longLoss is None:
                longLoss = loss
            else:
                longLoss = 0.99 * longLoss + 0.01 * loss
            
            print(f"Loss: {longLoss}")

if __name__ == '__main__':
    horses = []
    extract_horses('data/data_batch_1', horses)
    extract_horses('data/data_batch_2', horses)
    extract_horses('data/data_batch_3', horses)
    extract_horses('data/data_batch_4', horses)
    extract_horses('data/data_batch_5', horses)
    extract_horses('data/test_batch', horses)

    horses = np.array([normalize(squeeze01(x)) for x in horses])
    reshaped = np.reshape(horses, (horses.shape[0], 3, 32, 32))
    horses = torch.tensor(reshaped, dtype=torch.float32)

    diffusion = Diffusion()
    train(diffusion, 0.00001, 10, horses)