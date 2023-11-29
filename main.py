import numpy as np
import matplotlib.pyplot as plt
from diffusion import Diffusion
from ema import ExponentialMovingAverage
from utils import *
from torch import optim, nn
import torch

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def extract_horses(filename, horses, labels):
    data = unpickle(filename)
    for i in range(len(data[b'labels'])):
        if data[b'labels'][i] == 7:
            horses.append(data[b'data'][i])
            labels.append(data[b'labels'][i])

def extract_all_data(filename, images, labels):
    data = unpickle(filename)
    for i in range(len(data[b'labels'])):
        images.append(data[b'data'][i])
        labels.append(data[b'labels'][i])

def train(diffusion, lr, num_epochs, train_images, train_labels, batch_size):
    criterion = nn.MSELoss()
    longLoss = None

    # load from state
    diffusion.model.to(device)
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=lr)
    # diffusion.model.load_state_dict(torch.load('model_0.pt', map_location=device))
    # optimizer.load_state_dict(torch.load('optimizer_0.pt', map_location=device))
    ema = ExponentialMovingAverage(0.99, diffusion.model)

    diffusion.model.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        # start by training on one at a time
        num_batches = len(train_images) // batch_size

        for i in range(num_batches):
            image = train_images[i * batch_size : (i + 1) * batch_size]
            #labels = train_labels[i * batch_size : (i + 1) * batch_size]
            t = torch.randint(1, diffusion.noising_steps, (batch_size,))

            image_t, noise = diffusion.noise_image(image, t)

            image_t = image_t.to(device)
            noise = noise.to(device)
            #labels = labels.to(device)

            #pred_noise = diffusion.model(image_t, t, labels)
            pred_noise = diffusion.model(image_t, t)
            loss = criterion(noise, pred_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            if longLoss is None:
                longLoss = loss.item() 
            else:
                longLoss = 0.995 * longLoss + 0.005 * loss.item()
            
            if i % 50 == 0:
                print(f"Step {i} of {num_batches} Long: {round(longLoss, 6)}, Current: {round(loss.item(), 6)}")
        
        if epoch > 0 and epoch % 40 == 0:
            torch.save(diffusion.model.state_dict(), f"model_{epoch}.pt")
            torch.save(optimizer.state_dict(), f"optimizer_{epoch}.pt")

        if epoch > 0 and epoch % 20 == 0:
            show_4_images(diffusion.generate(4), save=True, name=f"Epoch_{epoch}")
            ema_model = ema.getEMAModel()
            tmpDiff = Diffusion(device=device, num_classes=10)
            tmpDiff.model = ema_model
            show_4_images(tmpDiff.generate(4), save=True, name=f"EMA_Epoch_{epoch}")


if __name__ == '__main__':
    images = []
    labels = []
    extract_horses('data/data_batch_1', images, labels)
    extract_horses('data/data_batch_2', images, labels)
    extract_horses('data/data_batch_3', images, labels)
    extract_horses('data/data_batch_4', images, labels)
    extract_horses('data/data_batch_5', images, labels)
    extract_horses('data/test_batch', images, labels)

    images = np.array([normalize(squeeze01(x)) for x in images])
    reshaped = np.reshape(images, (images.shape[0], 3, 32, 32))
    images = torch.tensor(reshaped, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)

    diffusion = Diffusion(device=device, num_classes=10)
    train(diffusion, 3e-4, 500, images, labels, batch_size=12)