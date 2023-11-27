import numpy as np
import matplotlib.pyplot as plt
from diffusion import Diffusion
from ema import ExponentialMovingAverage
from utils import *
from torch import optim, nn
import torch

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def extract_horses(filename, horses):
    data = unpickle(filename)
    for i in range(len(data[b'labels'])):
        if data[b'labels'][i] == 7:
            horses.append(data[b'data'][i])

def train(diffusion, lr, num_epochs, train_images, batch_size):
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
            t = torch.randint(1, diffusion.noising_steps, (batch_size,))

            image_t, noise = diffusion.noise_image(image, t)

            image_t = image_t.to(device)
            noise = noise.to(device)

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

        if epoch > 0 and epoch % 40 == 0:
            show_image(diffusion.generate(1), save=True, name=f"Epoch {epoch}_1")
            show_image(diffusion.generate(1), save=True, name=f"Epoch {epoch}_2")
            ema_model = ema.getEMAModel()
            show_image(ema_model.generate(1), save=True, name=f"EMA Epoch {epoch}_1")
            show_image(ema_model.generate(1), save=True, name=f"EMA Epoch {epoch}_2")


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

    diffusion = Diffusion(device=device)
    train(diffusion, 3e-4, 500, horses, batch_size=12)