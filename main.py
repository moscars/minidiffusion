import numpy as np
import matplotlib.pyplot as plt
from diffusion import Diffusion
from ema import ExponentialMovingAverage
from dataset import TorchDataset
from utils import *
from torch import optim, nn
from torch.utils.data import DataLoader
import torch
import time

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

def train(diffusion, lr, num_epochs, dataset, batch_size):
    criterion = nn.MSELoss()
    longLoss = None

    # load from state
    diffusion.model.to(device)
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=lr)
    # diffusion.model.load_state_dict(torch.load('model_160.pt', map_location=device))
    # optimizer.load_state_dict(torch.load('optimizer_160.pt', map_location=device))
    ema = ExponentialMovingAverage(0.995, diffusion.model)

    show_4_images(diffusion.generate(4, 1), save=True, name=f"1_Label_Epoch_0") # airplaine
    exit()

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    diffusion.model.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        # start by training on one at a time
        num_batches = len(dataset) // batch_size

        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            # move this to the GPU

            current_batch_size = image.shape[0]
            
            t = torch.randint(1, diffusion.noising_steps, (current_batch_size,))
            t = t.to(device)
            
            image_t, noise = diffusion.noise_image(image, t)

            if np.random.random() < 0.1:
                label = None

            pred_noise = diffusion.model(image_t, t, label)
            #pred_noise = diffusion.model(image_t, t)
            loss = criterion(noise, pred_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            if longLoss is None:
                longLoss = loss.item() 
            else:
                longLoss = 0.998 * longLoss + 0.002 * loss.item()
            
            if i % 30 == 0:
                print(f"Step {i} of {num_batches} Long: {round(longLoss, 6)}, Current: {round(loss.item(), 6)}")
        
        if epoch > 0 and epoch % 40 == 0:
            torch.save(diffusion.model.state_dict(), f"label_model_{epoch}.pt")
            torch.save(optimizer.state_dict(), f"label_optimizer_{epoch}.pt")

        if epoch > 0 and epoch % 10 == 0:
            show_4_images(diffusion.generate(4, 1), save=True, name=f"1_Label_Epoch_{epoch}") # airplaine
            show_4_images(diffusion.generate(4, 4), save=True, name=f"4_Label_Epoch_{epoch}") # deer
            show_4_images(diffusion.generate(4, 7), save=True, name=f"7_Label_Epoch_{epoch}") # horse
            show_4_images(diffusion.generate(4, 8), save=True, name=f"8_Label_Epoch_{epoch}") # ship
            ema_model = ema.getEMAModel()
            tmpDiff = Diffusion(device=device, num_classes=10)
            tmpDiff.model = ema_model
            show_4_images(tmpDiff.generate(4, 1), save=True, name=f"1_Label_EMA_Epoch_{epoch}") # airplaine
            show_4_images(tmpDiff.generate(4, 4), save=True, name=f"4_Label_EMA_Epoch_{epoch}") # deer
            show_4_images(tmpDiff.generate(4, 7), save=True, name=f"7_Label_EMA_Epoch_{epoch}") # horse
            show_4_images(tmpDiff.generate(4, 8), save=True, name=f"8_Label_EMA_Epoch_{epoch}") # ship


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    start = time.time()

    classes = set([1, 4, 7, 8])

    images = []
    labels = []
    extract_all_data('data/data_batch_1', images, labels, classes)
    extract_all_data('data/data_batch_2', images, labels, classes)
    extract_all_data('data/data_batch_3', images, labels, classes)
    extract_all_data('data/data_batch_4', images, labels, classes)
    extract_all_data('data/data_batch_5', images, labels, classes)
    extract_all_data('data/test_batch', images, labels, classes)

    images = np.array([normalize(squeeze01(x)) for x in images])
    reshaped = np.reshape(images, (images.shape[0], 3, 32, 32))
    images = torch.tensor(reshaped, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)

    dataset = TorchDataset(images, labels)

    print(f"Length of dataset: {len(dataset)}\n")

    diffusion = Diffusion(device=device, num_classes=10)
    train(diffusion, 6e-4, 500, dataset, batch_size=32)

    end = time.time()
    print(f"Total time: {end - start}")