import numpy as np
import matplotlib.pyplot as plt
from src.diffusion import Diffusion
from src.ema import ExponentialMovingAverage
from src.dataset import TorchDataset
from utils import *
from torch import optim, nn
from torch.utils.data import DataLoader
import torch
import time

def train(diffusion, lr, num_epochs, dataset, batch_size):
    criterion = nn.MSELoss()
    longLoss = None

    print(f"Number of parameters: {diffusion.get_num_params()}")
    # load from state
    diffusion.model.to(device)
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=lr)

    # diffusion.model.load_state_dict(torch.load('model_160.pt', map_location=device))
    # optimizer.load_state_dict(torch.load('optimizer_160.pt', map_location=device))
    ema = ExponentialMovingAverage(0.995, diffusion.model)

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
            
            if i % 50 == 0:
                print(f"Step {i} of {num_batches} Long: {round(longLoss, 6)}, Current: {round(loss.item(), 6)}")
        
        if epoch > 0 and epoch % 5 == 0:
            torch.save(diffusion.model.state_dict(), f"L_label_model_{epoch}.pt")
            torch.save(optimizer.state_dict(), f"L_label_optimizer_{epoch}.pt")

        if epoch > 0 and epoch % 5 == 0:
            show_image(diffusion.generate(1, 2), save=True, name=f"L_1_Label_Epoch_{epoch}") # airplane
            show_image(diffusion.generate(1, 3), save=True, name=f"L_4_Label_Epoch_{epoch}") # deer
            show_image(diffusion.generate(1, 5), save=True, name=f"L_7_Label_Epoch_{epoch}") # horse
            show_image(diffusion.generate(1, 9), save=True, name=f"L_8_Label_Epoch_{epoch}") # ship
            ema_model = ema.getEMAModel()
            tmpDiff = Diffusion(device=device, num_classes=10)
            tmpDiff.model = ema_model
            show_image(tmpDiff.generate(1, 2), save=True, name=f"L_1_Label_EMA_Epoch_{epoch}") # airplane
            show_image(tmpDiff.generate(1, 3), save=True, name=f"L_4_Label_EMA_Epoch_{epoch}") # deer
            show_image(tmpDiff.generate(1, 5), save=True, name=f"L_7_Label_EMA_Epoch_{epoch}") # horse
            show_image(tmpDiff.generate(1, 9), save=True, name=f"L_8_Label_EMA_Epoch_{epoch}") # ship


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    start = time.time()
    #classes = set([1, 4, 7, 8])
    classes = set([2, 3, 5, 9])

    images = []
    labels = []
    # extract_all_data('data/data_batch_1', images, labels, classes)
    # extract_all_data('data/data_batch_2', images, labels, classes)
    # extract_all_data('data/data_batch_3', images, labels, classes)
    # extract_all_data('data/data_batch_4', images, labels, classes)
    # extract_all_data('data/data_batch_5', images, labels, classes)
    # extract_all_data('data/test_batch', images, labels, classes)
    read_binary_data('stl10_binary/train_X.bin', 'stl10_binary/train_Y.bin', images, labels, classes)
    images = np.array([normalize(squeeze01(x)) for x in images])
    print(images.shape)

    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)

    dataset = TorchDataset(images, labels)

    print(f"Length of dataset: {len(dataset)}")

    diffusion = Diffusion(device=device, num_classes=10, img_size=96)
    train(diffusion, 5e-5, 500, dataset, batch_size=1)

    end = time.time()
    print(f"Total time: {end - start}")