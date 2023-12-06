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
from taesd.taesd import TAESD

def encodeBatch(batch, vae):
    with torch.no_grad():
        encoded = vae.encoder(batch)
        encoded = vae.scale_latents(encoded)
        return encoded

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
    vae = TAESD(encoder_path="taesd/taesd_encoder.pth", decoder_path="taesd/taesd_decoder.pth").to(device)
    diffusion.vae = vae

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        # start by training on one at a time
        num_batches = len(dataset) // batch_size

        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            # move this to the GPU
           # image = encodeBatch(image, vae)
            current_batch_size = image.shape[0]

            t = torch.randint(1, diffusion.noising_steps, (current_batch_size,))
            t = t.to(device)
            
            image_t, noise = diffusion.noise_image(image, t)

            if np.random.random() < 0.1:
                label = None

            pred_noise = diffusion.model(image_t, t, label)
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
        
        if epoch > 20 and epoch % 10 == 0:
            torch.save(diffusion.model.state_dict(), f"L_label_model_{epoch}.pt")
            torch.save(optimizer.state_dict(), f"L_label_optimizer_{epoch}.pt")

        if epoch > 0 and epoch % 10 == 0:
            show_4_images(diffusion.generate(4), save=True, name=f"11_256_vae{epoch}")
            ema_model = ema.getEMAModel()
            tmpDiff = Diffusion(device=device, num_classes=4, in_channels=4)
            tmpDiff.model = ema_model
            tmpDiff.vae = vae
            show_4_images(tmpDiff.generate(4), save=True, name=f"11_256_vae_ema{epoch}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    start = time.time()
    #classes = set([1, 4, 7, 8])
    #classes = set([2, 3, 5, 9])

    images = []
    labels = []
    # extract_all_data('data/data_batch_1', images, labels, classes)
    # extract_all_data('data/data_batch_2', images, labels, classes)
    # extract_all_data('data/data_batch_3', images, labels, classes)
    # extract_all_data('data/data_batch_4', images, labels, classes)
    # extract_all_data('data/data_batch_5', images, labels, classes)
    # extract_all_data('data/test_batch', images, labels, classes)
    # read_binary_data('stl10_binary/train_X.bin', 'stl10_binary/train_Y.bin', images, labels, classes)

    # images, labels = read_lineaus('linnaeus/train')
    # images = np.array([squeeze01(x) for x in images])
    # labels = np.array(labels)
    # #save images and labels to file
    # np.save('images.npy', images)
    # np.save('labels.npy', labels)
    # read images and labels from file
    images = np.load('encoded_images.npy')
    labels = np.load('labels.npy')

    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int32)

    # vae = TAESD(encoder_path="taesd/taesd_encoder.pth", decoder_path="taesd/taesd_decoder.pth").to(device)
    # new_ims = []
    # # for batch in images
    # for i in range(0, len(images), 100):
    #     print(i, len(images))
    #     tmp = images[i:i+100]
    #     tmp = tmp.to(device)
    #     encoded = encodeBatch(tmp, vae)
    #     for j in range(len(encoded)):
    #         new_ims.append(encoded[j].cpu().numpy())
    # images = np.array(new_ims)
    # #sav
    # np.save('encoded_images.npy', images)
    # exit()

    dataset = TorchDataset(images, labels)

    print(f"Length of dataset: {len(dataset)}")

    diffusion = Diffusion(device=device, num_classes=4, img_size=32, in_channels=4)
    train(diffusion, 6e-4, 500, dataset, batch_size=24)

    end = time.time()
    print(f"Total time: {end - start}")