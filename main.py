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

CIFAR_SIZE = 32

def encodeBatch(batch, vae):
    with torch.no_grad():
        encoded = vae.encoder(batch)
        encoded = vae.scale_latents(encoded)
        return encoded

def train(diffusion, lr, num_epochs, dataset, batch_size):
    criterion = nn.MSELoss()
    longLoss = None

    print(f"Number of parameters: {diffusion.get_num_params()}")
    diffusion.model.to(device)
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=lr)

    diffusion.model.load_state_dict(torch.load('14_jan_model32_300.pt', map_location=device))
    optimizer.load_state_dict(torch.load('14_jan_optimizer32_300.pt', map_location=device))
    ema = ExponentialMovingAverage(0.998, diffusion.model)
    ema.loadModel('14_jan_ema32_300.pt')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    diffusion.model.train()
    # vae = TAESD(encoder_path="taesd/taesd_encoder.pth", decoder_path="taesd/taesd_decoder.pth").to(device)
    # diffusion.vae = vae

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
                longLoss = 0.9995 * longLoss + 0.0005 * loss.item()

            # f = open('loss_32.txt', 'a')
            # f.write(f"Epoch: {epoch}, Step {i} of {num_batches}: {loss.item()}\n")
            # f.close()
            
            if i % 100 == 0:
                print(f"Step {i} of {num_batches} Long: {round(longLoss, 6)}, Current: {round(loss.item(), 6)}")
        
        if epoch > 3 and epoch % 10 == 0:
            torch.save(diffusion.model.state_dict(), f"14_jan_model{CIFAR_SIZE}_{epoch}.pt")
            torch.save(optimizer.state_dict(), f"14_jan_optimizer{CIFAR_SIZE}_{epoch}.pt")
            ema.saveEMAModel(f"14_jan_ema{CIFAR_SIZE}_{epoch}.pt")

        if epoch > 0 and epoch % 20 == 0:
            show_4_images(diffusion.generate(4, 0)[0], save=True, name=f"example_gen{CIFAR_SIZE}_0_{epoch}")
            show_4_images(diffusion.generate(4, 1)[0], save=True, name=f"example_gen{CIFAR_SIZE}_1_{epoch}")
            show_4_images(diffusion.generate(4, 2)[0], save=True, name=f"example_gen{CIFAR_SIZE}_2_{epoch}")
            show_4_images(diffusion.generate(4, 3)[0], save=True, name=f"example_gen{CIFAR_SIZE}_3_{epoch}")
            ema_model = ema.getEMAModel()
            tmpDiff = Diffusion(device=device, num_classes=4, img_size=CIFAR_SIZE, in_channels=3)
            tmpDiff.model = ema_model
            tmpDiff.model.device = device
            show_4_images(tmpDiff.generate(4, 0)[0], save=True, name=f"example_gen{CIFAR_SIZE}_0_{epoch}_ema")
            show_4_images(tmpDiff.generate(4, 1)[0], save=True, name=f"example_gen{CIFAR_SIZE}_1_{epoch}_ema")
            show_4_images(tmpDiff.generate(4, 2)[0], save=True, name=f"example_gen{CIFAR_SIZE}_2_{epoch}_ema")
            show_4_images(tmpDiff.generate(4, 3)[0], save=True, name=f"example_gen{CIFAR_SIZE}_3_{epoch}_ema")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    start = time.time()

    if CIFAR_SIZE == 32:
        classes = set([1, 4, 7, 8])

        images = []
        labels = []
        extract_all_data('data/data_batch_1', images, labels, classes)
        extract_all_data('data/data_batch_2', images, labels, classes)
        extract_all_data('data/data_batch_3', images, labels, classes)
        extract_all_data('data/data_batch_4', images, labels, classes)
        extract_all_data('data/data_batch_5', images, labels, classes)
        extract_all_data('data/test_batch', images, labels, classes)

        images = np.array(images)
        labels = np.array(labels)
        images = np.reshape(images, (images.shape[0], 3, 32, 32))
    

    elif CIFAR_SIZE == 64:
        images = np.load('images.npy')
        labels = np.load('labels.npy')

    images = np.array([normalize(squeeze01(x)) for x in images])
    labels = np.array(labels)

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

    diffusion = Diffusion(device=device, num_classes=4, img_size=CIFAR_SIZE, in_channels=3)

    # 1e-4 and batch size 4 for 64
    train(diffusion, 6e-4, 1000, dataset, batch_size=24)

    end = time.time()
    print(f"Total time: {end - start}")