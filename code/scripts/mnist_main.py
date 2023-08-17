import sys


sys.dont_write_bytecode = True

import os
import torch 
from torch.utils.data import TensorDataset , DataLoader
from torch.optim import Adam

from mnist_model import VAE_CNN
from dataset import DatasetMNIST
from tabular_train import train_vae_tabular



def main():
    print(f'PROCESS ID:\t\t{os.getpid()}\n')

    # Device
    CUDA_DEVICE_NUM = 1
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 1 if DEVICE == 'cpu' else 4
    print(f'DEVICE {DEVICE} with {NUM_WORKERS} workers.\n')

    # HYPERPARAMETER
    NUM_EPOCHS = 2  if DEVICE == 'cpu' else 3
    BATCH_SIZE = 16 if DEVICE == 'cpu' else 128
    LEARNING_RATE = 0.000005

    print(f'''HYPERPARAMETER:
    \tEpochs:\t\t\t{NUM_EPOCHS}
    \tBatch size:\t\t{BATCH_SIZE}
    \tLearning rate:\t{LEARNING_RATE}
    ''')

    # LOAD DATA (full; no split)
    data:DatasetMNIST = DatasetMNIST(is_debug=True)
    dataset:TensorDataset = data.get_data()

    X, y = dataset.tensors 

    # LOGGING: show data properties (len, shapes, img resolution)
    len = X.shape[0]
    img_resolution = (X.shape[2], X.shape[3])
    print(f"Length of dataset {len}")
    print(f"Batch of labels shape: {y.shape}")
    print(f"Batch of images shape: {X.shape}")
    print(f'Img resolution is {img_resolution}={img_resolution[0]*img_resolution[1]}')
    print('\n')

    loader_train:DataLoader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    # MODEL
    model:VAE_CNN = VAE_CNN(
        io_size=(img_resolution[0] * img_resolution[1])
    )
    model.to(DEVICE)

    # OPTIMIZER
    OPTIMIZER:Adam = Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
    )

    # TEST
    X1, _ = next(iter(loader_train))    

    X1 = X1.to(DEVICE)
    pred = model(X1)
    print('Done')


    return 
    # TRAINING
     
    train_vae_tabular(
        device=DEVICE, 
        model=model, 
        num_epochs=NUM_EPOCHS ,
        optimizer=OPTIMIZER, 
        train_loader=loader_train,
    )

if __name__ == "__main__": 
    main()