import os

import numpy as np 
import torch 
from torch.utils.data import TensorDataset , DataLoader
from torch.optim import Adam

from dataset import DatasetKDD
from model import VAE_Tabular
from train import train_vae_tabular

def main():

    print(f'PID:\t\t{os.getpid()}')

    # Device
    CUDA_DEVICE_NUM = 1
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)

    # HYPERPARAMETERS
    LEARNING_RATE = 0.000005
    BATCH_SIZE = 16#128 
    NUM_EPOCHS = 2#4
    NUM_WORKERS = 1# 4



    # LOAD DATA (full; no split)
    data:DatasetKDD = DatasetKDD(is_debug=True)
    dataset:TensorDataset = data.get_data()
    loader_train:DataLoader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # MODEL
    model:VAE_Tabular = VAE_Tabular()
    model.to(DEVICE)

    # OPTIMIZER
    OPTIMIZER:Adam = Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
    )

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