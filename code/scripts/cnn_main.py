import sys
sys.dont_write_bytecode = True

import os
import torch 
from torch.utils.data import TensorDataset , DataLoader
from torch.optim import Adam

from dataset import DatasetMNIST



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
    print('''HYPERPARAMETER:
    \tEpochs:\t{NUM_EPOCHS}
    \tBatch size:\t{BATCH_SIZE}
    \tLearning rate:\t{LEARNING_RATE}
    ''')

    # LOAD DATA (full; no split)
    data:DatasetMNIST = DatasetMNIST(is_debug=True)
    dataset:TensorDataset = data.get_data()
    loader_train:DataLoader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    # MODEL
    data_iter = iter(loader_train)
    images, labels = next(data_iter)
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")

    return 
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