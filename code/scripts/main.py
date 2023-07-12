import numpy as np 
import torch 
from torch.utils.data import TensorDataset , DataLoader
from torch.optim import Adam

from dataset import DatasetKDD
from model import VAE_Tabular
from train import train_vae_tabular

def main():

    # HYPERPARAMETERS
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 256
    NUM_EPOCHS = 50

    # Device
    CUDA_DEVICE_NUM = 1
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    print('Device:', DEVICE)


    # LOAD DATA (full; no split)
    data:DatasetKDD = DatasetKDD(is_debug=True)
    dataset:TensorDataset = data.get_data()
    loader_train:DataLoader = DataLoader(dataset, batch_size=16, shuffle=True)

    # MODEL
    model:VAE_Tabular = VAE_Tabular()
    model.to(DEVICE)

    # OPTIMIZER
    optimizer:Adam = Adam(model.parameters(), lr=LEARNING_RATE)

    # TRAINING
    log_dict:dict = train_vae_tabular(
        model=model, 
        num_epochs=10 ,
        optimizer=optimizer, 
        device=DEVICE, 
        train_loader=loader_train,
        skip_epoch_stats=True,
        logging_interval=1500
    )

if __name__ == "__main__": 
    main()