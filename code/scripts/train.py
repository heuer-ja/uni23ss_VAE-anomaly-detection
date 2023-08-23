
import time
import torch
import torch.nn.functional as F
import torch 
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
        device:str, 
        model:nn.Module, 
        num_epochs:int, 
        optimizer:Optimizer, 
        train_loader:DataLoader, 
    ):

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (X, _) in enumerate(train_loader):
            X = X.to(device)

            # FORWARD PASS
            forward_dict:dict = model(X)
           
            # LOSS
            loss = forward_dict['loss']
            kl = forward_dict['kl']
            recon_loss = forward_dict['recon_loss']

            # BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if batch_idx % 500 == 0:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | KL: %.4f | RecLoss: %.4f'
                      % (epoch, num_epochs, batch_idx,
                          len(train_loader), loss, kl, recon_loss))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    return 


