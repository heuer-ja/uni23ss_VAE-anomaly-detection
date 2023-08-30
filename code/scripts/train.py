
import time
import torch
import torch.nn.functional as F
import torch 
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader

from helper_classes import LogTrainPreds
from visualization import plot_train_preds

def train(
        device:str, 
        model:nn.Module, 
        num_epochs:int, 
        optimizer:Optimizer, 
        train_loader:DataLoader, 
    ):

    log_train_preds:LogTrainPreds = LogTrainPreds([], [], [])

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
            log_train_preds.loss.append(loss)
            log_train_preds.kl.append(kl)
            log_train_preds.recon_loss.append(recon_loss)

            if batch_idx % 500 == 0:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | KL: %.4f | RecLoss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss, kl, recon_loss))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # PLOT TRAINING PROGRESS
    log_train_preds.loss = torch.stack(log_train_preds.loss).cpu().detach().numpy()
    log_train_preds.kl = torch.stack(log_train_preds.kl).cpu().detach().numpy()
    log_train_preds.recon_loss = torch.stack(log_train_preds.recon_loss).cpu().detach().numpy()

    plot_train_preds(log_train_preds)
    return 


