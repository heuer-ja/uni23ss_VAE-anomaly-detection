
import time
import torch
import torch.nn.functional as F
import torch 
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model import VAE_CNN, VAE_Tabular, pVAE_CNN, pVAE_Tabular


from helper_classes import ModelToTrain, VAELogTrain, pVAELogTrain
from visualization import plot_VAE_train_pred, plot_pVAE_train_pred, plot_mnist_orig_and_recon

def train(
        device:str, 
        model:nn.Module, 
        is_model_probabilistic:bool,
        num_epochs:int, 
        optimizer:Optimizer, 
        lr_scheduler:StepLR,
        train_loader:DataLoader, 
    ):

    # LOGGING
    log_train_preds:VAELogTrain = pVAELogTrain([], [], []) if is_model_probabilistic else VAELogTrain([])

    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)	

            # FORWARD PASS
            forward_dict:dict = model(X)
           
            # LOSS
            loss = forward_dict['loss']

            # BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_train_preds.loss.append(loss)
            if is_model_probabilistic:
                log_train_preds.kl.append(forward_dict['kl'])
                log_train_preds.recon_loss.append(forward_dict['recon_loss'])

            if batch_idx % 500 == 0:
                if is_model_probabilistic:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f | KL: %.4f | RecLoss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss, forward_dict['kl'], forward_dict['recon_loss']))
                else:
                    print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                        % (epoch+1, num_epochs, batch_idx,
                            len(train_loader), loss))
                
        # RECONSTRUCTION
        print('Plot reconstruction after epoch: %d' % (epoch + 1))
        batch_reconstructions:torch.Tensor = model.reconstruct(x=X)
        batch_reconstructions  = batch_reconstructions.squeeze(1)
        
        plot_mnist_orig_and_recon(
            batch_size=len(X) //4, 
            x_orig=X, 
            x_recon=batch_reconstructions,
            y=y, 
        )

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        lr_scheduler.step()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # PLOT TRAINING PROGRESS
    plot_pVAE_train_pred(log_train_preds) if is_model_probabilistic else plot_VAE_train_pred(log_train_preds)
    return 


