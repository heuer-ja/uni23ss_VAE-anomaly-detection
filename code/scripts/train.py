
import time
import torch
import torch.nn.functional as F
import torch 
import torch.nn as nn

from model import VAE_Tabular
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torch.distributions import Normal, kl_divergence


def train_vae_tabular(
        model:VAE_Tabular, 
        num_epochs:int, 
        optimizer:Optimizer, 
        device:str, 
        train_loader:DataLoader, 
        loss_fn=None,
        logging_interval=100,  # TODO
        skip_epoch_stats=False, # TODO
        reconstruction_term_weight=1,# TODO
        save_model=None # TODO
    ):

    # assign loss function
    loss_fn = F.mse_loss if loss_fn is None else loss_fn
    
    log_dict:dict = {
        'train_combined_loss_per_batch': [],
        'train_combined_loss_per_epoch': [],
        'train_reconstruction_loss_per_batch': [],
        'train_kl_loss_per_batch': []
    }

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (X, _) in enumerate(train_loader):
            # FORWARD PASS
            encoded, z_mean, z_log_var, decoded = model(X)
            
           
            # LOSS
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                        - z_mean**2 
                                        - torch.exp(z_log_var), 
                                        axis=1) # sum over latent dimension

            batchsize = kl_div.size(0)
            kl_div = kl_div.mean() # average over batch dimension

            pixelwise = loss_fn(decoded, X, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels
            pixelwise = pixelwise.mean() # average over batch dimension

            loss = reconstruction_term_weight*pixelwise + kl_div


            # BACKWARD PASS
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))
    
        if not skip_epoch_stats:
                model.eval()
                
                with torch.set_grad_enabled(False):  # save memory during inference
                    
                    train_loss = compute_epoch_loss_autoencoder(
                        model, train_loader, loss_fn, device)
                    print('***Epoch: %03d/%03d | Loss: %.3f' % (
                        epoch+1, num_epochs, train_loss))
                    log_dict['train_combined_per_epoch'].append(train_loss.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
        
    return log_dict



def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss