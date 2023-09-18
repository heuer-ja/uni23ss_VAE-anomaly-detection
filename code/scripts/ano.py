import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from model import IVAE 


def detect_alpha2(
        device:str, 
        model:IVAE,
        loader_train:DataLoader, 
    ) -> float :
    # 1. calc loss of each training instance
    # 2a. if VAE is probabilistic: take min loss (lowest reconstruction probability)
    # 2b. if VAE is deterministic: take max loss (highest reconstruction error)


    alpha:float = 1e10 if model.is_probabilistic else 0

    for x_batch, _ in loader_train:
        x_batch = x_batch.to(device)

        # calc loss
        with torch.no_grad():
            pred_dict:dict = model.predict(x_batch)
            loss_dict:dict = model.get_loss(x_batch, pred_dict)

        loss:torch.Tensor = loss_dict['loss']
        temp_alpha:float = loss.min().item() if model.is_probabilistic else loss.max().item()

        # update alpha
        if model.is_probabilistic:
            if temp_alpha < alpha:
                alpha = temp_alpha
                print(f'\tNew alpha: {alpha}')
        else:   
            if temp_alpha > alpha:
                alpha = temp_alpha
                print(f'\tNew alpha: {alpha}')

    print(f'\tFinal Alpha: {alpha}\n')
    return alpha