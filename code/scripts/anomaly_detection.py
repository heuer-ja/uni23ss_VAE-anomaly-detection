import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import IVAE 

def determine_alpha(
        model:IVAE, 
        loader_train:DataLoader,
        DEVICE:str
        ) -> float:
    '''
    alpha is highest reconstruction probability of train data X
    '''
    # detect alpha (max reconstruction prob. of train data)
    alpha:float = 0
    for x_batch, _ in loader_train:        
        x_batch = x_batch.to(DEVICE)
        probs:torch.Tensor = model.reconstruction_probability(x_batch)

        if probs.max().item() > alpha:
            alpha = probs.max().item()
            print(f'\tNew alpha: {alpha}')

    print(f'\tFinal Alpha: {alpha}\n')
    return alpha

def detect_anomalies(
        model:IVAE, 
        loader_train:DataLoader, 
        loader_test:DataLoader, 
        DEVICE:str) :
    
    # determine alpha
    alpha:float = determine_alpha(model, loader_train, DEVICE)

    # detect anomalies
    anomalies:torch.Tensor = None
    for x, _ in loader_test:        
        x_batch = x_batch.to(DEVICE)
        anomalies:torch.Tensor = model.is_anomaly(x_batch, alpha)
        print(anomalies)

        # TODO: combine anomalies with label 
        
    return