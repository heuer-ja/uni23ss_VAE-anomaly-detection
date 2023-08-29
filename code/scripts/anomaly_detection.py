import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from typing import List
from model import IVAE 

from helper_classes import dict_kdd1999_labels


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
        DEVICE:str,
        class_labels:List 
        ) :
    
    # determine alpha
    alpha:float = determine_alpha(model, loader_train, DEVICE)

    y_train = loader_train.dataset.tensors[1].squeeze().to(DEVICE)
    y_test = loader_test.dataset.tensors[1].squeeze().to(DEVICE)

    # detect anomalies
    anomalies_bitmask:List = []
    for x_batch, y_batch in loader_test:        

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        anomalies_batch:torch.Tensor = model.is_anomaly(x_batch, alpha)
        anomalies_bitmask.append(anomalies_batch)

    # concatenate all batches
    anomalies_bitmask = torch.cat(anomalies_bitmask, dim=0).bool()

    # combine anomalies with labels
    anomalies_bitmask = torch.stack([anomalies_bitmask, y_test], dim=1)
    
    # Distribution of classes
    d = []
    for c in class_labels:

        len_train:int = len(y_train[y_train==c])
        len_test:int = len(y_test[y_test==c])
        len_anomalies:int = len(anomalies_bitmask[anomalies_bitmask[:,1]==c])
        d.append({
            'Class': c,
            '#Data (Train)': len_train,
            '#Data (Test)': len_test,
            '#Normals (Test)': len_test - len_anomalies,
            '#Anomalies (Test)': len_anomalies,
        })

    df:pd.DataFrame = pd.DataFrame(d)
    df.loc['Total'] = df.sum(numeric_only=True, axis=0)

    print(df.head(11))
    return 

    ### THIS WORKS FOR KDD1999 ONLY

    # print distribution of classes 
    print('\nDistribution TRAIN SET')
    print(f'Class || #Instances')
    y_train = loader_train.dataset.tensors[1].squeeze().to(DEVICE)
    for k, v in dict_kdd1999_labels.items():
        instances = len(y_train[y_train==v])
        print(f'{k} || {instances}')


    print('\nDistribution TEST SET')
    print(f'Class || #Instances | #Normals | #Anomalies')
    for k, v in dict_kdd1999_labels.items():
        instances = len(y_test[y_test==v])
        anomalies = len(anomalies_bitmask[anomalies_bitmask[:,1]==v])
        normals = instances - anomalies

        print(f'{k} || {instances} | {normals} | {anomalies}')

    return