import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from model import IVAE 

from helper_classes import LabelsKDD1999, LabelsMNIST, ModelToTrain


def determine_alpha(
        model:nn.Module, 
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
        model:nn.Module, 
        loader_train:DataLoader, 
        loader_test:DataLoader, 
        DEVICE:str,
        model_to_train:ModelToTrain
        ) :
    
    # determine alpha
    alpha:float = determine_alpha(model, loader_train, DEVICE)

    y_train:torch.Tensor = loader_train.dataset.tensors[1].squeeze().to(DEVICE)
    y_test:torch.Tensor  = loader_test.dataset.tensors[1].squeeze().to(DEVICE)

    max_prob:float = 0

    # detect anomalies
    anomalies_bitmask:[torch.Tensor] = []
    for x_batch, y_batch in loader_test:        

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        anomalies_batch:torch.Tensor 
        anomalies_batch, p = model.is_anomaly(x_batch, alpha)

        # LOGGING: max prob
        if p.max().item() > max_prob:
            max_prob = p.max().item()
            print(f'\tNew test max_recon_prob: {max_prob}')

        anomalies_bitmask.append(anomalies_batch)

    # concatenate all batches
    anomalies_bitmask = torch.cat(anomalies_bitmask, dim=0).bool()

    # combine anomalies with labels
    anomalies_bitmask = torch.stack([anomalies_bitmask, y_test], dim=1)
    
    # LOGGING: Distribution of classes
    data = []
    classes:[] = [c for c in LabelsKDD1999] if model_to_train == ModelToTrain.FULLY_TABULAR else [c for c in LabelsMNIST]
    for c in classes:
        c_int:int = c.value.encoded if model_to_train == ModelToTrain.FULLY_TABULAR else c.value
        c_label:str = c.value.label if model_to_train == ModelToTrain.FULLY_TABULAR else c.value

        len_train:int = len(y_train[y_train==c_int])
        len_test:int = len(y_test[y_test==c_int])
        len_anomalies:int = len(anomalies_bitmask[anomalies_bitmask[:,1]==c_int])

        data.append({
            'Class': c_label,
            '#Data (Train)': len_train,
            '#Data (Test)': len_test,
            '#Normals (Test)': len_test - len_anomalies,
            '#Anomalies (Test)': len_anomalies,
        })

    df:pd.DataFrame = pd.DataFrame(data)
    df.loc['Total'] = df.sum(numeric_only=True, axis=0)

    print(df.head(11))
    return