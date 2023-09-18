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


    print(f"\tDetecting alpha (threshold).\n\tModel is {'PROBABILISTIC' if model.is_probabilistic else 'DETERMINISTIC'}, so alpha is {'MIN' if model.is_probabilistic else 'MAX'} loss {'(i.e., recon. prob.)' if model.is_probabilistic else '(i.e., recon. error)'} of all TRAINING instances")
    alpha:float = 1e10 if model.is_probabilistic else 0

    # 1. calc loss of each training instance
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
                print(f'\t\tNew alpha: {alpha}')
        else:   
            if temp_alpha > alpha:
                alpha = temp_alpha
                print(f'\t\tNew alpha: {alpha}')

    # multiply alpha by factor
    alpha *= 2

    print(f'\t\tFinal Alpha: {alpha}\n')
    return alpha

def detect_anomalies(
        device:str, 
        model:IVAE,
        loader_train:DataLoader, 
        loader_test:DataLoader,
) -> None:
    
    # 1. determine alpha based on TRAINING data
    # 2. detect anomalies in TEST data based on alpha
    # 3. show anomalies distribution (i.e., how many anomalies are detected in each class)

    # alpha
    alpha:float = detect_alpha2(device, model, loader_train)

    # anomalies
    df_anomalies:pd.DataFrame = pd.DataFrame(
        columns=['label', 'loss', 'is_anomaly']
    )

    for x_batch, y_batch in loader_test:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # calc loss
        with torch.no_grad():
            pred_dict:dict = model.predict(x_batch)
            loss_dict:dict = model.get_loss(x_batch, pred_dict)

        loss:torch.Tensor = loss_dict['loss']
        print(f'\t\tLoss range: [{loss.min().item()}, {loss.max().item()}]')
        
        # anomalies
        is_anomaly_bitmask:torch.Tensor = loss < alpha if model.is_probabilistic else loss > alpha
        
        # create batch dataframe  
        df_batch = pd.DataFrame({
            'label': y_batch.cpu().numpy(),
            'loss': loss.cpu().numpy(),
            'is_anomaly': is_anomaly_bitmask.cpu().numpy()
        })

        print(df_batch.head(20))
        return 

        # append to anomalies dataframe
        df_anomalies = pd.concat([df_anomalies, df_batch])


    # anomalies distribution 
    print('\tAnomalies distribution:')
    df_result = df_anomalies.groupby(['label', 'is_anomaly']).size().reset_index(name='amount')
    total_count = df_result.groupby('label')['amount'].transform('sum')
    df_result['percentage'] = (df_result['amount'] / total_count * 100).round(2).astype(str) + '%'
    print(df_result.head(20))

    pass