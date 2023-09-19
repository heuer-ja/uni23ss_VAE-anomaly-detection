from typing import List
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch import Tensor 

from model import IVAE 
from helper_classes import LabelsKDD1999, ModelToTrain


def detect_alpha2(
        device:str, 
        model:IVAE,
        loader_train:DataLoader,
        model_to_train=ModelToTrain 
    ) -> float :
    print(f"\tDetecting alpha (threshold).\n\tModel is {'PROBABILISTIC' if model.is_probabilistic else 'DETERMINISTIC'}, so loss {'(i.e., recon. prob.)' if model.is_probabilistic else '(i.e., recon. error)'} of ANOMALIES should to be {'SMALLER' if model.is_probabilistic else 'BIGGER'} than alpha.")
    alpha:float = 1e10 if model.is_probabilistic else 0

    labels:List[int] = loader_train.dataset.tensors[1].unique().tolist()
    loss_per_class:dict = {label:[] for label in labels}    

    # 1. calc loss of each training instance
    for x_batch, y_batch in loader_train:
        x_batch = x_batch.to(device)

        # calc loss
        with torch.no_grad():
            pred_dict:dict = model.predict(x_batch)
            rec_loss:Tensor = model.get_reconstruction_loss(x_batch, pred_dict) # [batch_size]

        # append loss to list
        for i, label in enumerate(y_batch):
            loss_per_class[label.item()].append(rec_loss[i].item())

    # Dataframe to see loss distribution per class
    df_list = []  # Liste der DataFrames für jede Klasse
    for label in labels:
        df_batch = pd.DataFrame({
            'label': [label],
            'avg_loss': [sum(loss_per_class[label]) / len(loss_per_class[label])],
            'min_loss': [min(loss_per_class[label])],
            'max_loss': [max(loss_per_class[label])]
        })
        df_list.append(df_batch)
    df = pd.concat(df_list, ignore_index=True)

    # if KDD 1999, map label to str using LabelsKDD1999
    if model_to_train == ModelToTrain.FULLY_TABULAR:
        df = _df_label_mapping_kdd1999(df)

    print('\tLoss distribution per class (Training):')
    print(df.head(20))

    # alpha is highest avg loss
    alpha:float = df['avg_loss'].min() if model.is_probabilistic else df['avg_loss'].max()
    print(f'\n\tAlpha: {alpha}')

    return alpha

def detect_anomalies(
        device:str, 
        model:IVAE,
        loader_train:DataLoader, 
        loader_test:DataLoader,
        model_to_train:ModelToTrain
) -> None:
    model.eval()  
    # 1. determine alpha based on TRAINING data
    # 2. detect anomalies in TEST data based on alpha
    # 3. show anomalies distribution (i.e., how many anomalies are detected in each class)

    # alpha
    alpha:float = detect_alpha2(device, model, loader_train, model_to_train)

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
            rec_loss:dict = model.get_reconstruction_loss(x_batch, pred_dict)

        #print(f'\t\tLoss range: [{rec_loss.min().item()}, {rec_loss.max().item()}]')
        
        # anomalies
        is_anomaly_bitmask:torch.Tensor = rec_loss < alpha if model.is_probabilistic else rec_loss > alpha
        
        # create batch dataframe  
        df_batch = pd.DataFrame({
            'label': y_batch.cpu().numpy(),
            'loss': rec_loss.cpu().numpy(),
            'is_anomaly': is_anomaly_bitmask.cpu().numpy()
        })

        #print(df_batch.head(50))
        #return 

        # append to anomalies dataframe
        df_anomalies = pd.concat([df_anomalies, df_batch])

    if model_to_train == ModelToTrain.FULLY_TABULAR:
        df_anomalies = _df_label_mapping_kdd1999(df_anomalies)

    # anomalies distribution 
    print('\n\tAnomalies distribution (TEST):')
    df_result = df_anomalies.groupby(['label', 'is_anomaly']).size().reset_index(name='amount')
    total_count = df_result.groupby('label')['amount'].transform('sum')
    df_result['percentage'] = (df_result['amount'] / total_count * 100).round(2).astype(str) + '%'
    print(df_result.head(20))


    # avg loss per class
    df_new = df_anomalies.groupby('label')['loss'].mean().reset_index()
    df_new.columns = ['label', 'avg_loss']

    print(df_new.head(20))

    pass


def _df_label_mapping_kdd1999(df:pd.DataFrame) -> pd.DataFrame:
    label_mapping:dict = {si.value.encoded: si.value.label for si in LabelsKDD1999}
    df['label'] = df['label'].map(label_mapping)
    return df