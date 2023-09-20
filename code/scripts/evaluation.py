
from enum import Enum
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import IVAE
from helper_classes import ModelToTrain
from visualization import plot_roc_curve


PATH:str = '../plots/'


def get_test_data_loss(
        model:IVAE,
        loader_test:DataLoader,
        device:str,

    ) -> pd.DataFrame:

    model.eval()
    df:pd.DataFrame = pd.DataFrame(columns=['label', 'loss'])

    for X, y in loader_test:
        X = X.to(device)
        y = y.to(device)

        # calc loss
        with torch.no_grad():
            pred_dict:dict = model.predict(X)
            rec_loss:dict = model.get_reconstruction_loss(X, pred_dict)
        
        # create batch dataframe  
        df_batch = pd.DataFrame({
            'label': y.cpu().numpy(),
            'loss':  rec_loss.cpu().numpy(),
        })
        df = pd.concat([df, df_batch])

    # normalize loss
    df['loss_normalized'] = df['loss'].round(3)

    min_loss = df['loss'].min()
    max_loss = df['loss'].max()
    df['loss_normalized'] = ((df['loss'] - min_loss) / (max_loss - min_loss)).round(3)

    return df

def get_metrics(
    model:IVAE,
    loader_test:DataLoader,
    device:str,
    anomaly_class:Enum,
    model_to_train:ModelToTrain,
    plot_roc:bool = True,    
):
    print('EVALLUATION:')

    # test data loss
    df:pd.DataFrame = get_test_data_loss(
        model=model,
        loader_test=loader_test,
        device=device,
    )

    # add is_anomaly_class column to df
    anomaly_class_value = anomaly_class.value if model_to_train == ModelToTrain.CNN_MNIST else anomaly_class.value.encoded
    anomaly_class_label = anomaly_class.value if model_to_train == ModelToTrain.CNN_MNIST else anomaly_class.value.label
    df['is_anomaly_class'] = df['label'] == anomaly_class_value
    
    #1 AUC ROC
    # calc roc curve
    fpr, tpr, thresholds = roc_curve(y_true=df['is_anomaly_class'],y_score=df['loss_normalized'],)
    auc_score = auc(fpr, tpr)
    
    # plot ROC CURVE
    if plot_roc:
        plot_roc_curve(model_to_train,anomaly_class_label,fpr,tpr,auc_score)

    # 2. F1 scores
    f1_scores = [f1_score(df['is_anomaly_class'], df['loss_normalized'] > threshold) for threshold in thresholds]
    f1_max = np.max(f1_scores)

    # 3. AUC PRC
    auc_prc = average_precision_score(df['is_anomaly_class'], df['loss_normalized'])

    return auc_score, f1_max, auc_prc
