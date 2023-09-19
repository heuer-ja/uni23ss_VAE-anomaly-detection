
from enum import Enum
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from helper_classes import ModelToTrain

from model import IVAE


PATH:str = '../plots/'


def get_test_data_loss(
        model:IVAE,
        loader_test:DataLoader,
        device:str,

    ) -> pd.DataFrame:
    model.eval()
    
    df:pd.DataFrame = pd.DataFrame(
        columns=['label', 'loss']
    )
    for x_batch, y_batch in loader_test:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # calc loss
        with torch.no_grad():
            pred_dict:dict = model.predict(x_batch)
            rec_loss:dict = model.get_reconstruction_loss(x_batch, pred_dict)
        # anomalies
        
        # create batch dataframe  
        df_batch = pd.DataFrame({
            'label': y_batch.cpu().numpy(),
            'loss': rec_loss.cpu().numpy(),
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
    fpr, tpr, thresholds = roc_curve(y_true=df['is_anomaly_class'],y_score=df['loss'],)
    auc_score = auc(fpr, tpr)
    
    # plot ROC CURVE
    if plot_roc:
        dataset_name = "mnist" if model_to_train == ModelToTrain.CNN_MNIST else "kdd"
        directory = f'roc_{dataset_name}'
        file_name:str = f'{PATH}{directory}/roc_anomaly-class-{anomaly_class_label}.png'

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(auc_score))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(file_name)

    # 2. F1 scores
    f1_scores = [f1_score(df['is_anomaly_class'], df['loss'] > threshold) for threshold in thresholds]
    f1_max = np.max(f1_scores)

    # 3. AUC PRC
    auc_prc = average_precision_score(df['is_anomaly_class'], df['loss'])

    return auc_score, f1_max, auc_prc
