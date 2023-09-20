
import sys
import time

sys.dont_write_bytecode = True

# libraries
import os
import torch 
from typing import List
from enum import Enum
import pandas as pd

# own classes
from helper_classes import LabelsKDD1999, LabelsMNIST, ModelToTrain
from run import run_for_one_anomaly_class


PATH:str = '../results/'

def main():
    print(f'PROCESS ID:\t\t{os.getpid()}\n')

    IS_MODEL_PROBABILISTIC = True
    MODEL_TO_TRAIN = ModelToTrain.FULLY_TABULAR

    # DEVICE
    CUDA_DEVICE_NUM = 1
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 1 if DEVICE == 'cpu' else 4
    print(f'DEVICE:\t\t{DEVICE} with {NUM_WORKERS} workers.\n')

    # HYPERPARAMETER
    if MODEL_TO_TRAIN == ModelToTrain.CNN_MNIST:
        NUM_EPOCHS = 2  if DEVICE == 'cpu' else 1
        BATCH_SIZE = 16 if DEVICE == 'cpu' else 64
        LEARNING_RATE = 1e-4
    
    elif MODEL_TO_TRAIN == ModelToTrain.FULLY_TABULAR:
        NUM_EPOCHS = 2  if DEVICE == 'cpu' else 1
        BATCH_SIZE = 16 if DEVICE == 'cpu' else 128 
        LEARNING_RATE = 1e-5

    else:
        raise Exception('Invalid model to train')
    
    # ANOMALY CLASS
    anomaly_enum:Enum = LabelsKDD1999 if MODEL_TO_TRAIN == ModelToTrain.FULLY_TABULAR else LabelsMNIST
    anomaly_classes:List[Enum] = [ano_class for ano_class in anomaly_enum]  

    # run for each anomaly class and track metrics
    df = pd.DataFrame(columns=['anomaly_class', 'auc', 'auc_prc', 'f1', 'anomaly_class_%_in_test_data'])

    start_time = time.time()

    for anomaly_class in anomaly_classes:

        print(f'RUNNING: FOR ANOMALY CLASS: ', anomaly_class.name)
        print('------------------------------------------------')
        auc:float = None
        auc_prc:float = None
        f1:float = None
        ano_class_percentage: float = None


        auc, f1, auc_prc, ano_class_percentage = run_for_one_anomaly_class(
            # MODEL & ANOMALY CLASS
            is_model_probabilistic=IS_MODEL_PROBABILISTIC,
            model_to_train=MODEL_TO_TRAIN,
            anomaly_class=anomaly_class,
            # DEVICE
            device=DEVICE,
            num_workers=NUM_WORKERS,
            # HYPERPARAMETER
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
        )

        df_temp:pd.DataFrame = pd.DataFrame({
            'anomaly_class': [anomaly_class.name],
            'auc': [auc.round(3)],
            'auc_prc': [auc_prc.round(3)],
            'f1': [f1.round(3)],
            'anomaly_class_%_in_test_data': [round(ano_class_percentage, 3)]
        })

        df = pd.concat([df, df_temp], ignore_index=True)

        print('\n\nTemporary overview about metrics:\n')
        print(df.head(100))

        print('\n\nTOTAL TIME ELAPSED: %.2f min' % ((time.time() - start_time)/60))
        print('================================================\n\n\n\n\n')

    print('FINAL RESULT')
    print(df.head(100))


    file_name:str = f'{PATH}{"probabilistic" if IS_MODEL_PROBABILISTIC else "deterministic"}-VAE_{"mnist" if MODEL_TO_TRAIN==ModelToTrain.CNN_MNIST else "kdd"}_results.csv'
    print(f'Saving results to {file_name}')
    df.to_csv(file_name, index=False)


    print('\n\nScript finished.')
    return 


if __name__ == "__main__": 
    main()