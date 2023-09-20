# run with
#   CUDA_VISIBLE_deviceS=0,1 nohup python main.py > log.txt           
#   or
#   CUDA_VISIBLE_deviceS=0,1 nohup python mnist_main.py > log.txt & (to run in background)


from enum import Enum
import sys
from time import sleep

from visualization import plot_mnist_orig_and_recon

sys.dont_write_bytecode = True

# libraries
import os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


# own classes
from model import VAE_CNN, VAE_Tabular
from dataset import IDataset, DatasetMNIST, DatasetKDD
from helper_classes import LabelsKDD1999, LabelsMNIST, ModelToTrain
from train import train
from anomaly_detection import detect_anomalies
from evaluation import get_metrics

def run_for_one_anomaly_class(
    # MODEL & ANOMALY CLASS
    is_model_probabilistic:bool = False,
    model_to_train:ModelToTrain = ModelToTrain.FULLY_TABULAR,
    anomaly_class:Enum = LabelsKDD1999.Normal,
    # DEVICE
    device:str = 'cuda:1',
    num_workers:int = 1,
    # HYPERPARAMETER
    num_epochs:int = 1,
    batch_size:int = 128,
    learning_rate:float = 1e-5,


):
    


    # LOAD DATA (full; no split)
    data:IDataset = DatasetMNIST(is_debug=True)  if model_to_train == ModelToTrain.CNN_MNIST else DatasetKDD(is_debug=True)
    dataset_train:TensorDataset = None 
    dataset_test:TensorDataset = None 
    dataset_train, dataset_test = data.get_data(anomaly_class=anomaly_class)
    
    loader_train:DataLoader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=True,
    )

    loader_test:DataLoader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    X, y = dataset_train.tensors 
    len = X.shape[0]

    X_test, y_test = dataset_test.tensors
    len_test = X_test.shape[0]

    print(f'''HYPERPARAMETER:
    \tProbabilistic:\t{is_model_probabilistic}
    \tModel:\t\t\t{model_to_train}
    \tAnomaly class:\t{anomaly_class.value}
    \tEpochs:\t\t\t{num_epochs}
    \tBatch size:\t\t{batch_size}
    \tLearning rate:\t{learning_rate}
    \tLength of training dataset {len}
    \tLength of test dataset {len_test}
    ''')



    # MODEL
    model:nn.Module = None 
    if model_to_train == ModelToTrain.CNN_MNIST:
        # LOGGING: show data properties (shapes, img resolution)
        img_resolution = (X.shape[2], X.shape[3])
        model:VAE_CNN = VAE_CNN(
            is_probabilistic=is_model_probabilistic,
            io_size=(img_resolution[0] * img_resolution[1])
        )

        print(f'''DATA SHAPE:
        Length of dataset {len}
        Labels shape: {y.shape}
        Images shape: {X.shape}
        Img resolution is {img_resolution}={img_resolution[0]*img_resolution[1]}
        ''')

    elif model_to_train == ModelToTrain.FULLY_TABULAR:
        model:VAE_Tabular = VAE_Tabular(is_probabilistic=is_model_probabilistic) 

    else:
        raise Exception('Invalid model to train')

    model.to(device)

    # OPTIMIZER
    OPTIMIZER:Adam = Adam(
        model.parameters(), 
        lr=learning_rate,
    )
    LR_SCHEDULER = StepLR(OPTIMIZER, step_size=5, gamma=0.1)  

    # TRAINING
    train(
        model=model, 
        model_to_train=model_to_train,
        device=device, 
        num_epochs=num_epochs ,
        optimizer=OPTIMIZER, 
        lr_scheduler=LR_SCHEDULER,
        train_loader=loader_train,
    )

    ##############################################
    # extract only anomalie instances of test data 
    #X_test, y_test = dataset_test.tensors
    #X_test = X_test[y_test == anomaly_class.value]
    #y_test = y_test[y_test == anomaly_class.value]
    #dataset_test_ANO = TensorDataset(X_test, y_test)
    #loader_testANO = DataLoader(
    #    dataset_test_ANO,
    #    batch_size=batch_size,
    #    num_workers=num_workers,
    #    shuffle=True,
    #)
    #if model_to_train == ModelToTrain.CNN_MNIST:
    #    
    #    for X, y in loader_testANO:
    #        X = X.to(device)
    #        y = y.to(device)
#
    #        batch_reconstructions:torch.Tensor = model.reconstruct(x=X)
    #        batch_reconstructions  = batch_reconstructions.squeeze(1)
    #        plot_mnist_orig_and_recon(
    #                batch_size=10, 
    #                x_orig=X, 
    #                x_recon=batch_reconstructions,
    #                y=y, 
    #            ) 
    #        
    #        sleep(1)
#
    
    # EVALUATION
    auc, f1, auc_prc = get_metrics(
        model=model,
        loader_test=loader_test,
        device=device,
        anomaly_class=anomaly_class,
        model_to_train=model_to_train,
        plot_roc=True,
    )
    
    print(f'\t\tAUC value = {auc.round(3)}')
    print(f'\t\tF1 value = {f1.round(3)}')
    print(f'\t\tAUC PRC value = {auc_prc.round(3)}')

    # ANOMALY DETECTION
    #print('ANOMALY DETECTION')
#
    #detect_anomalies(
    #    model=model,
    #    device=device,
    #    loader_train=loader_train,
    #    loader_test=loader_test,
    #    model_to_train=model_to_train
    #)

    return auc, f1, auc_prc



if __name__ == "__main__": 
    run_for_one_anomaly_class(
        anomaly_class= LabelsMNIST.Zero,
        model_to_train=ModelToTrain.CNN_MNIST,
        learning_rate=1e-4,
        batch_size=64
    )