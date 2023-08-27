# run with
#   CUDA_VISIBLE_DEVICES=0,1 nohup python main.py > log.txt           
#   or
#   CUDA_VISIBLE_DEVICES=0,1 nohup python mnist_main.py > log.txt & (to run in background)


import sys


sys.dont_write_bytecode = True

# libraries
import os
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
from torch.optim import Adam

# own classes
from model import VAE_CNN, VAE_Tabular
from dataset import IDataset, DatasetMNIST, DatasetKDD
from helper_classes import LabelsMNIST, ModelToTrain
from train import train
from anomaly_detection import detect_anomalies


def main():
    MODEL_TO_TRAIN = ModelToTrain.CNN_MNIST	

    print(f'PROCESS ID:\t\t{os.getpid()}\n')

    # DEVICE
    CUDA_DEVICE_NUM = 1
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 1 if DEVICE == 'cpu' else 4
    print(f'DEVICE:\t\t{DEVICE} with {NUM_WORKERS} workers.\n')

    # HYPERPARAMETER
    if MODEL_TO_TRAIN == ModelToTrain.CNN_MNIST:
        NUM_EPOCHS = 2  if DEVICE == 'cpu' else 10
        BATCH_SIZE = 16 if DEVICE == 'cpu' else 64
        LEARNING_RATE = 5e-8 
    
    elif MODEL_TO_TRAIN == ModelToTrain.FULLY_TABULAR:
        NUM_EPOCHS = 2  if DEVICE == 'cpu' else 3
        BATCH_SIZE = 16 if DEVICE == 'cpu' else 128 
        LEARNING_RATE = 5e-7
    
    else:
        raise Exception('Invalid model to train')

    print(f'''HYPERPARAMETER:
    \tModel:\t\t\t{MODEL_TO_TRAIN}
    \tEpochs:\t\t\t{NUM_EPOCHS}
    \tBatch size:\t\t{BATCH_SIZE}
    \tLearning rate:\t{LEARNING_RATE}
    ''')


    # LOAD DATA (full; no split)
    data:IDataset = DatasetMNIST(is_debug=True)  if MODEL_TO_TRAIN == ModelToTrain.CNN_MNIST else DatasetKDD(is_debug=True)
    dataset_train:TensorDataset = None 
    dataset_test:TensorDataset = None 
    
    dataset_train, dataset_test = data.get_data(anomaly_class=LabelsMNIST.Four)
    loader_train:DataLoader = DataLoader(
        dataset_train, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    # MODEL
    model:nn.Module = None 
    X, y = dataset_train.tensors 
    if MODEL_TO_TRAIN == ModelToTrain.CNN_MNIST:
        # LOGGING: show data properties (len, shapes, img resolution)
        len = X.shape[0]
        img_resolution = (X.shape[2], X.shape[3])
        model:VAE_CNN = VAE_CNN(
            io_size=(img_resolution[0] * img_resolution[1])
        )

        print(f'''DATA SHAPE:
        Length of dataset {len}
        Labels shape: {y.shape}
        Images shape: {X.shape}
        Img resolution is {img_resolution}={img_resolution[0]*img_resolution[1]}

        ''')

    elif MODEL_TO_TRAIN == ModelToTrain.FULLY_TABULAR:
        model:VAE_Tabular = VAE_Tabular()

    else:
        raise Exception('Invalid model to train')

    
    return 
    model.to(DEVICE)

    # OPTIMIZER
    OPTIMIZER:Adam = Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
    )

    # TRAINING
    train(
        device=DEVICE, 
        model=model, 
        num_epochs=NUM_EPOCHS ,
        optimizer=OPTIMIZER, 
        train_loader=loader_train,
    )

    # ANOMALY DETECTION
    print('ANOMALY DETECTION')

    detect_anomalies(
        model=model,
        loader_train=loader_train,
        loader_test=None, # TODO
        DEVICE=DEVICE,
    )

    # detect anomalies
    #batch1_X, batch1_y = next(iter(loader_train))
    #batch1_X = batch1_X.to(DEVICE)
    #outliers = model.is_anomaly(batch1_X)
    #print(f'\tOutliers: {outliers}')

if __name__ == "__main__": 
    main()