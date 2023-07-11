import numpy as np 
import torch 

from dataset import DatasetKDD
from model import VAE_Tabular
from torch.utils.data import TensorDataset 


def main():
    print("Hello")

    # LOAD DATA (full; no split)
    data = DatasetKDD(is_debug=True)
    dataset:TensorDataset = data.get_data()
    
    x1, y1 = dataset[0]

    # MODEL
    model = VAE_Tabular()
    encoded, z_mean, z_log_var, decoded  = model.forward(x1)

    print(x1)
    print(encoded)
    print(z_mean)
    print(z_log_var)
    print(decoded)

    #a = model.forward()

if __name__ == "__main__": 
    main()