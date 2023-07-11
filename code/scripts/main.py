import numpy as np 
import torch 

from dataset import DatasetKDD
from model import VAE_Tabular


def main():
    print("Hello")

    # LOAD DATA (full; no split)
    data = DatasetKDD(is_debug=True)
    df_X = data.get_data()
    
    row_df = df_X.iloc[:1, :]
    row: torch.Tensor = torch.from_numpy(np.array(row_df))

    # MODEL
    model = VAE_Tabular()
    encoded, z_mean, z_log_var, decoded  = model.forward(x=row)

    print(row)
    print(encoded)
    print(z_mean)
    print(z_log_var)
    print(decoded)

    #a = model.forward()

if __name__ == "__main__": 
    main()