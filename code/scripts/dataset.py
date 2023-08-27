import torch
import pandas as pd
import  numpy as np

from abc import ABC, abstractclassmethod
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset 
from typing import Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from helper_classes import LabelsMNIST

class IDataset(ABC):

    @abstractclassmethod
    def get_data(self) -> TensorDataset:
        pass     

    @abstractclassmethod
    def load(self):
        pass 

    @abstractclassmethod
    def to_tensor_dataset(self) -> TensorDataset:
        pass

    @abstractclassmethod
    def get_anomaly_train_test(self, 
                X_train:np.ndarray,
                y_train:np.ndarray,
                X_test:np.ndarray,
                y_test:np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        "splits dataset into train (only normal) and test (normal, and anomaly) set"
        pass


class DatasetMNIST(IDataset):
    def __init__(self, is_debug=True) -> None:
        super().__init__()
        self.is_debug = is_debug
        pass 


    def get_data(
            self, 
            anomaly_class:LabelsMNIST = LabelsMNIST.Two.value
        ) -> Tuple[TensorDataset,TensorDataset]:
        print(f'LOADING DATA (anomaly class = {anomaly_class}):') if self.is_debug else ''

        # Load & pre-process data
        train_dataset, test_dataset = self.load()
        X_train, y_train = self.reshape(train_dataset)
        X_test, y_test = self.reshape(test_dataset)

        # extract anomaly class from train and add anomaly to test
        X_train, y_train, X_test, y_test = self.get_anomaly_train_test(
            X_train, y_train, 
            X_test, y_test, 
            anomaly_class=anomaly_class)

        # convert to TensorDataset
        dataset_train:TensorDataset = self.to_tensor_dataset(X_train, y_train) 
        dataset_test:TensorDataset = self.to_tensor_dataset(X_test, y_test)

        print('\t\t(✓) loaded data\n') if self.is_debug else ''
        return dataset_train, dataset_test
    

    def load(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transform, download=True)

        print('\t\t(✓) downloaded data (train & test)') if self.is_debug else ''
        return train_dataset, test_dataset

    def reshape(self, dataset) -> Tuple[np.ndarray,np.ndarray]:
        y = dataset.targets
        X = dataset.data
        X = X.view(-1, 1, 28, 28)  # Reshape to [batch_size, channels, height, width]
        print('\t\t(✓) reshaped X for train/test') if self.is_debug else ''
        return X,y
    

    def get_anomaly_train_test(self, 
                X_train:np.ndarray,
                y_train:np.ndarray,
                X_test:np.ndarray,
                y_test:np.ndarray,
                anomaly_class:LabelsMNIST
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        "splits dataset into train (only normal) and test (normal, and anomaly) set"

        # add anomaly class to test
        X_test = torch.cat([X_test, X_train[y_train == anomaly_class]], dim=0)
        y_test = torch.cat([y_test, y_train[y_train == anomaly_class]], dim=0)

        # remove anomaly class from train
        X_train = X_train[y_train != anomaly_class]
        y_train = y_train[y_train != anomaly_class]

        print(f'\t\t(✓) Training set only contains NORMALS, NO ANOMALIES CLASS {anomaly_class}.') if self.is_debug else ''
        print(f'\t\t\tlabels:\t{y_train.unique().tolist()}') if self.is_debug else ''
        print(f'\t\t(✓) Test set contains NORAMLS and ANOMALY CLASS {anomaly_class}.') if self.is_debug else ''
        print(f'\t\t\tlabels:\t{y_test.unique().tolist()}') if self.is_debug else ''

        return X_train, y_train, X_test, y_test
    
    def to_tensor_dataset(self, X:np.ndarray,y:np.ndarray ) -> TensorDataset:
        tensor_dataset:TensorDataset = TensorDataset(X,y)
        print('\t\t(✓) casted X,y to TensorDataset for train/test') if self.is_debug else ''
        return tensor_dataset

        
class DatasetKDD(IDataset):
    def __init__(self, is_debug=True) -> None:
        super().__init__()
        self.is_debug = is_debug
        pass 

    def get_data(self) -> TensorDataset:
        print('LOADING DATA:') if self.is_debug else ''
        df:pd.DataFrame
        X:np.ndarray
        y_encoded:np.ndarray
        dataset:TensorDataset

        # Load & pre-process data (dataframe)
        df = self.load()
        df = self.fix_dtypes(df)
        df = self.normalize(df)
        df = self.one_hot_encoding(df)

        # split into X, y(encoded) 
        X, y_encoded = self.get_X_Yencoded(df)

        # to TesnorDataset (DataLoader expects Dataset)
        dataset = self.to_tensor_dataset(X, y_encoded)
        print('\t\t(✓) loaded data\n') if self.is_debug else ''
        return dataset
     
    def load(self) -> pd.DataFrame:
        '''load local data from directory and setup a dataframe'''

        # FETCH DATA: 10% labeled data
        ### 1. Load
        df = pd.read_csv('../../data/kddcup.data_10_percent.gz', header=None)
        cols = pd.read_csv('../../data/kddcup.names',header=None)
        print('\t\t(✓) downloaded labeled data') if self.is_debug else ''
        
        ### 2. Add column names to DataFrame
        if cols[0][0] == 'back':
            cols = cols.drop(cols.index[0])
            cols.reset_index(drop=True, inplace=True)

        cols = cols.dropna(axis=1)
         
        ### split merged column names (name:type --> name | type)
        cols[[0,1]] = cols[0].str.split(':',expand = True)

        ### add column names to DataFrame
        names = cols[0].tolist()
        names.append('label')
        df.columns = names

        ### 3. Rename y-labels and show their occurances
        df['label'] = df['label'].str.replace('.', '', regex=False)
        df.groupby(['label']).size().sort_values()

        #--------------------------------
        # FETCH DATA: ATTACK TYPE  (Summarize labels (i. e., attack types))
        ### 1. Download Attack Types 
        df_attack_types = pd.read_csv('../../data/training_attack_types')
        print('\t\t(✓) downloaded attack type data') if self.is_debug else ''

        ### 2. Split columns (fetched data contains two features in one column, so split it)
        df_temp = pd.DataFrame(columns=['Attack','Type'])
        df_temp[['Attack','Type']] = df_attack_types['back dos'].str.split(' ', expand=True)

        row_normal = pd.DataFrame({'Attack': 'normal', 'Type':'normal'}, index=[0]) # add normal to attacks
        df_temp = pd.concat([df_temp, row_normal], ignore_index=True)

        ### 3. Add column 'Attack Type' to df 
        df['Attack Type'] = df['label'].map(df_temp.set_index('Attack')['Type'])

        ### 4. Rearrange columns
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index('Attack Type')))
        df = df.loc[:, cols]
        cols.insert(1, cols.pop(cols.index('label')))
        df = df.loc[:, cols]
        return df 

    def fix_dtypes(self, df:pd.DataFrame)-> pd.DataFrame:
        ''' Some columns are categorical (0,1) 
            but due to the import they are considered to be numerical
            '''
        cols_categorical = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
        df[cols_categorical] = df[cols_categorical].astype(str)
        print(f'''\t\t(✓) fixed dtypes (int -> str) 
        \twith len =\t\t{len(df.select_dtypes(exclude=[float, int]).columns )} 
        \twith columns =\t{df.select_dtypes(exclude=[float, int]).columns.values}''') if self.is_debug else ''
        return df

    def normalize(self, df:pd.DataFrame)-> pd.DataFrame:
        '''standardization of numerical values (mean = 0, std. dev. = 1)'''

        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[float, int]).columns

        # Perform standardization
        scaler = StandardScaler(with_mean=True, with_std=True)
        df_standardized = df.copy()
        df_standardized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print('\t\t(✓) standardization of numerical values (mean = 0, std. dev. = 1)') if self.is_debug else ''

        return df_standardized

    def one_hot_encoding(self,df:pd.DataFrame)-> pd.DataFrame:
        ### Remove y-variables for one-hot-encoding
        df_no_ylabel = df.iloc[:, 2::1]
        non_number_cols = df_no_ylabel.select_dtypes(exclude=[float, int]).columns
        df_encoded = pd.get_dummies(df_no_ylabel, columns=non_number_cols).astype(float)

        ### merge y-variable with one-hot encoded features
        df = pd.concat([df.iloc[:, 0:2:1], df_encoded], axis=1)
        print('\t\t(✓) one-hot encoded categorical columns') if self.is_debug else ''

        return df 

    def get_X_Yencoded(self, df:pd.DataFrame)-> Tuple[np.ndarray,np.ndarray]:
        ''' splits data into X and y
            and encodes y
        '''
        # split into X,y 
        X:pd.DataFrame = df.iloc[:, 2:]
        y:pd.DataFrame = df.iloc[:, :1] # only use ['Attack Type'], not ['label']

        # X to numpy
        X:np.ndarray = X.values 

        # encode labels
        label_encoder = LabelEncoder()
        y_encoded:np.ndarray = label_encoder.fit_transform(y.values.ravel())
        print('\t\t(✓) casted DataFrame into X, y (y is encoded)') if self.is_debug else ''

        return X,y_encoded

    def to_tensor_dataset(self, X:np.ndarray, y:np.ndarray) -> TensorDataset:
        '''
        transform X,y to `TensorDataset`, since `DataLoader` expects it for training
        '''
        # to Dataset (DataLoader expects Dataset)
        X_tensor:torch.Tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor:torch.Tensor  = torch.tensor(y, dtype=torch.float32)

        # floatify
        X_tensor = X_tensor.float()
        y_tensor = y_tensor.float()

        dataset:TensorDataset = TensorDataset(X_tensor, y_tensor)
        print('\t\t(✓) casted X,y to TensorDataset') if self.is_debug else ''
        return dataset

def main() -> None:
    kdd:DatasetKDD = DatasetKDD()
    df:pd.DataFrame = kdd.get_data()
    print('\n')
    print(df.head())

if __name__ == "__main__":
    main()