import torch
import pandas as pd
import  numpy as np

from abc import ABC, abstractclassmethod
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset 
from typing import Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from helper_classes import LabelsKDD1999, LabelsMNIST

class IDataset(ABC):

    @abstractclassmethod
    def get_data(self) -> TensorDataset:
        pass     

    @abstractclassmethod
    def load(self):
        pass 

    @abstractclassmethod
    def normalize(self):
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
            anomaly_class:LabelsMNIST = LabelsMNIST.Two
        ) -> Tuple[TensorDataset,TensorDataset]:
        print(f'LOADING DATA (anomaly class = {anomaly_class.value}):') if self.is_debug else ''

        # Load & pre-process data
        train_dataset, test_dataset = self.load()
        X_train, y_train = self.reshape(train_dataset)
        X_test, y_test = self.reshape(test_dataset)

        # extract anomaly class from train and add anomaly to test
        X_train, y_train, X_test, y_test = self.get_anomaly_train_test(
            X_train, y_train, 
            X_test, y_test, 
            anomaly_class=anomaly_class)
    
        # normalize pixel range from [0,255] to [0,1]
        X_train, X_test = self.normalize(X_train, X_test)

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

    def normalize(self, X_train:np.ndarray, X_test:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        '''normalizes pixel range from [0,255] to [0,1]'''


        X_train_new = X_train / 255
        X_test_new = X_test / 255

        print(f'\t\t(✓) normalized pixel range from [{X_train.min()}, {X_train.max()}] to[{X_train_new.min()}, {X_train_new.max()}]') if self.is_debug else ''

        return X_train_new, X_test_new


    def get_anomaly_train_test(self, 
                X_train:np.ndarray,
                y_train:np.ndarray,
                X_test:np.ndarray,
                y_test:np.ndarray,
                anomaly_class:LabelsMNIST
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        "splits dataset into train (only normal) and test (normal, and anomaly) set"

        # add anomaly class to test
        X_test = torch.cat([X_test, X_train[y_train == anomaly_class.value]], dim=0)
        y_test = torch.cat([y_test, y_train[y_train == anomaly_class.value]], dim=0)

        # remove anomaly class from train
        X_train = X_train[y_train != anomaly_class.value]
        y_train = y_train[y_train != anomaly_class.value]

        print(f'\t\t(✓) Training set only contains NORMALS, NO ANOMALIES CLASS {anomaly_class.value}.') if self.is_debug else ''
        print(f'\t\t\tlabels:\t{y_train.unique().tolist()}') if self.is_debug else ''
        print(f'\t\t(✓) Test set contains NORAMLS and ANOMALY CLASS {anomaly_class.value}.') if self.is_debug else ''
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

    def get_data(self, anomaly_class:LabelsKDD1999) -> [TensorDataset, TensorDataset]:
        print('LOADING DATA:') if self.is_debug else ''
        # Load & pre-process data (dataframe)
        df:pd.DataFrame = self.load()
        df = self.fix_dtypes(df)
        df = self.normalize(df)
        df = self.one_hot_encoding(df)

        # drop all rows with Attack Type = nan
        df = df.dropna(subset=['Attack Type'])

        # split into train and test
        df_train:pd.DataFrame = df.sample(frac=0.8, random_state=42)
        df_test:pd.DataFrame = df.drop(df_train.index)

        # extract anomaly class from df_train and add anomaly to df_test
        df_train, df_test = self.get_anomaly_train_test(df_train, df_test, anomaly_class)
    
        # split into X, y(encoded) 
        X_train, y_train_encoded = self.get_X_Yencoded(df_train)
        X_test, y_test_encoded = self.get_X_Yencoded(df_test)

        # to TensorDataset (DataLoader expects Dataset)
        dataset_train:TensorDataset = self.to_tensor_dataset(X_train, y_train_encoded)
        dataset_test:TensorDataset = self.to_tensor_dataset(X_test, y_test_encoded)

        print('\t\t(✓) loaded data\n') if self.is_debug else ''
        return dataset_train, dataset_test
     
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
        '''Normalize numerical values into [0, 1] range using Min-Max scaling.'''
     
        # Get numerical columns
        df_numerical = df.select_dtypes(include=[float, int])
        
        # Normalize using Min-Max scaling handling division by zero
        min_max_scaler = lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else x
        df_normalized = df_numerical.apply(min_max_scaler)
        
        # Combine the normalized numerical columns with the non-numerical columns
        df[df_normalized.columns] = df_normalized

        # Print min and max values for each numerical columns
        #for col in df_normalized.columns:
        #    print(f'\t\t\t{col}: [{df_normalized[col].min()}, {df_normalized[col].max()}]') if self.is_debug else ''

        print('\t\t(✓) normalized numerical columns into [0, 1] range') if self.is_debug else ''
        return df

        

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

        # create encoded y based LabelKDD1999
        y_encoded:np.ndarray = y['Attack Type'].map(
            {class_label.value.label : class_label.value.encoded for class_label in LabelsKDD1999}).values



        print('\t\t(✓) casted DataFrame into X, y (y is one-hot encoded)') if self.is_debug else ''
        return X,y_encoded
        
    def get_anomaly_train_test(
            self,
            df_train:pd.DataFrame,
            df_test:pd.DataFrame,
            anomaly_class:LabelsKDD1999                   
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "splits dataset into train (only normal) and test (normal, and anomaly) set"
        
        # add anomaly class to test
        df_test = pd.concat([df_test, df_train[df_train['Attack Type'] == anomaly_class.value.label]], ignore_index=True)

        # remove anomaly class from train
        df_train = df_train[df_train['Attack Type'] != anomaly_class.value.label]

        print(f'\t\t(✓) Training set only contains NORMALS, NO ANOMALIES CLASS {anomaly_class.value.label}.') if self.is_debug else ''
        print(f'\t\t\tlabels:\t{df_train["Attack Type"].unique().tolist()}') if self.is_debug else ''
        print(f'\t\t(✓) Test set contains NORAMLS and ANOMALY CLASS {anomaly_class.value.label}.') if self.is_debug else ''
        print(f'\t\t\tlabels:\t{df_test["Attack Type"].unique().tolist()}') if self.is_debug else ''
        return df_train, df_test

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