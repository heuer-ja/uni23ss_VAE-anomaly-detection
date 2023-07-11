import pandas as pd
from abc import ABC, abstractclassmethod
from sklearn.preprocessing import StandardScaler

class IDataset(ABC):

    @abstractclassmethod
    def get_data(self) -> pd.DataFrame:
        # load data

        # get correct shape 
        #  - naming
        #  - positioning of columns
        #  - correct dtypes 
        #  - remove nans

        # normalization (0,1)

        # one hot encoding 
        pass     

    @abstractclassmethod
    def load(self) -> pd.DataFrame:
        pass 

    @abstractclassmethod
    def normalize(self) -> pd.DataFrame:
        '''standardization of numerical values (mean = 0, std. dev. = 1)'''
        pass 

    @abstractclassmethod
    def one_hot_encoding(self)-> pd.DataFrame:
        pass 

        
class DatasetKDD(IDataset):
    def __init__(self, is_debug=True) -> None:
        super().__init__()
        self.is_debug = is_debug
        pass 

    def get_data(self) -> pd.DataFrame:
        df = self.load()
        df = self.fix_dtypes(df)
        df = self.normalize(df)
        df = self.one_hot_encoding(df)

        # split into X,y 
        X,y = df.iloc[:, 2:], df.iloc[:, :2]

        # floatify
        X = X.astype(float)

        return X
     
    def load(self) -> pd.DataFrame:
        '''load local data from directory and setup a dataframe'''

        # FETCH DATA: 10% labeled data
        ### 1. Load
        df = pd.read_csv('../../data/kddcup.data_10_percent.gz', header=None)
        cols = pd.read_csv('../../data/kddcup.names',header=None)
        print('(✓) Downloaded labeled data') if self.is_debug else ''
        
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
        print('(✓) Loaded attack type data') if self.is_debug else ''

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
        print(f'''(✓) Fixed dtypes (int -> str) 
        with len =\t\t{len(df.select_dtypes(exclude=[float, int]).columns )} 
        with columns =\t{df.select_dtypes(exclude=[float, int]).columns}''') if self.is_debug else ''
        return df

    def normalize(self, df:pd.DataFrame)-> pd.DataFrame:
        '''standardization of numerical values (mean = 0, std. dev. = 1)'''

        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[float, int]).columns

        # Perform standardization
        scaler = StandardScaler(with_mean=True, with_std=True)
        df_standardized = df.copy()
        df_standardized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print('(✓) standardization of numerical values (mean = 0, std. dev. = 1)') if self.is_debug else ''

        return df_standardized

    def one_hot_encoding(self,df:pd.DataFrame)-> pd.DataFrame:
        ### Remove y-variables for one-hot-encoding
        df_no_ylabel = df.iloc[:, 2::1]
        non_number_cols = df_no_ylabel.select_dtypes(exclude=[float, int]).columns
        df_encoded = pd.get_dummies(df_no_ylabel, columns=non_number_cols)

        ### merge y-variable with one-hot encoded features
        df = pd.concat([df.iloc[:, 0:2:1], df_encoded], axis=1)
        print('(✓) one-hot encoded categorical columns') if self.is_debug else ''

        return df 


def main() -> None:
    kdd:DatasetKDD = DatasetKDD()
    df:pd.DataFrame = kdd.get_data()
    print('\n')
    print(df.head())

if __name__ == "__main__":
    main()