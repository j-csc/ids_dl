# Core Libraries
import pandas as pd
import numpy as np
import os

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

# MLxtend
from mlxtend.feature_selection import ColumnSelector

# Misc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.utils import resample

class preprocessor():
  # Variables
  df = pd.DataFrame()

  # Init function
  def __init__(self):
    pass

  # Load bulk data
  def load_data_bulk(self, path="./data"):
    df_list = []
    for filename in os.listdir(path):
      print(("Loading data from {}\n").format(filename))
      temp_df = pd.read_csv(os.path.join('./data', filename), index_col=None)
      df_list.append(temp_df)
    self.df = pd.concat(df_list, axis=0, ignore_index=True)

  def preprocessing(self):
    # Rename columns
    print("Renaming columns...\n")

    self.df.rename(columns=lambda x: x.lower().lstrip()
          .rstrip().replace(" ", "_"), inplace=True)
    self.df['flow_bytes/s'] = self.df['flow_bytes/s'].astype('float64')
    self.df['flow_packets/s'] = self.df['flow_packets/s'].astype('float64')

    # Remove unneccesary columns
    print("Pruning unneccessary columns...\n")
    cols = [col for col in list(self.df.columns) if 'mean' not in col and 'variance' not in col 
            and 'std' not in col and 'min' not in col and 'max' not in col]
    self.df = self.df[cols]

    # For categorical labels, one hot encode
    print("One hot encoding categorical labels...\n")
    encoder = LabelEncoder()
    categorical_labels = (encoder.fit_transform(self.df.label))
    self.df = pd.concat([self.df.drop(['label'], 1),
              pd.DataFrame({'label': categorical_labels})], axis=1).reindex()

    # find all infinite or -infinite values
    print("Removing infinite and -infinite values...\n")
    self.df = self.df.replace([np.inf, -np.inf], np.nan)

    # Drop all nan values and remaining non-finite values
    self.df = self.df[~np.any(np.isnan(self.df), axis=1)]
    self.df = self.df[np.all(np.isfinite(self.df), axis=1)]

    # Obtaining cleaned data
    numerical_cols = self.df[[i for i in list(self.df.columns) if 'label' not in i]]
    target_variables = self.df.label

    # Saving data to hdf format
    print("Saving data...\n")
    self.df.to_hdf('./data/clean_df.h5', key='df', mode='w')
    numerical_cols.to_hdf('./data/clean_num_cols.h5', key='df', mode='w')
    target_variables.to_hdf('./data/clean_target_vars.h5', key='df', mode='w')

    print("Finished preprocessing data!\n")