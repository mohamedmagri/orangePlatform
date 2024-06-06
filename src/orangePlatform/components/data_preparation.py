import os
from orangePlatform import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from orangePlatform.entity.config_entity import DataPreparationConfig
from sklearn.preprocessing import MinMaxScaler
import joblib


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path,index_col='Period')
        df= data['LTE'].copy()
        dates = pd.date_range(start='2022-03-23', periods=len(df), freq='D')
        df = df.to_frame()
        df.set_index(dates, inplace=True)
        # Split the data into training and test sets. (0.75, 0.25) split.
        train_lstm=df[:310]
        test_lstm=df[310:]
        scaler = MinMaxScaler()
        scaler.fit(train_lstm)
        scaled_train = scaler.transform(train_lstm)
        scaled_test = scaler.transform(test_lstm)
        scaled_trainn = pd.DataFrame(scaled_train,)
        scaled_testt = pd.DataFrame(scaled_test)

        scaled_trainn.to_csv(os.path.join(self.config.root_dir, "train_LTE.csv"),index = False)
        scaled_testt.to_csv(os.path.join(self.config.root_dir, "test_LTE.csv"),index = False)
        joblib.dump(scaler, os.path.join(self.config.root_dir, "scaler.joblib"))

        logger.info("Splited data into training and test sets")
        logger.info(scaled_train.shape)
        logger.info(scaled_test.shape)

        print(scaled_train.shape)
        print(scaled_test.shape)
        