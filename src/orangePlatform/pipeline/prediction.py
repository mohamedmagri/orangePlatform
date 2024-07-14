import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from orangePlatform.entity.config_entity import ModelTrainerConfig
from sklearn.preprocessing import MinMaxScaler
from orangePlatform.components.data_preparation import MinMaxScaler
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import subprocess
import threading
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt




def load_scaler_from_artifact(artifact_folder):
    
    files = os.listdir(artifact_folder)

    
    joblib_files = [file for file in files if file.endswith('.joblib')]

    
    joblib_files.sort()

    
    if joblib_files:
        
        first_joblib_file = joblib_files[0]
        scaler_path = os.path.join(artifact_folder, first_joblib_file)
        scaler = joblib.load(scaler_path)
        return scaler
    else:
        print("No CSV files found in the artifact folder.")
        return None



artifact_folder_path = 'C:\\Users\\mohamed\\Desktop\\stage pfe Orange\\orangePlatform\\orangePlatform\\artifacts\\data_preparation'
#C:\Users\mohamed\Desktop\stage pfe Orange\orangePlatform\orangePlatform\artifacts\data_preparation
scaler = load_scaler_from_artifact(artifact_folder_path)




def load_data_from_artifact(artifact_folder):
    
    files = os.listdir(artifact_folder)

    
    csv_files = [file for file in files if file.endswith('.csv')]

    
    csv_files.sort()

    
    if csv_files:
        
        first_csv_file = csv_files[0]
        csv_path = os.path.join(artifact_folder, first_csv_file)
        csv = pd.read_csv(csv_path,index_col='Period')
        return csv
    else:
        print("No CSV files found in the artifact folder.")
        return None

artifact1_folder_path='C:\\Users\\mohamed\\Desktop\\stage pfe Orange\\orangePlatform\\orangePlatform\\artifacts\\data_ingestion'
#C:\Users\mohamed\Desktop\stage pfe Orange\orangePlatform\orangePlatform\artifacts\data_preparation
new_data = load_data_from_artifact(artifact1_folder_path)


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


class PredictionPipeline:
    def __init__(self,  config: ModelTrainerConfig):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.config = config
        self.retraining = False

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def check_loss(self):
        
        df= new_data['LTE'].copy()
        dates = pd.date_range(start='2022-03-23', periods=len(df), freq='D')
        df = df.to_frame()
        df.set_index(dates, inplace=True)
        
        test_lstm=df[310:]
        scaled_test = scaler.transform(test_lstm)
        scaled_testt = pd.DataFrame(scaled_test)
        test=scaled_testt.values
        
        n_features = 1
        n_seq =5
        n_steps = int(self.config.nsteps // n_seq)
        X,y=split_sequence(test, self.config.nsteps)
        
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
        yhat = self.model.predict(X)
        (rmse, mae, r2) = self.eval_metrics(y, yhat)
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        return scores

    def get_model_metrics(self):
        file_path = 'C:\\Users\\mohamed\\Desktop\\stage pfe Orange\\orangePlatform\\orangePlatform\\artifacts\\model_evaluation\\metrics.json'

        # Print the last modified time
        #print("Last modified:", time.ctime(os.path.getmtime(file_path)))

        # Load the metrics.json file
        with open(file_path, 'r') as file:
            metrics = json.load(file)

        # Print the loaded metrics for debugging
        #print("Metrics loaded from file:", metrics)

        # Extract the metrics
        rmse = metrics['rmse']
        mae = metrics['mae']
        r2 = metrics['r2']

        scores = {"rmse": rmse, "mae": mae, "r2": r2}
    
        return scores

    # def get_model_metrics(self):
    #     # Load the metrics.json file
    #     with open('C:/Users/mohamed/Desktop/stage pfe Orange/orangePlatform1/orangePlatform/artifacts/model_evaluation/metrics.json', 'r') as file:
    #         metrics = json.load(file)

    #     # Extract the metrics
    #     rmse = metrics['rmse']
    #     mae = metrics['mae']
    #     r2 = metrics['r2']

    #     scores = {"rmse": rmse, "mae": mae, "r2": r2}
        
    #     return scores
  
    def trigger_retraining(self):
        if not self.retraining:
            self.retraining = True
            threading.Thread(target=self.retrain_pipeline).start()

    def retrain_pipeline(self):
        subprocess.run(["dvc", "repro", "-f"], check=True)
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.retraining = False
        
    
    def predict(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        test=test_data.values
        n_features = 1
        n_seq =5
        n_steps = int(self.config.nsteps // n_seq)
        


        dict={}
        dict['2024-03-21']=1.437148e+06
        dates=['2024-03-21']
        i=0
        test=test_data.values

        while i<15 :

            a,b=split_sequence(test, self.config.nsteps)
            test = np.roll(test, -1, axis=0)
            x_input = a.reshape((a.shape[0], n_seq, 1, n_steps, n_features))

            yhat = self.model.predict(x_input, verbose=0)# forecasting using ConvLSTM
            yhat = scaler.inverse_transform(yhat)
            
            date = datetime.strptime(dates[i], '%Y-%m-%d')
            next_date = date + timedelta(days=1)
            next_date_str = next_date.strftime('%Y-%m-%d')
            dict[next_date_str]=yhat[-1]
            dates.append(next_date_str)
            i+=1
        list(dict.values())
        
        # index = pd.date_range(start='2024-03-21', periods=len(dates), freq='D')
        # df = pd.DataFrame(list(dict.values()), index=index)
        # #df_hat=pd.DataFrame(yhat, index=index, columns=['Values'])
        # # Plot the time series data
        # plt.figure(figsize=(12, 6))
        # plt.plot(df, color='blue')
        # plt.plot(df_2G, color='red')
        # plt.title('2G')
        # plt.xlabel('Date')
        # plt.ylabel('Value')
        # plt.grid(True)
        # plt.show()
    

        # X,y=split_sequence(test, self.config.nsteps)
        # last_window=X[-1].reshape((1, n_seq, 1, n_steps, n_features))

        # prediction = self.model.predict(last_window, verbose=0)
        # yhat = scaler.inverse_transform(prediction)
        live_loss=self.check_loss()
        static_loss=self.get_model_metrics()
        print("=========================================Live__loss===========================")
        print(live_loss['mae'])
        print("==========================================static__loss===========================")
        print(static_loss['mae'])
        if live_loss['mae'] >= static_loss['mae']:
            print("===========================starting the online learning========================== ")
            self.trigger_retraining()
        else:
            subprocess.run(["dvc", "repro", "-s", "data_ingestion", "-f"], check=True)
        return dict

        
    



