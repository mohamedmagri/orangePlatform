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




def load_scaler_from_artifact(artifact_folder):
    # Get a list of all files in the artifact folder
    files = os.listdir(artifact_folder)

    # Filter CSV files
    joblib_files = [file for file in files if file.endswith('.joblib')]

    # Sort the CSV files to ensure consistent order
    joblib_files.sort()

    # Check if there are any CSV files in the folder
    if joblib_files:
        # Load the first CSV file
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

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def check_loss(self):
        test_data = pd.read_csv(self.config.test_data_path)
        test=test_data.values
        n_features = 1
        n_seq =5
        n_steps = int(self.config.nsteps // n_seq)
        X,y=split_sequence(test, self.config.nsteps)
        
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
        yhat = self.model.predict(X, verbose=0)
        (rmse, mae, r2) = self.eval_metrics(y, yhat)
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        return scores

    

    def get_model_metrics(self):
        # Load the metrics.json file
        with open('metrics.json', 'r') as file:
            metrics = json.load(file)

        # Extract the metrics
        rmse = metrics['rmse']
        mae = metrics['mae']
        r2 = metrics['r2']

        # Return the metrics as a dictionary
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
  
    def trigger_retraining(self):
        if not self.retraining:
            self.retraining = True
            threading.Thread(target=self.retrain_pipeline).start()

    def retrain_pipeline(self):
        subprocess.run(["dvc", "repro"], check=True)
        self.model = self.load_model()
        self.retraining = False
        
    
    def predict(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        test=test_data.values
        n_features = 1
        n_seq =5
        n_steps = int(self.config.nsteps // n_seq)

        X,y=split_sequence(test, self.config.nsteps)
        last_window=X[-1].reshape((1, n_seq, 1, n_steps, n_features))

        prediction = self.model.predict(last_window, verbose=0)
        yhat = scaler.inverse_transform(prediction)
        live_loss=self.check_loss(test)
        static_loss=self.get_model_metrics()
        if live_loss['mae'] >= static_loss['mae']:
            self.trigger_retraining()
        
        return yhat

        
    



