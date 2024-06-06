import pandas as pd
import os
from orangePlatform import logger
import joblib
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, TimeDistributed
from keras.layers import ConvLSTM2D
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import yaml
from orangePlatform.entity.config_entity import Model_TrainerConfig


def create_lstm_model_3_layers(num_lstm_units, learning_rate, nsteps):
    n_seq = 5
    n_steps = int(nsteps // n_seq)
    model = Sequential()
    model.add(ConvLSTM2D(filters=num_lstm_units, kernel_size=(1, 2), activation='relu', padding='same', input_shape=(n_seq, 1, n_steps, 1), return_sequences=True))
    model.add(ConvLSTM2D(filters=num_lstm_units//2, kernel_size=(1, 2), activation='relu', padding='same', return_sequences=True))
    model.add(ConvLSTM2D(filters=num_lstm_units//4, kernel_size=(1, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model

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

class BestModelTrainer:
    def __init__(self, config: Model_TrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train=train_data.values
        test=test_data.values

        hyperparams = {
            'num_lstm_units': [256],
            'learning_rate': [0.001],
            'epochs': [100],
            'early_stopping_patience': [10],
            'nsteps': [10]
        }
        best_param={'num_lstm_units': 256, 'learning_rate': 0.001, 'epochs': 180, 'early_stopping_patience': 10, 'nsteps': 15}
        loss=   1e+20
        # Hyperparameter search loop
        for num_lstm_units in hyperparams['num_lstm_units']:
            for learning_rate in hyperparams['learning_rate']:
                for epochs in hyperparams['epochs']:
                    for early_stopping_patience in hyperparams['early_stopping_patience']:
                        for nsteps in hyperparams['nsteps']:
                        

                            # Create and train the LSTM model
                            model = create_lstm_model_3_layers(num_lstm_units, learning_rate, nsteps)
                            X, y = split_sequence(train, nsteps)  # You need to define split_sequence
                            n_features = 1
                            n_seq =5
                            n_steps = int(nsteps // n_seq)
                            X = X.reshape((X.shape[0], n_seq, 1, n_steps, 1))
                            X_val, y_val = split_sequence(test, nsteps)
                            X_val = X_val.reshape((X_val.shape[0], n_seq, 1, n_steps, 1))
                            # Define early stopping callback
                            early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
                            # Train the model and log metrics using MLflow
                            history = model.fit(X, y, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
                            if history.history['val_loss'][-1] < loss:
                                loss = history.history['val_loss'][-1]
                                best_param={'num_lstm_units': num_lstm_units, 'learning_rate': learning_rate, 'epochs': epochs, 'early_stopping_patience': early_stopping_patience, 'nsteps': nsteps}

        
           # Load the existing params.yaml file
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)

# Update the parameters in the params dictionary
        params['ConvLSTM2D'].update(best_param)

# Write the updated parameters back to the params.yaml file
        with open('params.yaml', 'w') as file:
            yaml.dump(params, file, default_flow_style=False)
                            


