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
from orangePlatform.entity.config_entity import ModelTrainerConfig


def create_lstm_model(num_lstm_units, learning_rate, nsteps):
    n_seq = 5
    n_steps =int(nsteps // n_seq)
    model = Sequential()
    model.add(ConvLSTM2D(filters=num_lstm_units, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, 1)))
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
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        train=train_data.values
        test=test_data.values
        print(type(test))
        model = create_lstm_model(self.config.num_lstm_units, self.config.learning_rate, self.config.nsteps)
        X, y = split_sequence(train, self.config.nsteps)  # You need to define split_sequence
        n_features = 1
        n_seq =5
        n_steps = int(self.config.nsteps // n_seq)
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, 1))
        X_val, y_val = split_sequence(test, self.config.nsteps)  # You need to define split_sequence
        X_val = X_val.reshape((X_val.shape[0], n_seq, 1, n_steps, 1))
                    # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.config.early_stopping_patience, restore_best_weights=True)
                    # Train the model and log metrics using MLflow
        history = model.fit(X, y, epochs=self.config.epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

