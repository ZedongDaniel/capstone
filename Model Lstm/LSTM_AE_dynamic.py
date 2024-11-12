import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import requests
import tensorflow as tf
import seaborn as sns
import itertools
import random

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix

from tensorflow.keras.models import Model, Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, BatchNormalization, Input, LayerNormalization # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import mae # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore


class LSTMAutoencoderAnomalyDetector_Dynamic:
    def __init__(self, 
                 sequence_length=15, 
                 n_layers=1, 
                 units=64, 
                 activation='tanh', 
                 dropout_rate=0.0,
                 learning_rate=0.001,
                 batch_size=32,
                 epochs=100,
                 patience=10,
                 threshold_percentile=95,
                 volatility_window=30,
                 volatility_low_percentile=33,
                 volatility_high_percentile=66,
                 scaler=None):
        self.sequence_length = sequence_length
        self.n_layers = n_layers
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold_percentile = threshold_percentile
        self.volatility_window = volatility_window
        self.volatility_low_percentile = volatility_low_percentile
        self.volatility_high_percentile = volatility_high_percentile
        self.scaler = scaler if scaler else StandardScaler()
        self.model = None
        self.history = None
        self.thresholds_per_feature = None
        self.train_data = None
        self.test_data = None
        self.train_size = None
        self.n_features = None
        self.X_train = None
        self.X_test = None
        self.X_train_volatility = None
        self.X_test_volatility = None
        self.anomalies = None

    def preprocess_data(self, data, train_size=0.8):
        self.train_size = int(len(data) * train_size)
        self.train_data = data.iloc[:self.train_size]
        self.test_data = data.iloc[self.train_size:]

        self.train_scaled = self.scaler.fit_transform(self.train_data)
        self.test_scaled = self.scaler.transform(self.test_data)

        train_volatility = pd.DataFrame(self.train_scaled, index=self.train_data.index, columns=self.train_data.columns).rolling(window=self.volatility_window).std().dropna()
        test_volatility = pd.DataFrame(self.test_scaled, index=self.test_data.index, columns=self.test_data.columns).rolling(window=self.volatility_window).std().dropna()

        self.train_scaled = self.train_scaled[self.volatility_window - 1:self.train_size]
        self.test_scaled = self.test_scaled[self.volatility_window - 1:]

        self.X_train = self._create_sequences(self.train_scaled)
        self.X_test = self._create_sequences(self.test_scaled)
        self.X_train_volatility = self._create_sequences(train_volatility.values)
        self.X_test_volatility = self._create_sequences(test_volatility.values)

        self.n_features = self.X_train.shape[2]
        print(f'Data preprocessed: {self.X_train.shape[0]} training sequences, {self.X_test.shape[0]} test sequences.')

    def _create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def build_model(self):
        inputs = Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        # Encoder
        for i in range(self.n_layers):
            return_sequences = (i < self.n_layers - 1)
            x = LSTM(self.units, activation=self.activation, return_sequences=return_sequences)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        encoded = x
        # Decoder
        x = RepeatVector(self.sequence_length)(encoded)
        for i in range(self.n_layers):
            x = LSTM(self.units, activation=self.activation, return_sequences=True)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        outputs = TimeDistributed(Dense(self.n_features))(x)
        self.model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mae')
        print('Model built successfully.')

    def train_model(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(self.patience / 2), min_lr=1e-6)
        self.history = self.model.fit(
            self.X_train, self.X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            shuffle=False,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        print('Model trained successfully.')

    def evaluate_model(self):
        X_train_pred = self.model.predict(self.X_train)
        train_errors = np.mean(np.abs(self.X_train - X_train_pred), axis=1)

        train_volatilities = np.mean(self.X_train_volatility, axis=1)

        low_percentile = self.volatility_low_percentile
        high_percentile = self.volatility_high_percentile
        low_vol_thresholds = np.percentile(train_volatilities, low_percentile, axis=0)
        high_vol_thresholds = np.percentile(train_volatilities, high_percentile, axis=0)

        self.thresholds_per_feature = {}
        for feature in range(self.n_features):
            low_thresh = low_vol_thresholds[feature]
            high_thresh = high_vol_thresholds[feature]

            volatilities = train_volatilities[:, feature]
            regimes = np.where(volatilities <= low_thresh, 'low',
                      np.where(volatilities <= high_thresh, 'medium', 'high'))

            self.thresholds_per_feature[feature] = {}
            for regime in ['low', 'medium', 'high']:
                indices = regimes == regime
                if np.sum(indices) == 0:
                    self.thresholds_per_feature[feature][regime] = None
                else:
                    errors_in_regime = train_errors[indices, feature]
                    self.thresholds_per_feature[feature][regime] = np.percentile(errors_in_regime, self.threshold_percentile)

        X_test_pred = self.model.predict(self.X_test)
        test_errors = np.mean(np.abs(self.X_test - X_test_pred), axis=1)

        test_volatilities = np.mean(self.X_test_volatility, axis=1)

        self.anomalies = np.zeros(test_errors.shape, dtype=bool)
        for feature in range(self.n_features):
            low_thresh = low_vol_thresholds[feature]
            high_thresh = high_vol_thresholds[feature]

            volatilities = test_volatilities[:, feature]
            regimes = np.where(volatilities <= low_thresh, 'low',
                      np.where(volatilities <= high_thresh, 'medium', 'high'))

            for regime in ['low', 'medium', 'high']:
                indices = regimes == regime
                threshold = self.thresholds_per_feature[feature][regime]
                if threshold is not None:
                    self.anomalies[indices, feature] = test_errors[indices, feature] > threshold

        self.X_test_pred = X_test_pred
        print('Model evaluation with dynamic thresholds completed.')

    def plot_train_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def get_anomaly_dataframe(self):
        num_test_sequences = self.X_test.shape[0]
        sequence_indices = np.arange(num_test_sequences)
        
        start_index = (self.volatility_window - 1) + self.sequence_length - 1
        test_dates = self.test_data.index[start_index:]
        
        if len(test_dates) != num_test_sequences:
            print(f"Adjusted test_dates length: {len(test_dates)}")
            print(f"Number of test sequences: {num_test_sequences}")
            raise ValueError("Mismatch between the number of test dates and the number of test sequences.")
        
        anomaly_df = pd.DataFrame(self.anomalies, index=test_dates, columns=self.train_data.columns)
        anomaly_df = anomaly_df.astype(int)
        return anomaly_df

    def plot_anomalies(self):
        anomaly_df = self.get_anomaly_dataframe()
        num_features = self.n_features
        ncols = 3
        nrows = int(np.ceil(num_features / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
        fig.tight_layout(pad=3.0)
        fig.subplots_adjust(top=0.92)
        axes = axes.flatten()

        for i, feature in enumerate(self.train_data.columns):
            ax = axes[i]

            feature_test_data = self.test_data[feature].iloc[(self.volatility_window - 1) + self.sequence_length:]
            feature_test_dates = feature_test_data.index
            ax.plot(feature_test_dates, feature_test_data.values, label='Test Data')

            anomalies = anomaly_df[feature]
            anomaly_dates = anomalies[anomalies == 1].index
            anomaly_values = self.test_data.loc[anomaly_dates, feature]
            ax.scatter(anomaly_dates, anomaly_values, color='red', label='Anomalies')
            ax.set_title(f'Feature: {feature}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()

        if len(axes) > num_features:
            for j in range(num_features, len(axes)):
                fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def plot_random_sequence(self):
        import random
        num_sequences_to_plot = 1
        random_indices = random.sample(range(self.X_test.shape[0]), num_sequences_to_plot)

        for idx in random_indices:
            plt.figure(figsize=(15, 5))
            for feature in range(self.X_test.shape[2]):
                plt.plot(self.X_test[idx, :, feature], label=f'Actual Feature {feature}')
                plt.plot(self.X_test_pred[idx, :, feature], linestyle='--', label=f'Reconstructed Feature {feature}')
            plt.title(f'Sequence {idx} - Actual vs. Reconstructed')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

    def save_model(self, file_path):
        """
        Save the trained model to the specified file path.
        
        Parameters:
        - file_path: str
            The file path where the model will be saved.
        """
        from tensorflow.keras.models import save_model # type: ignore
        self.model.save(file_path)
        print(f'Model saved to {file_path}.')

    def load_model(self, file_path):
        """
        Load a model from the specified file path.
        
        Parameters:
        - file_path: str
            The file path from which the model will be loaded.
        """
        from tensorflow.keras.models import load_model # type: ignore
        self.model = load_model(file_path)
        print(f'Model loaded from {file_path}.')