import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df_combined = None
        self.df_prices_only = None
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            # Load HAI data
            df_hai = pd.read_csv(self.config.HAI_DATA_PATH)
            print(f"HAI data loaded successfully: {df_hai.shape}")
            
            # Load prices data
            df_prices = pd.read_csv(self.config.PRICES_DATA_PATH)
            print(f"Prices data loaded successfully: {df_prices.shape}")
            
            # Store both dataframes
            self.df_combined = df_prices
            self.df_prices_only = df_prices  # Set this for predictor to use
            
            return self.df_combined
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def prepare_data(self):
        """Prepare data for training"""
        # Extract price columns and create sequences
        price_data = self.df_combined[self.config.PRICE_COLUMNS].values
        
        # Create sequences for training
        X, y = [], []
        for i in range(len(price_data)):
            row = price_data[i]
            for j in range(len(row) - self.config.INPUT_LENGTH - self.config.OUTPUT_LENGTH + 1):
                X.append(row[j:j + self.config.INPUT_LENGTH])
                y.append(row[j + self.config.INPUT_LENGTH:j + self.config.INPUT_LENGTH + self.config.OUTPUT_LENGTH])
        
        X = np.array(X).reshape(-1, self.config.INPUT_LENGTH, 1)
        y = np.array(y).reshape(-1, self.config.OUTPUT_LENGTH)
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
        
        # Split into train/val/test
        train_size = int(len(X_scaled) * self.config.TRAIN_SPLIT)
        val_size = int(len(X_scaled) * self.config.VAL_SPLIT)
        
        X_train = torch.FloatTensor(X_scaled[:train_size])
        y_train = torch.FloatTensor(y_scaled[:train_size])
        
        X_val = torch.FloatTensor(X_scaled[train_size:val_size])
        y_val = torch.FloatTensor(y_scaled[train_size:val_size])
        
        X_test = torch.FloatTensor(X_scaled[val_size:])
        y_test = torch.FloatTensor(y_scaled[val_size:])
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }