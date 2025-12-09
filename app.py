import logging
import sys
import traceback
from typing import Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class DeviceManager:
    """Manages PyTorch device compatibility (CPU/GPU)"""
    
    @staticmethod
    def get_device() -> torch.device:
        """
        Determine available device (GPU or CPU)
        
        Returns:
            torch.device: Available device
        """
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                return device
            else:
                device = torch.device('cpu')
                logger.info("GPU not available, using CPU")
                return device
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}. Falling back to CPU")
            return torch.device('cpu')

class DataValidator:
    """Validates input data for model processing"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> Tuple[bool, str]:
        """
        Validate DataFrame structure and content
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, message)
        """
        if df is None:
            return False, "DataFrame is None"
        
        if not isinstance(df, pd.DataFrame):
            return False, "Input is not a pandas DataFrame"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        if df.isnull().all().any():
            return False, "One or more columns are entirely null"
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
        
        return True, "DataFrame validation passed"
    
    @staticmethod
    def validate_numeric_data(data: np.ndarray) -> Tuple[bool, str]:
        """
        Validate numeric data for model input
        
        Args:
            data: Numeric array to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not isinstance(data, (np.ndarray, list)):
            return False, "Data must be numpy array or list"
        
        data = np.asarray(data)
        
        if data.size == 0:
            return False, "Data array is empty"
        
        if np.any(np.isnan(data)):
            return False, "Data contains NaN values"
        
        if np.any(np.isinf(data)):
            return False, "Data contains infinite values"
        
        return True, "Numeric data validation passed"
    
    @staticmethod
    def validate_sequence_length(sequence_length: int, min_length: int = 5) -> Tuple[bool, str]:
        """
        Validate sequence length for LSTM
        
        Args:
            sequence_length: Sequence length to validate
            min_length: Minimum required length
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not isinstance(sequence_length, int):
            return False, "Sequence length must be an integer"
        
        if sequence_length < min_length:
            return False, f"Sequence length must be at least {min_length}"
        
        return True, "Sequence length validation passed"

class LSTMModel(nn.Module):
    """Enhanced LSTM model with device compatibility"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int,
                 dropout: float = 0.2, device: Optional[torch.device] = None):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
            device: Device to place model on
        """
        super(LSTMModel, self).__init__()
        
        if device is None:
            device = DeviceManager.get_device()
        
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        try:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, output_size)
            self.to(self.device)
            logger.info(f"LSTM model initialized on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing LSTM model: {e}")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        try:
            x = x.to(self.device)
            lstm_out, (hidden, cell) = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

class DataPreprocessor:
    """Handles data preprocessing with validation"""
    
    def __init__(self, sequence_length: int = 30, device: Optional[torch.device] = None):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Length of sequences for LSTM
            device: PyTorch device
        """
        self.sequence_length = sequence_length
        self.device = device or DeviceManager.get_device()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Validate sequence length
        is_valid, msg = DataValidator.validate_sequence_length(sequence_length)
        if not is_valid:
            raise ValueError(msg)
    
    def preprocess(self, data: Union[np.ndarray, pd.Series], fit_scaler: bool = True) -> np.ndarray:
        """
        Preprocess data with validation and scaling
        
        Args:
            data: Input data to preprocess
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Preprocessed data
        """
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.Series):
                data = data.values.reshape(-1, 1)
            elif isinstance(data, (list, tuple)):
                data = np.asarray(data).reshape(-1, 1)
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")
            
            # Validate numeric data
            is_valid, msg = DataValidator.validate_numeric_data(data)
            if not is_valid:
                raise ValueError(msg)
            
            # Scale data
            if fit_scaler:
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)
            
            logger.info(f"Data preprocessed successfully. Shape: {scaled_data.shape}")
            return scaled_data
        
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            raise
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Preprocessed data
            
        Returns:
            Tuple of (X, y) sequences
        """
        try:
            if len(data) < self.sequence_length + 1:
                raise ValueError(
                    f"Data length ({len(data)}) must be > sequence_length ({self.sequence_length})"
                )
            
            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:i + self.sequence_length])
                y.append(data[i + self.sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
        
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            raise
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale
        
        Args:
            scaled_data: Scaled data
            
        Returns:
            Data in original scale
        """
        try:
            return self.scaler.inverse_transform(scaled_data)
        except Exception as e:
            logger.error(f"Error in inverse transform: {e}")
            raise

class ModelTrainer:
    """Handles model training with error handling"""
    
    def __init__(self, model: LSTMModel, device: Optional[torch.device] = None,
                 learning_rate: float = 0.001):
        """
        Initialize trainer
        
        Args:
            model: LSTM model to train
            device: PyTorch device
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device or DeviceManager.get_device()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info(f"Trainer initialized on {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        try:
            self.model.train()
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            return avg_loss
        
        except Exception as e:
            logger.error(f"Error during training epoch: {e}")
            raise
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation data
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        try:
            self.model.eval()
            total_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device).float()
                    batch_y = batch_y.to(self.device).float()
                    
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(val_loader)
            return avg_loss
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        try:
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor)
            
            return predictions.cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

# Main application logic for Google Colab
def main_colab():
    """Main application logic adapted for Google Colab"""
    try:
        print("TRIFUSION Forecasting App with DOSM Data")
        print("Enhanced with improved error handling, data validation, and PyTorch device compatibility\n")
        
        # Initialize device
        device = DeviceManager.get_device()
        print(f"Using device: {device}")
        
        # Configuration inputs
        sequence_length = int(input("Enter sequence length (default 30): ") or 30)
        hidden_size = int(input("Enter hidden size (default 64): ") or 64)
        num_layers = int(input("Enter number of LSTM layers (default 2): ") or 2)
        epochs = int(input("Enter number of epochs (default 50): ") or 50)
        batch_size = int(input("Enter batch size (default 32): ") or 32)
        learning_rate = float(input("Enter learning rate (default 0.001): ") or 0.001)
        
        # File upload simulation (in Colab, upload via files.upload())
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded. Please upload a CSV file.")
            return
        
        file_name = list(uploaded.keys())[0]
        df = pd.read_csv(file_name)
        
        # Validate DataFrame
        is_valid, msg = DataValidator.validate_dataframe(df)
        if not is_valid:
            print(f"Data validation failed: {msg}")
            return
        
        print("✓ Data uploaded and validated successfully")
        print(f"Data shape: {df.shape}")
        print(df.head())
        
        # Select target column
        print("\nAvailable columns:", list(df.columns))
        target_column = input("Enter target column for forecasting: ")
        if target_column not in df.columns:
            print("Invalid column selected.")
            return
        
        # Preprocess data
        try:
            preprocessor = DataPreprocessor(sequence_length=sequence_length, device=device)
            processed_data = preprocessor.preprocess(df[target_column].values)
            X, y = preprocessor.create_sequences(processed_data)
            
            print(f"✓ Data preprocessed. Sequences created: X={X.shape}, y={y.shape}")
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            print(f"✓ Data split: Train={len(X_train)}, Validation={len(X_val)}")
            
            # Initialize model and trainer
            model = LSTMModel(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                device=device
            )
            trainer = ModelTrainer(model, device=device, learning_rate=learning_rate)
            
            # Training
            print("\nModel Training")
            train_losses, val_losses = [], []
            
            for epoch in range(epochs):
                train_loss = trainer.train_epoch(train_loader)
                val_loss = trainer.evaluate(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Plot losses
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
            plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Losses')
            plt.show()
            
            print("✓ Model training completed")
            
            # Make predictions
            print("\nPredictions")
            predictions_train = trainer.predict(X_train)
            predictions_val = trainer.predict(X_val)
            
            # Inverse transform predictions
            predictions_train_original = preprocessor.inverse_transform(predictions_train)
            predictions_val_original = preprocessor.inverse_transform(predictions_val)
            y_train_original = preprocessor.inverse_transform(y_train)
            y_val_original = preprocessor.inverse_transform(y_val)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train_original, predictions_train_original)
            train_mae = mean_absolute_error(y_train_original, predictions_train_original)
            train_r2 = r2_score(y_train_original, predictions_train_original)
            
            val_mse = mean_squared_error(y_val_original, predictions_val_original)
            val_mae = mean_absolute_error(y_val_original, predictions_val_original)
            val_r2 = r2_score(y_val_original, predictions_val_original)
            
            # Display metrics
            print("\nTraining Metrics:")
            print(f"MSE: {train_mse:.6f}")
            print(f"MAE: {train_mae:.6f}")
            print(f"R² Score: {train_r2:.6f}")
            
            print("\nValidation Metrics:")
            print(f"MSE: {val_mse:.6f}")
            print(f"MAE: {val_mae:.6f}")
            print(f"R² Score: {val_r2:.6f}")
            
            # Plot predictions vs actual
            plt.figure(figsize=(12, 6))
            plt.plot(np.concatenate([y_train_original.flatten(), y_val_original.flatten()]), label="Actual")
            plt.plot(np.concatenate([predictions_train_original.flatten(), predictions_val_original.flatten()]), label="Predicted")
            plt.axvline(len(y_train_original), color='r', linestyle='--', label='Train/Val Split')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Predictions vs Actual')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error during model training: {e}\n{traceback.format_exc()}")
            print(f"Error during model training: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}\n{traceback.format_exc()}")
        print(f"An unexpected error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    main_colab()
