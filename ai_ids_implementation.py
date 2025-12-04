
# AI-Based Intrusion Detection System Implementation
# Based on the research abstract provided

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
import pickle
import logging
import warnings
warnings.filterwarnings('ignore')

class AIIntrusionDetectionSystem:
    '''
    AI-based Intrusion Detection System implementing multiple ML approaches:
    - Isolation Forest for anomaly detection
    - Random Forest and XGBoost for classification
    - LSTM Autoencoder for behavioral analysis
    '''

    def __init__(self):
        self.isolation_forest = None
        self.random_forest = None
        self.xgboost_model = None
        self.lstm_autoencoder = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ids_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_data(self, data, is_training=True):
        '''
        Comprehensive data preprocessing pipeline
        '''
        self.logger.info("Starting data preprocessing...")

        # Handle missing values
        data_cleaned = data.fillna(data.mean(numeric_only=True))

        # Feature engineering for network traffic data
        if 'duration' in data_cleaned.columns:
            data_cleaned['duration_log'] = np.log1p(data_cleaned['duration'])

        if 'src_bytes' in data_cleaned.columns and 'dst_bytes' in data_cleaned.columns:
            data_cleaned['total_bytes'] = data_cleaned['src_bytes'] + data_cleaned['dst_bytes']
            data_cleaned['byte_ratio'] = np.where(
                data_cleaned['dst_bytes'] > 0,
                data_cleaned['src_bytes'] / data_cleaned['dst_bytes'],
                0
            )

        # Encode categorical variables
        categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'label' and col != 'attack_type':  # Don't encode target variables
                if is_training:
                    data_cleaned[col] = self.label_encoder.fit_transform(data_cleaned[col].astype(str))
                else:
                    # Handle unseen categories
                    known_categories = set(self.label_encoder.classes_)
                    data_cleaned[col] = data_cleaned[col].map(
                        lambda x: x if x in known_categories else 'unknown'
                    )
                    data_cleaned[col] = self.label_encoder.transform(data_cleaned[col].astype(str))

        # Scale numerical features
        numerical_columns = data_cleaned.select_dtypes(include=[np.number]).columns
        if is_training:
            data_cleaned[numerical_columns] = self.scaler.fit_transform(data_cleaned[numerical_columns])
        else:
            data_cleaned[numerical_columns] = self.scaler.transform(data_cleaned[numerical_columns])

        self.logger.info(f"Preprocessing completed. Shape: {data_cleaned.shape}")
        return data_cleaned

    def train_isolation_forest(self, X_train, contamination=0.1):
        '''
        Train Isolation Forest for anomaly detection
        '''
        self.logger.info("Training Isolation Forest for anomaly detection...")

        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )

        # Train on normal traffic only (assuming label 0 is normal)
        normal_data = X_train  # Adjust based on your labeling
        self.isolation_forest.fit(normal_data)

        self.logger.info("Isolation Forest training completed")
        return self.isolation_forest

    def train_random_forest(self, X_train, y_train):
        '''
        Train Random Forest classifier
        '''
        self.logger.info("Training Random Forest classifier...")

        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        self.random_forest.fit(X_train, y_train)

        self.logger.info("Random Forest training completed")
        return self.random_forest

    def train_xgboost(self, X_train, y_train):
        '''
        Train XGBoost classifier
        '''
        self.logger.info("Training XGBoost classifier...")

        self.xgboost_model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        self.xgboost_model.fit(X_train, y_train)

        self.logger.info("XGBoost training completed")
        return self.xgboost_model

    def build_lstm_autoencoder(self, input_dim, encoding_dim=32):
        '''
        Build LSTM Autoencoder for behavioral analysis
        '''
        self.logger.info("Building LSTM Autoencoder...")

        # Encoder
        input_layer = Input(shape=(input_dim, 1))
        encoded = LSTM(encoding_dim, activation='relu')(input_layer)

        # Decoder
        decoded = RepeatVector(input_dim)(encoded)
        decoded = LSTM(1, activation='sigmoid', return_sequences=True)(decoded)

        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        self.lstm_autoencoder = autoencoder
        self.logger.info("LSTM Autoencoder built successfully")
        return autoencoder

    def train_lstm_autoencoder(self, X_train, epochs=50, batch_size=32):
        '''
        Train LSTM Autoencoder
        '''
        self.logger.info("Training LSTM Autoencoder...")

        # Reshape data for LSTM
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        history = self.lstm_autoencoder.fit(
            X_train_reshaped, X_train_reshaped,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )

        self.logger.info("LSTM Autoencoder training completed")
        return history

    def predict_anomalies(self, X_test):
        '''
        Detect anomalies using Isolation Forest
        '''
        anomaly_scores = self.isolation_forest.decision_function(X_test)
        anomaly_predictions = self.isolation_forest.predict(X_test)

        # Convert to binary (1 for normal, 0 for anomaly)
        anomaly_binary = np.where(anomaly_predictions == 1, 0, 1)

        return anomaly_binary, anomaly_scores

    def predict_attacks(self, X_test):
        '''
        Classify attack types using ensemble of RF and XGBoost
        '''
        rf_predictions = self.random_forest.predict(X_test)
        xgb_predictions = self.xgboost_model.predict(X_test)

        # Simple ensemble - majority voting
        ensemble_predictions = np.where(
            rf_predictions == xgb_predictions, 
            rf_predictions, 
            rf_predictions  # Default to RF in case of disagreement
        )

        # Get prediction probabilities for confidence scoring
        rf_proba = self.random_forest.predict_proba(X_test)
        xgb_proba = self.xgboost_model.predict_proba(X_test)

        # Average probabilities
        ensemble_proba = (rf_proba + xgb_proba) / 2

        return ensemble_predictions, ensemble_proba

    def detect_behavioral_anomalies(self, X_test, threshold=0.5):
        '''
        Detect behavioral anomalies using LSTM Autoencoder
        '''
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Get reconstruction error
        reconstructed = self.lstm_autoencoder.predict(X_test_reshaped)
        mse = np.mean(np.power(X_test_reshaped - reconstructed, 2), axis=1)

        # Classify as anomaly if reconstruction error > threshold
        behavioral_anomalies = np.where(mse > threshold, 1, 0)

        return behavioral_anomalies, mse

    def comprehensive_threat_assessment(self, X_test):
        '''
        Comprehensive threat assessment combining all models
        '''
        results = {}

        # Anomaly detection
        anomaly_pred, anomaly_scores = self.predict_anomalies(X_test)
        results['anomaly_detection'] = {
            'predictions': anomaly_pred,
            'scores': anomaly_scores
        }

        # Attack classification
        attack_pred, attack_proba = self.predict_attacks(X_test)
        results['attack_classification'] = {
            'predictions': attack_pred,
            'probabilities': attack_proba
        }

        # Behavioral analysis
        behavioral_pred, reconstruction_errors = self.detect_behavioral_anomalies(X_test)
        results['behavioral_analysis'] = {
            'predictions': behavioral_pred,
            'reconstruction_errors': reconstruction_errors
        }

        # Combined threat score
        threat_scores = (
            anomaly_scores * 0.3 +  # Weight for anomaly detection
            np.max(attack_proba, axis=1) * 0.4 +  # Weight for attack classification
            reconstruction_errors * 0.3  # Weight for behavioral analysis
        )

        results['combined_threat_score'] = threat_scores

        return results

    def evaluate_models(self, X_test, y_test):
        '''
        Evaluate all trained models
        '''
        evaluation_results = {}

        # Evaluate Random Forest
        rf_pred = self.random_forest.predict(X_test)
        evaluation_results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, average='weighted'),
            'recall': recall_score(y_test, rf_pred, average='weighted'),
            'f1_score': f1_score(y_test, rf_pred, average='weighted')
        }

        # Evaluate XGBoost
        xgb_pred = self.xgboost_model.predict(X_test)
        evaluation_results['xgboost'] = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, average='weighted'),
            'recall': recall_score(y_test, xgb_pred, average='weighted'),
            'f1_score': f1_score(y_test, xgb_pred, average='weighted')
        }

        return evaluation_results

    def save_models(self, model_dir='models/'):
        '''
        Save trained models
        '''
        import os
        os.makedirs(model_dir, exist_ok=True)

        # Save scikit-learn models
        with open(f'{model_dir}isolation_forest.pkl', 'wb') as f:
            pickle.dump(self.isolation_forest, f)

        with open(f'{model_dir}random_forest.pkl', 'wb') as f:
            pickle.dump(self.random_forest, f)

        with open(f'{model_dir}xgboost_model.pkl', 'wb') as f:
            pickle.dump(self.xgboost_model, f)

        # Save preprocessing objects
        with open(f'{model_dir}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(f'{model_dir}label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Save Keras model
        if self.lstm_autoencoder:
            self.lstm_autoencoder.save(f'{model_dir}lstm_autoencoder.h5')

        self.logger.info(f"Models saved to {model_dir}")

    def load_models(self, model_dir='models/'):
        '''
        Load trained models
        '''
        with open(f'{model_dir}isolation_forest.pkl', 'rb') as f:
            self.isolation_forest = pickle.load(f)

        with open(f'{model_dir}random_forest.pkl', 'rb') as f:
            self.random_forest = pickle.load(f)

        with open(f'{model_dir}xgboost_model.pkl', 'rb') as f:
            self.xgboost_model = pickle.load(f)

        with open(f'{model_dir}scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        with open(f'{model_dir}label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load Keras model
        from tensorflow.keras.models import load_model
        try:
            self.lstm_autoencoder = load_model(f'{model_dir}lstm_autoencoder.h5')
        except:
            self.logger.warning("LSTM Autoencoder model not found")

        self.logger.info(f"Models loaded from {model_dir}")


# Example usage and training pipeline
def main_training_pipeline():
    '''
    Main training pipeline demonstrating the IDS implementation
    '''
    # Initialize the IDS system
    ids = AIIntrusionDetectionSystem()

    # Note: In practice, you would load your actual dataset here
    # For demonstration, we'll show the structure

    print("AI-based Intrusion Detection System")
    print("="*50)
    print("Training pipeline ready for implementation")
    print()
    print("Steps to implement:")
    print("1. Load and preprocess network traffic data")
    print("2. Train Isolation Forest for anomaly detection")
    print("3. Train Random Forest and XGBoost for classification")
    print("4. Build and train LSTM Autoencoder for behavioral analysis")
    print("5. Evaluate model performance")
    print("6. Deploy models for real-time detection")

    return ids

if __name__ == "__main__":
    ids_system = main_training_pipeline()
