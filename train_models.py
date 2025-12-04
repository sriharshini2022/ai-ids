#!/usr/bin/env python3
# Training Script for AI-based IDS

import pandas as pd
import numpy as np
from ai_ids_implementation import AIIntrusionDetectionSystem
import sys
import os

def download_sample_data():
    '''
    Download or generate sample data for training
    In practice, you would download real datasets like NSL-KDD or CICIDS2017
    '''
    print("📊 Generating sample training data...")

    # Generate synthetic network traffic data for demonstration
    np.random.seed(42)
    n_samples = 10000

    # Simulate network features
    data = {
        'duration': np.random.exponential(2, n_samples),
        'protocol_type': np.random.choice([0, 1, 2], n_samples),  # tcp, udp, icmp
        'service': np.random.choice(range(10), n_samples),
        'flag': np.random.choice(range(5), n_samples),
        'src_bytes': np.random.exponential(1000, n_samples),
        'dst_bytes': np.random.exponential(800, n_samples),
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.poisson(0.1, n_samples),
        'urgent': np.random.poisson(0.05, n_samples),
        'hot': np.random.poisson(0.1, n_samples),
        'num_failed_logins': np.random.poisson(0.05, n_samples),
        'logged_in': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'num_compromised': np.random.poisson(0.01, n_samples),
        'root_shell': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'su_attempted': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'num_root': np.random.poisson(0.05, n_samples),
        'num_file_creations': np.random.poisson(0.1, n_samples),
        'num_shells': np.random.poisson(0.05, n_samples),
        'num_access_files': np.random.poisson(0.1, n_samples),
        'num_outbound_cmds': np.random.poisson(0.01, n_samples),
        'is_host_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
        'serror_rate': np.random.beta(1, 10, n_samples),
        'srv_serror_rate': np.random.beta(1, 10, n_samples),
        'rerror_rate': np.random.beta(1, 20, n_samples),
        'srv_rerror_rate': np.random.beta(1, 20, n_samples),
        'same_srv_rate': np.random.beta(5, 2, n_samples),
        'diff_srv_rate': np.random.beta(1, 5, n_samples),
        'srv_diff_host_rate': np.random.beta(1, 10, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'dst_host_srv_count': np.random.poisson(8, n_samples),
        'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
        'dst_host_diff_srv_rate': np.random.beta(1, 5, n_samples),
        'dst_host_same_src_port_rate': np.random.beta(1, 10, n_samples),
        'dst_host_srv_diff_host_rate': np.random.beta(1, 10, n_samples),
        'dst_host_serror_rate': np.random.beta(1, 20, n_samples),
        'dst_host_srv_serror_rate': np.random.beta(1, 20, n_samples),
        'dst_host_rerror_rate': np.random.beta(1, 30, n_samples),
        'dst_host_srv_rerror_rate': np.random.beta(1, 30, n_samples),
    }

    # Create target labels
    # 0: normal, 1: dos, 2: probe, 3: r2l, 4: u2r
    labels = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])

    df = pd.DataFrame(data)
    df['label'] = labels

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_training_data.csv', index=False)

    print(f"✅ Generated {n_samples} samples and saved to data/sample_training_data.csv")
    return df

def main():
    print("🚀 Starting AI-IDS Training Pipeline")
    print("=" * 50)

    # Check if data exists
    if not os.path.exists('data/sample_training_data.csv'):
        print("📊 Training data not found. Generating sample data...")
        data = download_sample_data()
    else:
        print("📊 Loading existing training data...")
        data = pd.read_csv('data/sample_training_data.csv')

    print(f"📈 Data shape: {data.shape}")

    # Initialize IDS system
    print("🤖 Initializing AI-IDS system...")
    ids = AIIntrusionDetectionSystem()

    # Separate features and labels
    X = data.drop(['label'], axis=1)
    y = data['label']

    # Preprocess data
    print("🔄 Preprocessing data...")
    X_processed = ids.preprocess_data(X, is_training=True)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")

    # Train models
    print("🎯 Training Isolation Forest...")
    ids.train_isolation_forest(X_train[y_train == 0])  # Normal traffic only

    print("🎯 Training Random Forest...")
    ids.train_random_forest(X_train, y_train)

    print("🎯 Training XGBoost...")
    ids.train_xgboost(X_train, y_train)

    print("🎯 Building LSTM Autoencoder...")
    ids.build_lstm_autoencoder(X_train.shape[1])

    print("🎯 Training LSTM Autoencoder...")
    X_train_normal = X_train[y_train == 0]  # Normal traffic for autoencoder
    ids.train_lstm_autoencoder(X_train_normal.values, epochs=20)

    # Evaluate models
    print("📊 Evaluating models...")
    evaluation_results = ids.evaluate_models(X_test, y_test)

    print("\n📈 Evaluation Results:")
    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save models
    print("💾 Saving trained models...")
    ids.save_models()

    print("\n✅ Training completed successfully!")
    print("🚀 You can now run the dashboard: streamlit run ids_dashboard.py")

if __name__ == "__main__":
    main()
