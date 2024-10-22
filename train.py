import os
from dotenv import load_dotenv 
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from tensorflow.keras.callbacks import EarlyStopping

# Load environment variables from .env file
load_dotenv()

def log_metric_to_grafana(metric_name, value):
    url = "http://localhost:3000/api/metrics"
    grafana_api_key = os.getenv("GRAFANA_API_KEY")  # Retrieve API key from the .env file
    headers = {
        'Authorization': f'Bearer {grafana_api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'metric': metric_name,
        'value': value
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print(f"Successfully logged {metric_name} to Grafana.")
    else:
        print(f"Failed to log {metric_name}. Status code: {response.status_code}, response: {response.text}")


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the parquet file
df_event2 = pd.read_parquet('combined_data_event2.parquet')
print("Data Loaded:")
print(df_event2.head())
print(f"Total rows: {len(df_event2)}\n")

# Drop completely empty columns
df_event2 = df_event2.dropna(axis=1, how='all')

# Forward and backward fill NaN class values per filename
df_event2['class'] = df_event2.groupby('filename')['class'].ffill()
df_event2['class'] = df_event2.groupby('filename')['class'].bfill()

# Fill NaN values using median for numeric columns
numeric_columns = df_event2.select_dtypes(include='number').columns.tolist()
for column in numeric_columns:
    df_event2[column] = df_event2.groupby('filename')[column].transform(lambda x: x.fillna(0) if x.isnull().all() else x.fillna(x.median()))

# Recode class labels to 0 and 1 (binary classification)
df_event2['class'] = df_event2['class'].replace({0: 0, 102: 1, 2: np.nan})
df_event2.dropna(subset=['class'], inplace=True)

# Split filenames for training and testing
unique_filenames = df_event2['filename'].unique()
train_filenames, test_filenames = train_test_split(unique_filenames, test_size=0.30)

# Prepare features and labels
df_train = df_event2[df_event2['filename'].isin(train_filenames)]
df_test = df_event2[df_event2['filename'].isin(test_filenames)]
X_train = df_train[numeric_columns].values.astype('float32')  # Ensure float32
y_train = df_train['class'].astype(int).values
X_test = df_test[numeric_columns].values.astype('float32')  # Ensure float32
y_test = df_test['class'].astype(int).values

# Reshape for RNN (GRU)
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Initialize MLflow run
if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run():
    # Define the GRU model
    rnn_model = Sequential()
    rnn_model.add(GRU(64, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
    rnn_model.add(Dropout(0.3))  # Dropout for regularization
    rnn_model.add(Dense(1, activation='sigmoid'))  # Binary classification

    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Set early stopping criteria
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=3,  
        restore_best_weights=True  
    )

    # Fit the model
    class_weight = {0: 1., 1: 1.}  
    print("Starting model training...")
    rnn_model.fit(
        X_train_rnn, 
        y_train, 
        validation_data=(X_test_rnn, y_test),  
        epochs=50,  
        batch_size=64, 
        verbose=1, 
        class_weight=class_weight, 
        callbacks=[early_stopping]  
    )
    print("Model training completed.")

    # Log the model in MLflow
    mlflow.keras.log_model(rnn_model, "rnn_model")

    # Evaluate the model on test data
    y_pred_rnn_prob = rnn_model.predict(X_test_rnn)
    y_pred_rnn = np.where(y_pred_rnn_prob > 0.5, 1, 0).astype(int).flatten()

    accuracy_rnn = accuracy_score(y_test, y_pred_rnn)
    f1_rnn = f1_score(y_test, y_pred_rnn, average='weighted')
    precision_rnn = precision_score(y_test, y_pred_rnn, average='weighted')

    print(f"\nGRU - Overall Accuracy: {accuracy_rnn}")
    print(f"GRU - Overall F1 Score: {f1_rnn}")

    # Log metrics to MLflow
    mlflow.log_metric("GRU_overall_accuracy", accuracy_rnn)
    mlflow.log_metric("GRU_overall_f1_score", f1_rnn)

    # Log metrics to Grafana
    log_metric_to_grafana("GRU_overall_accuracy", accuracy_rnn)
    log_metric_to_grafana("GRU_overall_f1_score", f1_rnn)

    # Evaluate GRU model on each filename separately
    all_time_lags = []
    filename_time_lags = []

    for filename in test_filenames:
        # Select the data for this filename
        df_filename = df_test[df_test['filename'] == filename]

        # Extract and reshape the features for the RNN model
        X_test_filename_rnn = df_filename[numeric_columns].values.astype('float32').reshape((df_filename.shape[0], 1, len(numeric_columns)))

        try:
            # Predict probabilities for this file
            y_pred_filename_rnn_prob = rnn_model.predict(X_test_filename_rnn)

            # Convert probabilities to class labels (0 or 1)
            y_pred_filename_rnn = np.where(y_pred_filename_rnn_prob > 0.5, 1, 0).astype(int).flatten()

            # Extract true labels for the file
            y_true_filename = df_filename['class'].values

            # Extract the timestamps for plotting
            timestamps = df_filename.index

            # Identify the first instance of class 102 in true and predicted values
            true_first_class_102 = df_filename[df_filename['class'] == 1].head(1).index
            pred_first_class_102 = df_filename.iloc[np.where(y_pred_filename_rnn == 1)].head(1).index

            # Print first transition to class 102
            if len(true_first_class_102) > 0 and len(pred_first_class_102) > 0:
                time_lag = (pred_first_class_102[0] - true_first_class_102[0]).total_seconds()
                all_time_lags.append(time_lag)
                filename_time_lags.append((filename, time_lag))

                # Log the time lag to MLflow
                mlflow.log_metric(f"time_lag_{filename}", time_lag)

                # Log the time lag to Grafana
                log_metric_to_grafana(f"time_lag_{filename}", time_lag)

            else:
                filename_time_lags.append((filename, float('inf')))
                print(f"No valid transitions found for filename: {filename}")

        except Exception as e:
            print(f"Error processing filename {filename}: {str(e)}")

    # Calculate average time lag
    avg_time_lag = np.mean(all_time_lags) if all_time_lags else float('inf')

    # Log time lag metrics to MLflow and Grafana
    mlflow.log_metric("GRU_avg_time_lag", avg_time_lag)
    log_metric_to_grafana("GRU_avg_time_lag", avg_time_lag)

    # Plot time lag graph
    filenames, time_lags = zip(*filename_time_lags)
    plt.figure(figsize=(10, 6))
    plt.bar(filenames, time_lags, color='skyblue')
    plt.xlabel('Filename')
    plt.ylabel('Time Lag (seconds)')
    plt.xticks(rotation=90)
    plt.title(f'Time Lag per Filename for GRU Model')
    plt.tight_layout()
    plt.show()
