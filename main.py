# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Initialize an empty list to store DataFrames
data_frames = []

# Walk through the home directory
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        # Construct the full path of the file
        file_path = os.path.join(dirname, filename)
        
        # Check if the file is a CSV file (you can modify this condition based on your file format)
        if file_path.endswith('.csv'):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Append the DataFrame to the list
            data_frames.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Display the final DataFrame
print(data)


# %%
data.info()

# %%
import pandas as pd

# Assuming data is your DataFrame
# Display the column names to verify their correctness
print("Column Names:", data.columns)

# Convert the 'arrival_time' and 'start_time' columns to datetime with specified format
data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%d-%m-%Y %H.%M')
data['start_time'] = pd.to_datetime(data['start_time'], format='%d-%m-%Y %H.%M')

# Calculate the waiting time as the difference between 'start_time' and 'arrival_time'
data['waiting_time'] = (data['start_time'] - data['arrival_time']).dt.total_seconds()

# Verify the results
print(data[['arrival_time', 'start_time', 'waiting_time']].head())

# %%
# Convert 'wait_time' to numeric, coercing errors to NaN
df['waiting_time'] = pd.to_numeric(df['wait_time'], errors='coerce')

# Drop rows with NaN values in 'waiting_time'
df = df.dropna(subset=['waiting_time'])

# %%
# Convert 'waiting_time' to numeric, coercing errors to NaN
df['waiting_time'] = pd.to_numeric(df['waiting_time'], errors='coerce')

# Drop rows with NaN values in 'waiting_time'
df = df.dropna(subset=['waiting_time'])

# %% [markdown]
# * **Time Series Analysis:**
# 1. *Plot the queue length over time to identify patterns or trends.*
# 1. *Analyze the waiting time variation throughout the day.*

# %%
# Convert 'finish_time' to datetime
df['finish_time'] = pd.to_datetime(df['finish_time'])

# Plot queue length over time
plt.figure(figsize=(12, 6))
plt.plot(df['finish_time'], df['queue_length'], marker='o', linestyle='-')
plt.title('Queue Length Over Time')
plt.xlabel('Finish Time')
plt.ylabel('Queue Length')
plt.show()

# Analyze waiting time variation throughout the day
df['hour'] = df['finish_time'].dt.hour
sns.boxplot(x='hour', y='waiting_time', data=df)
plt.title('Waiting Time Variation Throughout the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Waiting Time (seconds)')
plt.show()

# %% [markdown]
# * **Statistical Summary:**
# 
# 1. *Generate descriptive statistics for numeric columns.*

# %%
# Descriptive statistics
stats = df.describe()
print(stats)

# %% [markdown]
# * **Waiting Time Distribution by Queue Length:**
# 
# 1. *Visualize the distribution of waiting time for different queue lengths.*

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['queue_length'], y=df['waiting_time'])
plt.title('Boxplot of Waiting Time by Queue Length')
plt.xlabel('Queue Length')
plt.ylabel('Waiting Time (seconds)')
plt.show()

# %% [markdown]
# * **Correlation Analysis:**
# 
# 1. *Explore correlations between different numeric columns.*

# %%
# Check the data types of each column
print(df.dtypes)

# Convert non-numeric columns to numeric, coercing errors to NaN
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Specify the format for 'arrival_time' and 'start_time'
df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%d-%m-%Y %H.%M')
df['start_time'] = pd.to_datetime(df['start_time'], format='%d-%m-%Y %H.%M')

# Correlation matrix for numeric columns
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap for Numeric Columns")
plt.show()

# %%
# Specify the correlation threshold
correlation_threshold = 0.5  # You can adjust this value based on your preference

# Filter the correlation matrix
filtered_correlation_matrix = correlation_matrix[abs(correlation_matrix) > correlation_threshold]

# Plot the filtered correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(filtered_correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title(f"Filtered Correlation Matrix (Threshold = {correlation_threshold})")
plt.show()

# %%
# Specify a high correlation threshold
high_correlation_threshold = 0.9  # You can adjust this value based on your preference

# Find highly correlated pairs
highly_correlated_pairs = (abs(correlation_matrix) > high_correlation_threshold) & (correlation_matrix < 1.0)

# Display highly correlated pairs
print("Highly Correlated Variable Pairs:")
print(highly_correlated_pairs)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %%
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# %%
# Select relevant columns for the VAR model
columns_for_var = ['arrival_time', 'start_time', 'finish_time', 'wait_time', 'queue_length', 'waiting_time', 'hour']
df_var = df[columns_for_var]

# %%
# Train-test split
train_size = int(len(df_var) * 0.8)
train, test = df_var.iloc[:train_size, :], df_var.iloc[train_size:, :]
print(train.dtypes)

# %%
print(train.isnull().sum())


# %%
print(train.applymap(np.isreal))


# %%
def datetime_transform(X):
    return X.values.reshape(-1, 1)  # Convert datetime to a 2D array

# %%
train_array = np.asarray(train.dropna())  # Remove rows with missing values

# %%
from sklearn.preprocessing import FunctionTransformer
import numpy as np

def datetime_transform(X):
    return X.values.reshape(-1, 1)  # Convert datetime to a 2D array

datetime_transformer = FunctionTransformer(datetime_transform, validate=False)

# %%
def create_dataset(dataset, look_back=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        data_X.append(a)
        data_Y.append(dataset[i + look_back, 0])
    return np.array(data_X), np.array(data_Y)

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'train' is your training set
train_features = train[['wait_time']].values.astype(float)

# Apply Min-Max scaling to the training data
scaler = MinMaxScaler(feature_range=(0, 1))
train_features_scaled = scaler.fit_transform(train_features)

# Create the training dataset with look back
look_back = 1
train_X, train_y = create_dataset(train_features_scaled, look_back)

# Reshape the input data to be in the form [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

# Train the model
history = model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2, callbacks=[early_stopping])

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Assuming 'test' is your test set
test_features = test[['wait_time']].values.astype(float)

# Apply Min-Max scaling to the test data
test_features_scaled = scaler.transform(test_features)

# Create the test dataset with look back
test_X, test_y = create_dataset(test_features_scaled, look_back)

# Reshape the input data to be in the form [samples, time steps, features]
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Make predictions on the test set
test_predict = model.predict(test_X)

# Inverse transform the predictions to the original scale
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# Calculate MSE
mse = mean_squared_error(test_y.flatten(), test_predict.flatten())
print(f'Mean Squared Error (MSE) on Test Data: {mse}')

# Plot the results
plt.plot(test_y.flatten(), label='Actual Wait Time (Test)')
plt.plot(test_predict.flatten(), label='Predicted Wait Time (Test)')
plt.legend()
plt.show()

# %%
# Assuming 'test' is your test set
test_features = test[['wait_time']].values.astype(float)
test_features_scaled = scaler.transform(test_features)

# Create the test dataset with look back
test_X, test_y = create_dataset(test_features_scaled, look_back)

# Reshape the input data to be in the form [samples, time steps, features]
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Make predictions on the test set
test_predict = model.predict(test_X)

# Inverse transform the predictions to the original scale
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# Plot the results
plt.plot(test_y.flatten(), label='Actual Wait Time (Test)')
plt.plot(test_predict.flatten(), label='Predicted Wait Time (Test)')
plt.legend()
plt.show()

# %%
# Assuming 'test' is your test set
test_features = test[['wait_time']].values.astype(float)
test_features_scaled = scaler.transform(test_features)

# Create the test dataset with look back
test_X, test_y = create_dataset(test_features_scaled, look_back)

# Reshape the input data to be in the form [samples, time steps, features]
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Make predictions on the test set
test_predict = model.predict(test_X)

# Inverse transform the predictions to the original scale
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

# Print the actual and predicted values
print("Actual Wait Time (Test):", test_y.flatten())
print("Predicted Wait Time (Test):", test_predict.flatten())

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'df' with a datetime index
df['timestamp'] = pd.to_datetime(df['finish_time'])
df.set_index('timestamp', inplace=True)

# Feature scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['wait_time']])

# Create sequences and labels
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define sequence length
sequence_length = 10

# Create sequences and labels
X, y = create_sequences(df_scaled, sequence_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, model_checkpoint], verbose=2)

# Load the best model
best_model = load_model('best_model.h5')

# Make predictions
y_pred = best_model.predict(X_test)

# Inverse transform the scaled predictions and labels
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('LSTM Model Prediction')
plt.xlabel('Time')
plt.ylabel('Wait Time')
plt.legend()
plt.show()

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%



