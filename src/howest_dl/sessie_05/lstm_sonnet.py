import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


# Step 1: Create a sample DataFrame (replace with your actual data)
def create_sample_data(n_samples=1000, n_features=5):
    np.random.seed(42)

    # Create cycle variable (time indicator)
    cycles = np.arange(n_samples)

    # Create features with some time dependency
    data = {f'feature_{i}': np.sin(0.1 * cycles + i) + np.random.normal(0, 0.2, n_samples)
            for i in range(n_features)}

    # Add cycle column
    data['cycle'] = cycles

    # Create target variable with dependency on features and time
    target = np.zeros(n_samples)
    for i in range(n_samples):
        if i > 10:  # Some time dependency
            target[i] = (0.5 * data['feature_0'][i] +
                         0.3 * data['feature_1'][i] +
                         0.2 * data['feature_2'][i] +
                         0.1 * target[i - 1] +  # Autoregressive component
                         0.05 * (cycles[i] % 10))  # Cycle dependency

    # Add some noise to target
    data['target'] = target + np.random.normal(0, 0.1, n_samples)

    return pd.DataFrame(data)


# # Create sample DataFrame
# df = create_sample_data(n_samples=1000, n_features=5)
# print("Sample DataFrame:")
# print(df.head())


# Step 2: Create windowed dataset function
def create_windowed_dataset(dataframe, window_size, batch_size=32,
                            target_column='target', cycle_column='cycle',
                            test_size=0.2, shuffle=False, scale_data=True):
    """
    Create a windowed dataset from a pandas DataFrame for LSTM training.

    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input DataFrame containing features, target, and cycle columns
    window_size : int
        The number of time steps to include in each window
    batch_size : int
        Batch size for the dataset
    target_column : str
        Name of the target column in the DataFrame
    cycle_column : str
        Name of the cycle/time column in the DataFrame
    test_size : float
        Proportion of data to use for testing
    shuffle : bool
        Whether to shuffle the training data
    scale_data : bool
        Whether to scale the features

    Returns:
    --------
    X_train, X_test : numpy arrays
        Training and test feature windows with shape (samples, window_size, features)
    y_train, y_test : numpy arrays
        Training and test targets
    feature_scaler, target_scaler : sklearn scalers
        Fitted scalers for features and target (for inverse transformation)
    """
    # Sort by cycle to ensure time order
    df_sorted = dataframe.sort_values(by=cycle_column).reset_index(drop=True)

    # Separate features and target
    features = df_sorted.drop([target_column, cycle_column], axis=1)
    target = df_sorted[target_column].values.reshape(-1, 1)

    # Scale features and target if requested
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    if scale_data:
        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(target)
    else:
        features_scaled = features.values
        target_scaled = target

    # Create windows
    X = []
    y = []

    for i in range(len(df_sorted) - window_size):
        # Check if this window spans a continuous cycle sequence
        cycles_in_window = df_sorted[cycle_column].iloc[i:i + window_size].values
        if np.max(np.diff(cycles_in_window)) > 1:
            # Skip this window if cycles are not consecutive
            continue

        # Add window to dataset
        X.append(features_scaled[i:i + window_size])
        y.append(target_scaled[i + window_size])  # Predict the next value after the window

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=42)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Batch and prefetch for performance
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (X_train, X_test, y_train, y_test,
            train_dataset, test_dataset,
            feature_scaler, target_scaler)


# # Step 3: Create the windowed dataset
# window_size = 10  # Number of time steps in each window
# batch_size = 32
#
# (X_train, X_test, y_train, y_test,
#  train_dataset, test_dataset,
#  feature_scaler, target_scaler) = create_windowed_dataset(
#     df, window_size=window_size, batch_size=batch_size)
#
# print(f"\nWindow dataset created:")
# print(f"X_train shape: {X_train.shape}")  # (samples, window_size, features)
# print(f"y_train shape: {y_train.shape}")  # (samples, 1)
# print(f"X_test shape: {X_test.shape}")
# print(f"y_test shape: {y_test.shape}")
#
#

# Step 4: Build and train an LSTM model
def build_lstm_model(input_shape, units=64):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units // 2),
        Dropout(0.2),
        Dense(1)  # Single output for regression
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# # Get input shape from training data
# input_shape = (X_train.shape[1], X_train.shape[2])  # (window_size, n_features)
#
# # Build the model
# model = build_lstm_model(input_shape)
# model.summary()
#
# # Train the model
# history = model.fit(
#     train_dataset,
#     epochs=20,
#     validation_data=test_dataset,
#     verbose=1
# )


# Step 5: Evaluate and visualize results
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()


# plot_training_history(history)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Inverse transform if data was scaled
# if feature_scaler is not None and target_scaler is not None:
#     y_test_inv = target_scaler.inverse_transform(y_test)
#     y_pred_inv = target_scaler.inverse_transform(y_pred)
# else:
#     y_test_inv = y_test
#     y_pred_inv = y_pred
#
# # Plot predictions vs actual
# plt.figure(figsize=(12, 6))
# plt.plot(y_test_inv[:100], label='Actual')
# plt.plot(y_pred_inv[:100], label='Predicted')
# plt.title('LSTM Model: Predictions vs Actual')
# plt.xlabel('Time Step')
# plt.ylabel('Target Value')
# plt.legend()
# plt.show()


# Step 6: Function to make predictions with sliding window
def predict_sequence(model, initial_window, n_steps, feature_scaler=None, target_scaler=None):
    """
    Predict a sequence using the trained LSTM model with a sliding window approach.

    Parameters:
    -----------
    model : keras model
        Trained LSTM model
    initial_window : numpy array
        Initial window of shape (1, window_size, n_features)
    n_steps : int
        Number of steps to predict
    feature_scaler : sklearn scaler
        Scaler used for features
    target_scaler : sklearn scaler
        Scaler used for target

    Returns:
    --------
    predictions : numpy array
        Array of predictions
    """
    curr_window = initial_window.copy()
    predictions = []

    for _ in range(n_steps):
        # Make a prediction
        pred = model.predict(curr_window, verbose=0)
        predictions.append(pred[0, 0])

        # Create new window by shifting and adding the prediction
        # We need to handle the feature scaling for the new point
        if feature_scaler is not None and target_scaler is not None:
            # This is a simplified approach - in a real scenario, you'd need to
            # properly transform the prediction into feature space
            new_point = curr_window[0, -1, :].reshape(1, -1)  # Take last point's features
            # In a real case, you might update features based on the prediction
        else:
            new_point = curr_window[0, -1, :].reshape(1, -1)

        # Shift window and add new point
        curr_window = np.roll(curr_window, -1, axis=1)
        curr_window[0, -1, :] = new_point

    # Inverse transform predictions if needed
    if target_scaler is not None:
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = target_scaler.inverse_transform(predictions).flatten()

    return np.array(predictions)


# # Example of using the prediction function
# initial_window = X_test[:1]  # Take first test window
# predictions = predict_sequence(model, initial_window, n_steps=50,
#                                feature_scaler=feature_scaler,
#                                target_scaler=target_scaler)
#
# print("\nSequence prediction example:")
# print(f"Predicted sequence shape: {predictions.shape}")