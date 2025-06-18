import matplotlib.pyplot as plt

from howest_dl.sessie_05.lstm_sonnet import (
    create_sample_data,
    create_windowed_dataset,
    build_lstm_model,
    plot_training_history,
    predict_sequence,
)


if __name__ == '__main__':
    # Step 1: Create sample DataFrame
    df = create_sample_data(n_samples=1000, n_features=5)
    print("Sample DataFrame:")
    print(df.head())

    # Step 3: Create the windowed dataset
    window_size = 10  # Number of time steps in each window
    batch_size = 32

    (X_train, X_test, y_train, y_test,
     train_dataset, test_dataset,
     feature_scaler, target_scaler) = create_windowed_dataset(
        df, window_size=window_size, batch_size=batch_size)

    print(f"\nWindow dataset created:")
    print(f"X_train shape: {X_train.shape}")  # (samples, window_size, features)
    print(f"y_train shape: {y_train.shape}")  # (samples, 1)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")


    # Get input shape from training data
    input_shape = (X_train.shape[1], X_train.shape[2])  # (window_size, n_features)

    # Step 4a: Build the model
    model = build_lstm_model(input_shape)
    model.summary()

    # Step 4b: Train the model
    history = model.fit(
        train_dataset,
        epochs=20,
        validation_data=test_dataset,
        verbose=1
    )

    # Step5: Visualize training history
    plot_training_history(history)

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse transform if data was scaled
    if feature_scaler is not None and target_scaler is not None:
        y_test_inv = target_scaler.inverse_transform(y_test)
        y_pred_inv = target_scaler.inverse_transform(y_pred)
    else:
        y_test_inv = y_test
        y_pred_inv = y_pred

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv[:100], label='Actual')
    plt.plot(y_pred_inv[:100], label='Predicted')
    plt.title('LSTM Model: Predictions vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Target Value')
    plt.legend()
    plt.show()



    # Step 6: Example of using the prediction function
    initial_window = X_test[:1]  # Take first test window
    predictions = predict_sequence(model, initial_window, n_steps=50,
                                   feature_scaler=feature_scaler,
                                   target_scaler=target_scaler)

    print("\nSequence prediction example:")
    print(f"Predicted sequence shape: {predictions.shape}")
