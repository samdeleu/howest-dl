import pandas as pd
import numpy as np

def create_windowed_dataset(dataframe, window_size,
                            target_column='failure', timing_column='cycle'):
    """
    Create a windowed dataset from a pandas DataFrame for LSTM training.


    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input DataFrame containing features, target, and timing columns
    window_size : int
        The number of time steps to include in each window
    target_column : str
        Name of the target column in the DataFrame
    timing_column : str
        Name of the cycle/time column in the DataFrame

    Returns:
    --------
    X_windowed : numpy array
        feature set from dataframe with shape (samples, window_size, features)
    y_windowed : numpy array
        targets from dataframe with shape(samples, 1)
    """
    # Sort by cycle to ensure time order
    df_sorted = dataframe.sort_values(by=timing_column).reset_index(drop=True)

    # Separate features and target
    features_df = df_sorted.drop([target_column, timing_column], axis=1)
    target_df = df_sorted[target_column]

    features_np = features_df.to_numpy()
    target_np_0 = target_df.to_numpy()
    target_np = target_np_0.reshape(-1, 1)

    # Create windows
    X = []
    y = []
    time_tracker = []

    for i in range(len(features_np)):
        # If at start of the dataset, add padding if window size not reached yet
        if i < window_size - 1:
            xx = features_np[i].shape
            padding = np.zeros(shape=(window_size -i - 1, features_np.shape[1]), dtype=features_np[i].dtype)
            X.append(np.vstack((padding, features_np[:i + 1])))
        else:
            # Add window to dataset
            X.append(features_np[i-window_size+1:i+1])

        # define the targete
        y.append(target_np[i])
        time_tracker.append(df_sorted.iloc[i][timing_column])


    # Convert to numpy arrays
    X_windowed = np.array(X)
    y_windowed = np.array(y)
    time_tracker_np = np.array(time_tracker)

    return X_windowed, y_windowed , time_tracker_np


def create_windowed_dataset_with_df(
        dataframe, window_size,
        target_column='failure', timing_column='ttf',
        identifier="X",
):
    """
    Create a windowed dataset from a pandas DataFrame for LSTM training.


    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input DataFrame containing features, target, and timing columns
    window_size : int
        The number of time steps to include in each window
    target_column : str
        Name of the target column in the DataFrame
    timing_column : str
        Name of the cycle/time column in the DataFrame

    Returns:
    --------
    X_windowed : numpy array
        feature set from dataframe with shape (samples, window_size, features)
    y_windowed : numpy array
        targets from dataframe with shape(samples, 1)
    """
    # Sort by cycle to ensure time order
    df_sorted = dataframe.sort_values(by=timing_column).reset_index(drop=True)

    # Separate features and target
    features_df = df_sorted.drop([target_column, timing_column], axis=1)
    target_df = df_sorted[target_column]

    features_np = features_df.to_numpy()
    target_np_0 = target_df.to_numpy()
    target_np = target_np_0.reshape(-1, 1)

    # Create windows
    windowed_dataset = []
    X = []
    y = []
    time_tracker = []

    for i in range(len(features_np)):
        # If at start of the dataset, add padding if window size not reached yet
        if i < window_size - 1:
            xx = features_np[i].shape
            padding = np.zeros(shape=(window_size -i - 1, features_np.shape[1]), dtype=features_np[i].dtype)
            X_data = np.vstack((padding, features_np[:i + 1]))
        else:
            # Add window to dataset
            X_data = features_np[i-window_size+1:i+1]

        # collect X
        X.append(X_data)

        # collect y
        y.append(target_np[i])

        # collect time
        time_tracker.append(df_sorted.iloc[i][timing_column])

        data_entry = {
            "id": identifier,
            "X": X_data,
            "y": target_np[i],
            "time_tracker": df_sorted.iloc[i][timing_column]
        }
        windowed_dataset.append(data_entry)

    # Convert to numpy arrays
    X_windowed = np.array(X)
    y_windowed = np.array(y)
    time_tracker_np = np.array(time_tracker)

    X_df = pd.DataFrame(windowed_dataset)

    return X_windowed, y_windowed , time_tracker_np, X_df

if __name__ == '__main__':
    df_1 = pd.DataFrame(data={
        "time": [    0,    1,    2,    3,    4,    5,    6,    7,    8,    9,],
        "B": [      10,   11,   22,   33,   44,   55,   66,   77,   88,   99,],
        "C": [     100,  111,  222,  333,  444,  555,  666,  777,  888,  999,],
        "D": [    1000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, ],
        "target": [  0,    1,    2,    3,    4,    5,    6,    7,    8,    9, ],
    })

    df_2 = pd.DataFrame(data={
        "time": [    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,],
        "B": [      10,   11,   22,   33,   44,   55,   66,   77,   88,   99,],
        "C": [     100,  111,  222,  333,  444,  555,  666,  777,  888,  999,],
        "D": [    1000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, ],
        "target": [  0,    1,    2,    3,    4,    5,    6,    7,    8,    9, ],
    })

    # Step 1: Create sample DataFrame
    print("Sample DataFrame df_1:")
    print(df_1.head())

    print("Sample DataFrame df_2:")
    print(df_2.head())

    # Step 3: Create the windowed dataset
    WINDOW_SIZE = 6  # Number of time steps in each window

    # (X_train, y_train, time_tracker) = create_windowed_dataset(
    #     dataframe=df_1,
    #     target_column="target",
    #     timing_column="time",
    #     window_size=WINDOW_SIZE,
    # )
    #
    # print(f"{X_train.shape = }")


    (X_df1, y_df1, time_df1, data_df_1) = create_windowed_dataset_with_df(
        dataframe=df_1,
        target_column="target",
        timing_column="time",
        window_size=WINDOW_SIZE,
        identifier=1,
    )

    print(data_df_1)
    X_from_df_1 = np.stack(data_df_1['X'].to_numpy())
    print(X_from_df_1.shape)
    print(X_from_df_1)

    (X_df2, y_df2, time_df2, data_df_2) = create_windowed_dataset_with_df(
        dataframe=df_2,
        target_column="target",
        timing_column="time",
        window_size=WINDOW_SIZE,
        identifier = 2,
    )

    print(data_df_2)
    X_from_df_2 = np.stack(data_df_2['X'].to_numpy())
    print(X_from_df_2.shape)
    print(X_from_df_2)

    xx = pd.concat([data_df_1, data_df_2], axis=0).sort_values(["time_tracker"], ascending=False)
    print(xx)
