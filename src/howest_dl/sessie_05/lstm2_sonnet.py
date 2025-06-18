import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class LSTMBinaryClassifier:
    """
    LSTM-based Binary Classifier with customizable hyperparameters
    """

    def __init__(self, window_size, n_features):
        """
        Initialize the model with window size and number of features

        Parameters:
        -----------
        window_size : int
            The size of the time window for each sample
        n_features : int
            The number of features in each time step
        """
        self.window_size = window_size
        self.n_features = n_features
        self.model = None
        self.history = None
        self.scaler = StandardScaler()

    def create_windowed_dataset(self, X, y, window_size=None):
        """
        Create a windowed dataset from sequential data

        Parameters:
        -----------
        X : numpy array
            The input features with shape (n_samples, n_features)
        y : numpy array
            The target values with shape (n_samples,)
        window_size : int, optional
            The window size to use (defaults to self.window_size)

        Returns:
        --------
        X_windowed : numpy array
            The windowed input with shape (n_windows, window_size, n_features)
        y_windowed : numpy array
            The target values for each window with shape (n_windows,)
        """
        if window_size is None:
            window_size = self.window_size

        X_windowed = []
        y_windowed = []

        for i in range(len(X) - window_size):
            X_windowed.append(X[i:i + window_size])
            y_windowed.append(y[i + window_size])  # Predict the next value after the window

        return np.array(X_windowed), np.array(y_windowed)

    def build_model(self, lstm_units=[64], dropout_rate=0.2, recurrent_dropout=0.0,
                    bidirectional=False, optimizer='adam', learning_rate=0.001):
        """
        Build the LSTM model with specified hyperparameters

        Parameters:
        -----------
        lstm_units : list of int
            The number of units in each LSTM layer
        dropout_rate : float
            The dropout rate after each LSTM layer
        recurrent_dropout : float
            The recurrent dropout rate within LSTM cells
        bidirectional : bool
            Whether to use bidirectional LSTM layers
        optimizer : str
            The optimizer to use ('adam' or 'rmsprop')
        learning_rate : float
            The learning rate for the optimizer
        """
        model = Sequential()

        # Input layer shape: (window_size, n_features)
        for i, units in enumerate(lstm_units):
            if i == 0:  # First layer
                if bidirectional:
                    model.add(Bidirectional(
                        LSTM(units, return_sequences=(i < len(lstm_units) - 1),
                             recurrent_dropout=recurrent_dropout),
                        input_shape=(self.window_size, self.n_features)
                    ))
                else:
                    model.add(LSTM(
                        units, return_sequences=(i < len(lstm_units) - 1),
                        recurrent_dropout=recurrent_dropout,
                        input_shape=(self.window_size, self.n_features)
                    ))
            else:  # Subsequent layers
                if bidirectional:
                    model.add(Bidirectional(
                        LSTM(units, return_sequences=(i < len(lstm_units) - 1),
                             recurrent_dropout=recurrent_dropout)
                    ))
                else:
                    model.add(LSTM(
                        units, return_sequences=(i < len(lstm_units) - 1),
                        recurrent_dropout=recurrent_dropout
                    ))

            model.add(Dropout(dropout_rate))

        # Output layer (sigmoid for binary classification)
        model.add(Dense(1, activation='sigmoid'))

        # Configure optimizer
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Compile the model
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )

        self.model = model
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50,
            validation_split=0.2, verbose=1, early_stopping=True, patience=10):
        """
        Train the model on the provided data

        Parameters:
        -----------
        X_train : numpy array
            The training features
        y_train : numpy array
            The training target values
        X_val : numpy array, optional
            The validation features
        y_val : numpy array, optional
            The validation target values
        batch_size : int
            The batch size for training
        epochs : int
            The maximum number of epochs to train
        validation_split : float
            The fraction of training data to use for validation if X_val is not provided
        verbose : int
            The verbosity level for training
        early_stopping : bool
            Whether to use early stopping
        patience : int
            The patience for early stopping

        Returns:
        --------
        history : History object
            The training history
        """
        # Check if model has been built
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Prepare callbacks
        callbacks = []

        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ))

        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ))

        callbacks.append(ModelCheckpoint(
            'best_lstm_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ))

        # Train the model
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        self.history = history
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data

        Parameters:
        -----------
        X_test : numpy array
            The test features
        y_test : numpy array
            The test target values

        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': report
        }

        return metrics

    def plot_training_history(self):
        """Plot the training history"""
        if self.history is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_evaluation_metrics(self, metrics):
        """Plot evaluation metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # ROC curve
        ax1.plot(metrics['fpr'], metrics['tpr'], label=f"AUC = {metrics['roc_auc']:.3f}")
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Precision-Recall curve
        ax2.plot(metrics['recall'], metrics['precision'])
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)

        # Confusion matrix
        cm = metrics['confusion_matrix']
        ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax3.set_title('Confusion Matrix')
        tick_marks = np.arange(2)
        ax3.set_xticks(tick_marks)
        ax3.set_yticks(tick_marks)
        ax3.set_xticklabels(['0', '1'])
        ax3.set_yticklabels(['0', '1'])

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        ax3.set_ylabel('True Label')
        ax3.set_xlabel('Predicted Label')

        # Classification report
        report = metrics['classification_report']
        ax4.axis('off')
        ax4.text(0.1, 0.9, f"Accuracy: {report['accuracy']:.3f}", fontsize=12)
        ax4.text(0.1, 0.8, f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}", fontsize=12)
        ax4.text(0.1, 0.7, f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}", fontsize=12)
        ax4.text(0.1, 0.6, f"Class 0 Precision: {report['0']['precision']:.3f}", fontsize=12)
        ax4.text(0.1, 0.5, f"Class 0 Recall: {report['0']['recall']:.3f}", fontsize=12)
        ax4.text(0.1, 0.4, f"Class 0 F1: {report['0']['f1-score']:.3f}", fontsize=12)
        ax4.text(0.1, 0.3, f"Class 1 Precision: {report['1']['precision']:.3f}", fontsize=12)
        ax4.text(0.1, 0.2, f"Class 1 Recall: {report['1']['recall']:.3f}", fontsize=12)
        ax4.text(0.1, 0.1, f"Class 1 F1: {report['1']['f1-score']:.3f}", fontsize=12)
        ax4.set_title('Classification Report')

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        """Make predictions with the model"""
        return self.model.predict(X)

    def save_model(self, filepath):
        """Save the model to a file"""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load the model from a file"""
        self.model = tf.keras.models.load_model(filepath)


# Example usage
def generate_sample_data(n_samples=1000, n_features=10, window_size=20, seed=42):
    """Generate sample data for demonstration"""
    np.random.seed(seed)

    # Generate features
    X = np.random.normal(0, 1, (n_samples, n_features))

    # Generate target (binary classification)
    # Make it dependent on patterns in the data
    y = np.zeros(n_samples)

    # Create some patterns
    for i in range(window_size, n_samples):
        # Pattern 1: If the sum of feature 0 over the window is high
        if np.sum(X[i - window_size:i, 0]) > window_size * 0.5:
            y[i] = 1

        # Pattern 2: If feature 1 is trending up
        elif np.all(np.diff(X[i - window_size:i, 1]) > 0):
            y[i] = 1

        # Pattern 3: If feature 2 and 3 are both positive
        elif X[i, 2] > 0.5 and X[i, 3] > 0.5:
            y[i] = 1

    # Add some noise
    flip_indices = np.random.choice(range(window_size, n_samples), size=int(0.1 * n_samples), replace=False)
    y[flip_indices] = 1 - y[flip_indices]

    return X, y.astype(int)


# Main execution
def main():
    # Parameters
    window_size = 20
    n_features = 10
    batch_size = 32
    epochs = 50
    lstm_units = [64, 32]  # Two LSTM layers with 64 and 32 units
    dropout_rate = 0.3
    recurrent_dropout = 0.2
    bidirectional = True
    optimizer = 'adam'
    learning_rate = 0.001

    # Generate data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=2000, n_features=n_features, window_size=window_size)

    # Create classifier
    print("Creating LSTM classifier...")
    classifier = LSTMBinaryClassifier(window_size=window_size, n_features=n_features)

    # Create windowed dataset
    print("Creating windowed dataset...")
    X_windowed, y_windowed = classifier.create_windowed_dataset(X, y)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_windowed, y_windowed, test_size=0.2, random_state=42)

    # Build model
    print("Building model...")
    classifier.build_model(
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        bidirectional=bidirectional,
        optimizer=optimizer,
        learning_rate=learning_rate
    )

    # Print model summary
    classifier.model.summary()

    # Train model
    print("Training model...")
    classifier.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        early_stopping=True,
        patience=10
    )

    # Plot training history
    classifier.plot_training_history()

    # Evaluate model
    print("Evaluating model...")
    metrics = classifier.evaluate(X_test, y_test)

    # Print evaluation results
    print(f"Test