!git clone https://github.com/cknd/pyESN.git

import sys
sys.path.append('/content/pyESN')

from google.colab import files
uploaded = files.upload()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyESN import ESN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, precision_recall_curve, accuracy_score

data = pd.read_csv('continuous_factory_process.csv')

relevant_columns = [
    'Machine1.RawMaterialFeederParameter.U.Actual',
    'Machine1.Zone1Temperature.C.Actual',
    'Machine1.Zone2Temperature.C.Actual',
    'Stage2.Output.Measurement1.U.Actual',
    'Stage2.Output.Measurement1.U.Setpoint',
    'Stage2.Output.Measurement2.U.Actual',
    'Stage2.Output.Measurement2.U.Setpoint'
]
selected_data = data[relevant_columns]

# Sensor columns (input features)
sensor_columns = [
    'Machine1.RawMaterialFeederParameter.U.Actual',
    'Machine1.Zone1Temperature.C.Actual',
    'Machine1.Zone2Temperature.C.Actual',
    'Stage2.Output.Measurement1.U.Actual',
    'Stage2.Output.Measurement2.U.Actual'
]

normalized_data = (selected_data[sensor_columns] - selected_data[sensor_columns].min()) / (selected_data[sensor_columns].max() - selected_data[sensor_columns].min())

train_input, test_input, train_output, test_output = train_test_split(
    normalized_data.values[:-1], normalized_data.values[1:], test_size=0.2, random_state=42
)

esn = ESN(n_inputs=train_input.shape[1],
          n_outputs=train_output.shape[1],
          n_reservoir=1000,        # Increased reservoir size for better capacity
          sparsity=0.05,           # Reduced sparsity for more connections
          spectral_radius=1.2,     # Increased spectral radius to improve long-term memory
          random_state=42)

esn.fit(train_input, train_output)

predicted_train_output = esn.predict(train_input)
predicted_test_output = esn.predict(test_input)

predicted_train_flat = predicted_train_output.flatten()
train_output_flat = train_output.flatten()

predicted_test_flat = predicted_test_output.flatten()
test_output_flat = test_output.flatten()

smoothed_train_actual = pd.Series(train_output_flat).ewm(span=10).mean().values
smoothed_train_pred = pd.Series(predicted_train_flat).ewm(span=10).mean().values

smoothed_test_actual = pd.Series(test_output_flat).ewm(span=10).mean().values
smoothed_test_pred = pd.Series(predicted_test_flat).ewm(span=10).mean().values

error_margin = np.std(smoothed_train_actual - smoothed_train_pred)
threshold = 2 * error_margin

anomalies_train = np.where(np.abs(smoothed_train_pred - smoothed_train_actual) > threshold)[0]
anomalies_test = np.where(np.abs(smoothed_test_pred - smoothed_test_actual) > threshold)[0]

mse_train = mean_squared_error(train_output_flat, predicted_train_flat)
mae_train = mean_absolute_error(train_output_flat, predicted_train_flat)
r2_train = r2_score(train_output_flat, predicted_train_flat)

mse_test = mean_squared_error(test_output_flat, predicted_test_flat)
mae_test = mean_absolute_error(test_output_flat, predicted_test_flat)
r2_test = r2_score(test_output_flat, predicted_test_flat)

print(f"Train Set Mean Squared Error (MSE): {mse_train}")
print(f"Train Set Mean Absolute Error (MAE): {mae_train}")
print(f"Train Set R-squared: {r2_train}")
print(f"Test Set Mean Squared Error (MSE): {mse_test}")
print(f"Test Set Mean Absolute Error (MAE): {mae_test}")
print(f"Test Set R-squared: {r2_test}")


f1 = f1_score(test_output_flat > threshold, predicted_test_flat > threshold)
print(f"F1 Score (Test Set): {f1}")

from sklearn.model_selection import train_test_split

# Split your data into training and validation sets
train_input, val_input, train_output, val_output = train_test_split(
    normalized_data.values[:-1],
    normalized_data.values[1:],
    test_size=0.2,
    random_state=42
)


from sklearn.metrics import accuracy_score

# Lists to store losses and accuracies
training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []

# Number of epochs
num_epochs = 100  # Set your desired number of epochs

for epoch in range(20):
    # Fit the ESN model on training data
    esn.fit(train_input, train_output)

    # Make predictions for training set
    predicted_train_output = esn.predict(train_input)
    train_loss = mean_squared_error(train_output, predicted_train_output)
    train_accuracy = accuracy_score(train_output > threshold, predicted_train_output > threshold)  # Adjust threshold as needed

    # Make predictions for validation set
    predicted_val_output = esn.predict(val_input)
    val_loss = mean_squared_error(val_output, predicted_val_output)
    val_accuracy = accuracy_score(val_output > threshold, predicted_val_output > threshold)  # Adjust threshold as needed

    # Store losses and accuracies
    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    training_accuracies.append(train_accuracy)
    validation_accuracies.append(val_accuracy)

    # Print losses and accuracies for monitoring
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

