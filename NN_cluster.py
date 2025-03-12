import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from datetime import datetime

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def normalize_power_system_phases(phases):
    """Normalize phases by shifting +/-120° clusters toward 0°"""
    normalized_phases = phases.copy()
    
    # Constants for three-phase system (in radians)
    phase_120 = 2*np.pi/3  # 120 degrees in radians
    
    # Normalize positive cluster (around +120°)
    pos_mask = (normalized_phases > 1.9) & (normalized_phases < 2.3)
    normalized_phases[pos_mask] -= phase_120
    
    # Normalize negative cluster (around -120°)
    neg_mask = (normalized_phases > -2.3) & (normalized_phases < -1.9)
    normalized_phases[neg_mask] += phase_120
    
    return normalized_phases

def denormalize_power_system_phases(normalized_phases, original_phases):
    """
    Denormalize phases by shifting them back to their original clusters.
    Requires the original phases to determine which cluster each value belongs to.
    """
    denormalized_phases = normalized_phases.copy()
    phase_120 = 2*np.pi/3
    
    # Identify which points were in the +120° cluster
    pos_mask = (original_phases > 1.9) & (original_phases < 2.3)
    denormalized_phases[pos_mask] += phase_120
    
    # Identify which points were in the -120° cluster
    neg_mask = (original_phases > -2.3) & (original_phases < -1.9)
    denormalized_phases[neg_mask] -= phase_120
    
    return denormalized_phases
def plot_residuals(actual, predicted, o_dir, name, label=""):
    """Plot residuals scatter plot."""
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    residuals = actual - predicted
    plt.figure(figsize=(10, 5))
    plt.scatter(actual, residuals, label=f'{label} Residuals', color='red', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(f'Actual {label} Values')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title(f'{label} Residuals Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(o_dir, name), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying
    
    return residuals



def train_and_evaluate_nn(input_df, output_df, amplitude_cols=None, phase_cols=None, hidden_size=10, lr=0.001, num_epochs=1000):
    """
    Train and evaluate a neural network with configurable amplitude and phase columns.
    
    Parameters:
    -----------
    input_df : pandas.DataFrame
        Input data
    output_df : pandas.DataFrame
        Output data
    amplitude_cols : tuple or list, optional
        Start and end indices for amplitude columns (e.g., (0, 25) for columns 0-24), 
        default is first half of columns
    phase_cols : tuple or list, optional
        Start and end indices for phase columns (e.g., (26, 51) for columns 26-50),
        default is second half of columns  
    hidden_size : int, optional
        Size of the hidden layer in the neural network
    lr : float, optional
        Learning rate for the optimizer
    num_epochs : int, optional
        Number of training epochs
    """
    # Extract data
    X = input_df.values
    y = output_df.values
    
    # Configure amplitude and phase columns if not specified
    if amplitude_cols is None:
        amplitude_end = y.shape[1] // 2
        amplitude_cols = (0, amplitude_end)
    
    if phase_cols is None:
        phase_start = amplitude_cols[1]
        phase_cols = (phase_start, y.shape[1])
    
    # Print column configuration
    print(f"Amplitude columns: {amplitude_cols[0]} to {amplitude_cols[1]-1}")
    print(f"Phase columns: {phase_cols[0]} to {phase_cols[1]-1}")
    
    # Split data into training and testing sets
    n_samples = X.shape[0]
    n_train = int(n_samples * 0.7)
    n_test = int(n_samples * 0.15)
    n_eval = int(n_samples * 0.15)

    X_train, X_test , X_eval = X[:n_train], X[n_train:n_test+n_train], X[n_test+n_train:]
    y_train, y_test, y_eval = y[:n_train], y[n_train:n_test+n_train], y[n_test+n_train:]
    
    # Check sizes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_eval shape: {X_eval.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_eval shape: {y_eval.shape}")


    # Normalize data
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train_normalized = (X_train - X_mean) / X_std
    X_test_normalized = (X_test - X_mean) / X_std
    X_eval_normalized = (X_eval - X_mean) / X_std

    #y_mean = np.mean(y_train, axis=0)
    #y_std = np.std(y_train, axis=0)
    #y_train_normalized = (y_train - y_mean) / y_std
    #y_test_normalized = (y_test - y_mean) / y_std
    #y_eval_normalized = (y_eval - y_mean) / y_std

    # Split data into voltage magnitudes and phases
    amplitude_end = y.shape[1] // 2  # Assuming equal split between voltage and phase

    # Split training data
    y_voltage_train = y_train[:, :amplitude_end]
    y_phase_train = y_train[:, amplitude_end:]

    # Split test data
    y_voltage_test = y_test[:, :amplitude_end]
    y_phase_test = y_test[:, amplitude_end:]

    # Split evaluation data
    y_voltage_eval = y_eval[:, :amplitude_end]
    y_phase_eval = y_eval[:, amplitude_end:]

    # Store original phases for later denormalization
    y_phase_train_original = y_phase_train.copy()
    y_phase_test_original = y_phase_test.copy()
    y_phase_eval_original = y_phase_eval.copy()

    # 1. For voltage components: Use min-max scaling
    voltage_min = np.min(y_voltage_train, axis=0)
    voltage_max = np.max(y_voltage_train, axis=0)
    voltage_range = voltage_max - voltage_min + 1e-8  # Small epsilon to avoid division by zero

    # Apply phase normalization before trigonometric conversion
    y_phase_train_normalized = normalize_power_system_phases(y_phase_train)
    y_phase_test_normalized = normalize_power_system_phases(y_phase_test)
    y_phase_eval_normalized = normalize_power_system_phases(y_phase_eval)

    # Normalize voltage data
    y_voltage_train_normalized = (y_voltage_train - voltage_min) / voltage_range
    y_voltage_test_normalized = (y_voltage_test - voltage_min) / voltage_range
    y_voltage_eval_normalized = (y_voltage_eval - voltage_min) / voltage_range

    # 2. For phase components: Use trigonometric transformation
    # Convert phase angles to sin/cos pairs
    y_phase_train_sin = np.sin(y_phase_train_normalized)
    y_phase_train_cos = np.cos(y_phase_train_normalized)

    y_phase_test_sin = np.sin(y_phase_test_normalized)
    y_phase_test_cos = np.cos(y_phase_test_normalized)

    y_phase_eval_sin = np.sin(y_phase_eval_normalized)
    y_phase_eval_cos = np.cos(y_phase_eval_normalized)

    # 3. Combine voltage and phase components
    y_train_processed = np.hstack([y_voltage_train_normalized, 
                                y_phase_train_sin, 
                                y_phase_train_cos])

    y_test_processed = np.hstack([y_voltage_test_normalized, 
                                y_phase_test_sin, 
                                y_phase_test_cos])

    y_eval_processed = np.hstack([y_voltage_eval_normalized, 
                                y_phase_eval_sin, 
                                y_phase_eval_cos])

    # Convert to PyTorch tensors
    y_train_tensor = torch.tensor(y_train_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_processed, dtype=torch.float32)
    y_eval_tensor = torch.tensor(y_eval_processed, dtype=torch.float32)

    # Store the shapes to use during denormalization
    voltage_output_size = y_voltage_train.shape[1]
    phase_output_size = y_phase_train.shape[1]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    #y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    #y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32)
    X_eval_tensor = torch.tensor(X_eval_normalized, dtype=torch.float32)
    #y_eval_tensor = torch.tensor(y_eval_normalized, dtype=torch.float32)

    # Set up the neural network
    input_size = X_train_tensor.shape[1]
    output_size = y_train_tensor.shape[1]
    Net = SimpleNN(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Net.parameters(), lr=lr, weight_decay=1e-5)

    # Training loop
    losses = []
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        Net.train()
        # Forward pass
        outputs = Net(X_train_tensor)
        train_loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # Evaluate on Test Data
        Net.eval()
        with torch.no_grad():
            val_outputs = Net(X_eval_tensor)
            val_loss = criterion(val_outputs, y_eval_tensor)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    
    # Create a directory for saving plots if it doesn't exist
    output_dir = "output_plots_cluster"
    os.makedirs(output_dir, exist_ok=True)

    # Plot Training vs val Loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", alpha=0.7)
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(output_dir, "training_validation_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying

    # Evaluate on test data
    # Evaluate on test data
    Net.eval()
    with torch.no_grad():
        test_predictions = Net(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f'Test Loss (MSE): {test_loss.item():.4f}')

        # Calculate Mean Absolute Error (MAE) as an additional metric
        test_mae = torch.mean(torch.abs(test_predictions - y_test_tensor))
        print(f'Test MAE: {test_mae.item():.4f}')

    test_predictions = test_predictions.numpy()
    # Split predictions into voltage and phase components
    test_voltage_predictions = test_predictions[:, :voltage_output_size]
    test_phase_predictions = test_predictions[:, voltage_output_size:]

    # Split phase predictions into sin and cos components
    test_phase_sin = test_phase_predictions[:, :phase_output_size]
    test_phase_cos = test_phase_predictions[:, phase_output_size:]

    # Denormalize voltage predictions
    test_voltage_denormalized = test_voltage_predictions * voltage_range + voltage_min

    # Convert phase sin/cos back to normalized angles
    normalized_phase_angles = np.arctan2(test_phase_sin, test_phase_cos)

    # Denormalize phase angles using the original phase clusters as reference
    test_phase_denormalized = denormalize_power_system_phases(normalized_phase_angles, y_phase_test_original)

    # Combine voltage and phase components
    test_predictions_denormalized = np.hstack([test_voltage_denormalized, test_phase_denormalized])

    # Save to CSV
    pd.DataFrame(test_predictions_denormalized).to_csv('cluster_5_voltage_predictions.csv', index=False, header=False)

    # Similarly for actual test values
    y_test_voltage = y_test_tensor.numpy()[:, :voltage_output_size]
    y_test_phase_sin = y_test_tensor.numpy()[:, voltage_output_size:voltage_output_size+phase_output_size]
    y_test_phase_cos = y_test_tensor.numpy()[:, voltage_output_size+phase_output_size:]

    # Denormalize voltage actuals
    y_test_voltage_denormalized = y_test_voltage * voltage_range + voltage_min

    # Convert phase sin/cos back to normalized angles
    normalized_y_test_phase = np.arctan2(y_test_phase_sin, y_test_phase_cos)

    # Denormalize using original phase angles
    y_test_phase_denormalized = denormalize_power_system_phases(normalized_y_test_phase, y_phase_test_original)

    # Combine voltage and phase components
    y_test_denormalized = np.hstack([y_test_voltage_denormalized, y_test_phase_denormalized])

    # Print Shapes for Debugging
    print("Predictions Shape:", test_predictions_denormalized.shape)
    print("Actual Test Data Shape:", y_test_denormalized.shape)

    # Separate the data into two parts: Amplitudes and Phases using configurable indices
    amp_start, amp_end = amplitude_cols
    phase_start, phase_end = phase_cols

    voltage_actual = y_test_denormalized[:, amp_start:amp_end]
    voltage_predicted = test_predictions_denormalized[:, amp_start:amp_end]

    phases_actual = y_test_denormalized[:, phase_start:phase_end]
    phases_predicted = test_predictions_denormalized[:, phase_start:phase_end]

    """Compute RMSE and print results."""

    # Calculate metrics for voltage/amplitude
    voltage_r2 = r2_score(voltage_actual.flatten(), voltage_predicted.flatten())
    voltage_rmse = np.sqrt(mean_squared_error(voltage_actual.flatten(), voltage_predicted.flatten()))

    # Calculate metrics for phase
    phase_r2 = r2_score(phases_actual.flatten(), phases_predicted.flatten())
    phase_rmse = np.sqrt(mean_squared_error(phases_actual.flatten(), phases_predicted.flatten()))
    
    # Plot for the voltages greater than zero
    plt.figure(figsize=(10, 5))
    plt.scatter(voltage_actual, voltage_predicted, label='Predicted vs Actual (amplitudes)', color='blue')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Amplitude - Predicted vs Actual Values (Filtered)\nR² = {voltage_r2:.4f}, RMSE = {voltage_rmse:.4f}')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(output_dir, "all_v.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying

    # Plot for phases
    plt.figure(figsize=(10, 5))
    plt.scatter(phases_actual.flatten(), phases_predicted.flatten(), label='Predicted vs Actual (phases)', color='green')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Phases vs Actual Values\nR² = {phase_r2:.4f}, RMSE = {phase_rmse:.4f}')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(output_dir, "all_ph.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying

    # Flatten the arrays for easier handling
    phases_actual_flat = phases_actual.flatten()
    phases_predicted_flat = phases_predicted.flatten()

    voltage_mask = voltage_actual.flatten() > 0
    voltage_actual_filtered = voltage_actual.flatten()[voltage_mask]
    voltage_predicted_filtered = voltage_predicted.flatten()[voltage_mask]

    voltage_residuals = plot_residuals(voltage_actual_filtered, voltage_predicted_filtered, output_dir, "all_voltages_res.png", label="Voltage")
    phase_residuals = plot_residuals(phases_actual, phases_predicted,  output_dir, "all_phases_res.png", label="Phase")

    # Create masks for thresholds
    below_negative_one_mask = phases_actual_flat < -1
    between_neg1_and_1_mask = (phases_actual_flat >= -1) & (phases_actual_flat <= 1)
    above_one_mask = phases_actual_flat > 1

    # Categorize phases
    below_negative_one_actual = phases_actual_flat[below_negative_one_mask]
    below_negative_one_predicted = phases_predicted_flat[below_negative_one_mask]

    between_neg1_and_1_actual = phases_actual_flat[between_neg1_and_1_mask]
    between_neg1_and_1_predicted = phases_predicted_flat[between_neg1_and_1_mask]

    above_one_actual = phases_actual_flat[above_one_mask]
    above_one_predicted = phases_predicted_flat[above_one_mask]

    # Calculate metrics for each phase category
    if len(below_negative_one_actual) > 0:
        below_neg_r2 = r2_score(below_negative_one_actual, below_negative_one_predicted)
        below_neg_rmse = np.sqrt(mean_squared_error(below_negative_one_actual, below_negative_one_predicted))
    else:
        below_neg_r2, below_neg_rmse = 0, 0
        
    if len(between_neg1_and_1_actual) > 0:
        between_r2 = r2_score(between_neg1_and_1_actual, between_neg1_and_1_predicted)
        between_rmse = np.sqrt(mean_squared_error(between_neg1_and_1_actual, between_neg1_and_1_predicted))
    else:
        between_r2, between_rmse = 0, 0
        
    if len(above_one_actual) > 0:
        above_r2 = r2_score(above_one_actual, above_one_predicted)
        above_rmse = np.sqrt(mean_squared_error(above_one_actual, above_one_predicted))
    else:
        above_r2, above_rmse = 0, 0

    # Plot categorized scatter plots
    plt.figure(figsize=(15, 10))

    # Below -1
    plt.subplot(3, 1, 1)
    plt.scatter(below_negative_one_actual, below_negative_one_predicted, color='red', label='-120°', alpha=0.5)
    if len(below_negative_one_actual) > 0:
        ideal_line = np.linspace(min(below_negative_one_actual), max(below_negative_one_actual), 100)
        plt.plot(ideal_line, ideal_line, 'r--')
    plt.xlabel('Actual Values [rad]')
    plt.ylabel('Predicted Values [rad]')
    plt.title(f'Phases: -120°\nR² = {below_neg_r2:.4f}, RMSE = {below_neg_rmse:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Between -1 and 1
    plt.subplot(3, 1, 2)
    plt.scatter(between_neg1_and_1_actual, between_neg1_and_1_predicted, color='blue', label='0°', alpha=0.5)
    if len(between_neg1_and_1_actual) > 0:
        ideal_line = np.linspace(min(between_neg1_and_1_actual), max(between_neg1_and_1_actual), 100)
        plt.plot(ideal_line, ideal_line, 'r--')
    plt.xlabel('Actual Values[rad]')
    plt.ylabel('Predicted Values[rad]')
    plt.title(f'Phases: 0°\nR² = {between_r2:.4f}, RMSE = {between_rmse:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Above 1
    plt.subplot(3, 1, 3)
    plt.scatter(above_one_actual, above_one_predicted, color='green', label='120°', alpha=0.5)
    if len(above_one_actual) > 0:
        ideal_line = np.linspace(min(above_one_actual), max(above_one_actual), 100)
        plt.plot(ideal_line, ideal_line, 'r--')
    plt.xlabel('Actual Values[rad]')
    plt.ylabel('Predicted Values[rad]')
    plt.title(f'Phases: +120°\nR² = {above_r2:.4f}, RMSE = {above_rmse:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_dir, "phases.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying


    plot_residuals(below_negative_one_actual, below_negative_one_predicted, output_dir, "ph_below_minus_one.png", label="Below -1")
    plot_residuals(between_neg1_and_1_actual, between_neg1_and_1_predicted, output_dir, "ph_between_minus_one.png", label="Between -1 and 1")
    plot_residuals(above_one_actual, above_one_predicted, output_dir, "ph_above_one.png", label="Above 1")
    
    plt.hist(phases_actual.flatten(), bins=50, color="blue", alpha=0.7)
    plt.xlabel("Phase Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Phase Values in Training Data")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "phases_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying

    phase_differences = np.abs(phases_predicted - phases_actual)
    # Correct for circular differences (e.g., -179° vs +179°)
    phase_differences = np.minimum(phase_differences, 2*np.pi - phase_differences)
    plt.hist(phase_differences, bins=50)
    plt.savefig(os.path.join(output_dir, "phasediff.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying
    # Initialize categorized voltages
    below_negative_one_actual_voltages = []
    below_negative_one_predicted_voltages = []

    between_neg1_and_1_actual_voltages = []
    between_neg1_and_1_predicted_voltages = []

    above_one_actual_voltages = []
    above_one_predicted_voltages = []

    # Iterate row-by-row
    for row_idx in range(phases_actual.shape[0]):
        # Extract rows
        phase_row = phases_actual[row_idx]
        voltage_actual_row = voltage_actual[row_idx]
        voltage_predicted_row = voltage_predicted[row_idx]

        # Ensure all arrays have the same length
        min_length = min(len(phase_row), len(voltage_actual_row), len(voltage_predicted_row))
        phase_row = phase_row[:min_length]
        voltage_actual_row = voltage_actual_row[:min_length]
        voltage_predicted_row = voltage_predicted_row[:min_length]

        # Create masks for phases
        below_negative_one_mask = phase_row < -1
        between_neg1_and_1_mask = (phase_row >= -1) & (phase_row <= 1)
        above_one_mask = phase_row > 1

        # Apply masks to voltages row-by-row
        below_negative_one_actual_voltages.extend(voltage_actual_row[below_negative_one_mask])
        below_negative_one_predicted_voltages.extend(voltage_predicted_row[below_negative_one_mask])

        between_neg1_and_1_actual_voltages.extend(voltage_actual_row[between_neg1_and_1_mask])
        between_neg1_and_1_predicted_voltages.extend(voltage_predicted_row[between_neg1_and_1_mask])

        above_one_actual_voltages.extend(voltage_actual_row[above_one_mask])
        above_one_predicted_voltages.extend(voltage_predicted_row[above_one_mask])

 # Calculate metrics for each voltage category
    if len(below_negative_one_actual_voltages) > 0:
        below_voltage_r2 = r2_score(below_negative_one_actual_voltages, below_negative_one_predicted_voltages)
        below_voltage_rmse = np.sqrt(mean_squared_error(below_negative_one_actual_voltages, below_negative_one_predicted_voltages))
    else:
        below_voltage_r2, below_voltage_rmse = 0, 0
        
    if len(between_neg1_and_1_actual_voltages) > 0:
        between_voltage_r2 = r2_score(between_neg1_and_1_actual_voltages, between_neg1_and_1_predicted_voltages)
        between_voltage_rmse = np.sqrt(mean_squared_error(between_neg1_and_1_actual_voltages, between_neg1_and_1_predicted_voltages))
    else:
        between_voltage_r2, between_voltage_rmse = 0, 0
        
    if len(above_one_actual_voltages) > 0:
        above_voltage_r2 = r2_score(above_one_actual_voltages, above_one_predicted_voltages)
        above_voltage_rmse = np.sqrt(mean_squared_error(above_one_actual_voltages, above_one_predicted_voltages))
    else:
        above_voltage_r2, above_voltage_rmse = 0, 0

    # Plot categorized voltage scatter plots
    plt.figure(figsize=(15, 10))

    # Voltages for Phases < -1
    plt.subplot(3, 1, 1)
    plt.scatter(below_negative_one_actual_voltages, below_negative_one_predicted_voltages, 
               color='red', label='Voltages for -120°  ', alpha=0.5)
    if len(below_negative_one_actual_voltages) > 0:
        ideal_line = np.linspace(min(below_negative_one_actual_voltages), max(below_negative_one_actual_voltages), 100)
        plt.plot(ideal_line, ideal_line, 'r--')
    plt.xlabel('Actual Voltages (V)')
    plt.ylabel('Predicted Voltages (V)')
    plt.title(f'Voltages Corresponding to -120°\nR² = {below_voltage_r2:.4f}, RMSE = {below_voltage_rmse:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Voltages for Phases -1 to 1
    plt.subplot(3, 1, 2)
    plt.scatter(between_neg1_and_1_actual_voltages, between_neg1_and_1_predicted_voltages, 
               color='blue', label='Voltages for 0°', alpha=0.5)
    if len(between_neg1_and_1_actual_voltages) > 0:
        ideal_line = np.linspace(min(between_neg1_and_1_actual_voltages), max(between_neg1_and_1_actual_voltages), 100)
        plt.plot(ideal_line, ideal_line, 'r--')
    plt.xlabel('Actual Voltages (V)')
    plt.ylabel('Predicted Voltages (V)')
    plt.title(f'Voltages Corresponding to Phases 0° \nR² = {between_voltage_r2:.4f}, RMSE = {between_voltage_rmse:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Voltages for Phases > 1
    plt.subplot(3, 1, 3)
    plt.scatter(above_one_actual_voltages, above_one_predicted_voltages, 
               color='green', label='Voltages for +120°', alpha=0.5)
    if len(above_one_actual_voltages) > 0:
        ideal_line = np.linspace(min(above_one_actual_voltages), max(above_one_actual_voltages), 100)
        plt.plot(ideal_line, ideal_line, 'r--')
    plt.xlabel('Actual Voltages (V)')
    plt.ylabel('Predicted Voltages (V)')
    plt.title(f'Voltages Corresponding to +120° \nR² = {above_voltage_r2:.4f}, RMSE = {above_voltage_rmse:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_dir, "voltages.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory and avoid displaying
    plot_residuals(below_negative_one_actual_voltages, below_negative_one_predicted_voltages, output_dir, "V_below_minus_one.png", label="Voltages for Phases < -1")
    plot_residuals(between_neg1_and_1_actual_voltages, between_neg1_and_1_predicted_voltages, output_dir, "V_between_minus_one.png", label="Voltages for Phases -1 to 1")
    plot_residuals(above_one_actual_voltages, above_one_predicted_voltages, output_dir, "V_above_one.png", label="Voltages for Phases > 1")
    return Net, X_mean, X_std, voltage_min, voltage_range
def predict_single_row(model, input_row, X_mean, X_std, voltage_min, voltage_range):
    """
    Make a prediction for a single input row and measure prediction time.
    """
    # Normalize input row
    input_normalized = (input_row - X_mean) / X_std
    
    # Convert to tensor
    input_tensor = torch.tensor(input_normalized, dtype=torch.float32).unsqueeze(0)
    
    # Measure prediction time
    start_time = time.time()
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Calculate prediction time
    end_time = time.time()
    prediction_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Convert prediction to numpy
    prediction_np = prediction.numpy()[0]
    
    return {
        'prediction': prediction_np,
        'prediction_time_ms': prediction_time_ms
    }

def test_prediction_time(model, X, X_mean, X_std, voltage_min, voltage_range):
    """
    Test prediction time for multiple rows.
    """
    prediction_times = []
    
    for _ in range(10):
        # Select a random row each time
        random_index = np.random.randint(X.shape[0])
        random_input_row = X[random_index]
        
        # Predict and time
        result = predict_single_row(
            model, 
            random_input_row, 
            X_mean, 
            X_std, 
            voltage_min, 
            voltage_range
        )
        
        prediction_times.append(result['prediction_time_ms'])
    
    # Print prediction time statistics
    print("\nPrediction Time Statistics:")
    print(f"Average Prediction Time: {np.mean(prediction_times):.4f} ms")
    print(f"Standard Deviation: {np.std(prediction_times):.4f} ms")
    print(f"Minimum Time: {np.min(prediction_times):.8f} ms")
    print(f"Maximum Time: {np.max(prediction_times):.4f} ms")

def main():
    # Load data
    input_df = pd.read_csv("power_flow_nn/cluster_in_1_withp.csv")
    output_df = pd.read_csv("power_flow_nn/cluster_out_1.csv")
    # Train the model and get artifacts
    model, X_mean, X_std, voltage_min, voltage_range = train_and_evaluate_nn(
        input_df=input_df, 
        output_df=output_df,
        hidden_size=10
    )
    
    # Test prediction time
    test_prediction_time(
        model, 
        input_df.values, 
        X_mean, 
        X_std, 
        voltage_min, 
        voltage_range
    )


if __name__ == "__main__":
    main()