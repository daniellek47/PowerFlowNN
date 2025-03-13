# Power System Voltage and Power Flow Prediction using Neural Networks and Clustering

## Project Overview
This research project develops an advanced machine learning approach for predicting voltage magnitudes and power flows in electrical power systems using neural networks and intelligent clustering techniques.
Main training functions and time measurement in [NN_clusters.py](NN_cluster.py) optimal hidden layer sizes: Main network: 18, cluster 5: 10, cluster 1: 12 
Clusters Management, Datasets manipulation in [clusters.py](clusters.py)
Another experimental notebook in [nn_cluster.ipynb](nn_cluster.ipynb) - not the final product

## Key Features
- Multi-phase power system voltage prediction
- Advanced neural network architecture
- Intelligent clustering of power system nodes
- Complex power flow calculations

## Methodology

### Data Preprocessing
- Voltage magnitude and phase processing
- Admittance matrix handling
- Feature normalization techniques

### Neural Network Architecture
- Custom neural network design
- Dropout and regularization techniques
- Multiple hidden layer configuration

### Clustering Approach
- Node clustering based on electrical characteristics
- Identification of electrically similar groups
- Inter-cluster power flow analysis

## Key Graphs and Visualizations

### 1. Training Loss Curves
![Training Loss](training_validation_loss.png)
- Demonstrates model learning progress
- Shows convergence of training and validation loss

### 2. Voltage Prediction Scatter Plot
![Voltage Prediction](voltages.png)
- Actual vs. Predicted voltage magnitudes
- R² and RMSE metrics

### 3. Power Flow Visualization
![Power Flow](path/to/power_flow_visualization.png)
- Spatial representation of power flow
- Different colors for different clusters



## Performance Metrics
- Mean Squared Error (MSE)
- R² Score
- Mean Absolute Error (MAE)

## Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage
```bash
# NN_cluster.py main Usage:
    # Load data
    input_df = pd.read_csv("power_flow_nn/data_in.csv")
    output_df = pd.read_csv("power_flow_nn/data_out.csv")
    # Train the model and evaluate
    model, X_mean, X_std, voltage_min, voltage_range = train_and_evaluate_nn(
        input_df=input_df, 
        output_df=output_df,
        hidden_size=18
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
# clusters.py main usage:
    clusters = parse_cluster_info(document_content, load_indices)
    input_df = pd.read_csv("New Data\data_in.csv")
    output_df = pd.read_csv("New Data\data_out.csv")
    admittance_df = pd.read_csv('New Data\ieee123_y_matrix.csv', index_col=0, header=0)
    admittance_df = admittance_df.applymap(parse_complex)

    # the next function computes power in a given node using Y matrix
    power_df = compute_node_power(
            voltages_df=output_df,
             admittance_row=admittance_row,         target_node_index=230,  # Specify the index for Vi
            node_names=node_names.tolist(),
            neighbors=['101.1', '101.2', '101.3']
        )
```


## Future Work

- Incorporate more advanced clustering algorithms
- Explore ensemble neural network approaches
- Develop real-time power system prediction models
