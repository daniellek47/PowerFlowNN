# Power System Voltage and Power Flow Prediction using Neural Networks and Clustering

## Project Overview
This research project develops an advanced machine learning approach for predicting voltage magnitudes and power flows in electrical power systems using neural networks and intelligent clustering techniques.

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
![Training Loss](PowerFlowNN/training_validation_loss.png)
- Demonstrates model learning progress
- Shows convergence of training and validation loss

### 2. Voltage Prediction Scatter Plot
![Voltage Prediction](PowerFlowNN/voltages.png)
- Actual vs. Predicted voltage magnitudes
- R² and RMSE metrics

### 3. Power Flow Visualization
![Power Flow](path/to/power_flow_visualization.png)
- Spatial representation of power flow
- Different colors for different clusters

### 4. Node Clustering Diagram
![Node Clustering](path/to/node_clustering.png)
- Visualization of node clusters
- Electrical connectivity representation

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

## Installation
```bash
git clone https://github.com/yourusername/power-system-ml.git
cd power-system-ml
pip install -r requirements.txt

# Example code snippet
from power_flow_prediction import predict_voltages

results = predict_voltages(input_data)
```


## Future Work

- Incorporate more advanced clustering algorithms
- Explore ensemble neural network approaches
- Develop real-time power system prediction models
