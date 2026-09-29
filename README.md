# Server Machine Dataset - Anomaly Detection (SLIIT SE4050)

This repository contains an Unsupervised Deep Learning project for time-series anomaly detection using the Server Machine Dataset (SMD).

## Group Contributions
- **Member 1:** 1D-CNN Autoencoder (`cnn_model.py`, `run_cnn.py`)
- **Member 2:** LSTM Autoencoder (`lstm_model.py`, `run_lstm.py`)
- **Member 3:** Variational Autoencoder (`vae_model.py`, `run_vae.py`)

## Architecture
The project follows an Object-Oriented Programming (OOP) architecture. 
Core components are shared to prevent code duplication:
- `data_loader.py`: Handles data ingestion and sliding windows.
- `trainer.py`, `evaluator.py`, `visualizer.py`: Handle training loops, evaluation metrics, and plotting.

## Setup Instructions
1. Clone the repository.
2. Install Python 3.8 or higher.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Instructions
### 1. Exploratory Data Analysis (EDA)
To generate raw data plots for the assignment report:
```bash
python eda.py
```

### 2. Individual Model Execution
Team members can run their assigned models independently without causing GitHub conflicts:
```bash
python run_cnn.py
python run_lstm.py
python run_vae.py
```

### 3. Combined Pipeline Execution
To train all three models sequentially and generate a combined ROC comparison chart:
```bash
python main.py
```
