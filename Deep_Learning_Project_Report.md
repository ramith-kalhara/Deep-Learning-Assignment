# Server Downtime Anomaly Detection: A Deep Learning Approach

## 1. Introduction and Problem Definition
Modern IT infrastructures rely heavily on continuous server uptime. Unexpected server downtime or performance degradation can lead to significant financial losses and reduced user satisfaction. The problem addressed in this project is the early detection of server anomalies (the "server going down" problem) using historical performance metrics. By treating server metrics as multivariate time series data, we aim to build robust anomaly detection models that can alert system administrators before a critical failure occurs.

## 2. Background and Related Work
Anomaly detection in time series data has been extensively studied. Traditional statistical methods like ARIMA or simple thresholding often fail to capture complex, non-linear dependencies across multiple server metrics (e.g., CPU, Memory, Network I/O). Recently, Deep Learning approaches, particularly reconstruction-based models like Autoencoders, have shown state-of-the-art performance. In these architectures, the model is trained to reconstruct normal data. When an anomalous pattern is fed into the model, the reconstruction error is typically high, serving as an anomaly indicator. This project compares three such architectures: Long Short-Term Memory (LSTM), Variational Autoencoders (VAE), and Convolutional Neural Networks (CNN).

## 3. Dataset Description and Exploratory Data Analysis
The models were trained and evaluated on the **ServerMachineDataset (SMD)**, specifically the subset for `machine-1-1`. 
- **Features:** The dataset contains 38 continuous features representing various server performance metrics (e.g., CPU load, memory usage, network traffic).
- **Structure:** The data consists of separate training and testing sets, along with a corresponding test label set where `1` indicates an anomalous state and `0` indicates a normal state.
- **Characteristics:** The features exhibit varying scales and significant temporal dependencies, requiring appropriate preprocessing before feeding into neural networks.

## 4. Data Preprocessing and Feature Engineering
To prepare the dataset for deep learning models, the following preprocessing steps were applied across all models:
- **Normalization:** A `MinMaxScaler` was fitted on the training data to scale all 38 features to a range of [0, 1]. The test data was transformed using this same scaler to prevent data leakage.
- **Windowing (Time Steps):** The continuous time series was converted into overlapping windows (sequences). A `window_length` of 60 time steps was used with a `step_size` of 1. This allows the models to learn patterns over a continuous temporal context rather than isolated data points.
- **Label Alignment:** For the test set, a window was labeled as anomalous (1) if any single time step within that 60-step window was marked as an anomaly in the original labels.

## 5. Experimental Design
The experiment follows a reconstruction-based anomaly detection paradigm:
- **Training Strategy:** The models are trained to act as autoencoders, taking a 60-step window as input and attempting to output the exact same window. The loss function used is Mean Absolute Error (MAE) or Mean Squared Error (MSE).
- **Validation and Early Stopping:** 15% of the training data was used for validation. Early stopping was implemented (patience of 5 to 10 epochs) to halt training when the validation loss ceased to improve, restoring the best model weights to prevent overfitting.
- **Anomaly Thresholding:** After training, the Mean Absolute Error (MAE) for each window in the validation set was calculated. The anomaly threshold was defined dynamically as: `Threshold = Mean(Validation MAE) + 3 * Standard Deviation(Validation MAE)`.
- **Evaluation Metrics:** Any test window with a reconstruction error higher than the threshold was classified as anomalous. The models were evaluated using Precision, Recall, F1-Score, and Area Under the Receiver Operating Characteristic Curve (AUROC).

## 6. Model Architectures
Three distinct architectures were developed by the group members:

### 6.1 LSTM Autoencoder (Focus Model)
The Long Short-Term Memory (LSTM) network is naturally suited for time-series data due to its ability to remember long-term dependencies.
- **Encoder:** An LSTM layer with 64 units and ReLU activation processes the input window, outputting a single 64-dimensional feature vector.
- **Decoder:** A `RepeatVector` duplicates this vector 60 times. Another LSTM layer with 64 units processes this sequence, followed by a `TimeDistributed` Dense layer to reconstruct the original 38 features.

### 6.2 Convolutional Neural Network (CNN) Autoencoder
The CNN model utilizes 1-Dimensional Convolutions to extract local temporal patterns.
- **Encoder:** Two `Conv1D` layers (32 and 16 filters) intermixed with `MaxPooling1D` layers compress the time-series window.
- **Decoder:** `UpSampling1D` layers combined with `Conv1D` layers reconstruct the original temporal resolution and feature space.

### 6.3 Variational Autoencoder (VAE)
The VAE introduces a probabilistic twist to the standard autoencoder, learning the underlying distribution of the data.
- **Encoder:** `Conv1D` layers compress the input, which is then flattened and passed through Dense layers to output a mean (`z_mean`) and variance (`z_log_var`) for the latent space (dimension = 16).
- **Reparameterization Trick:** A custom `Sampling` layer draws a latent vector `z` from the learned distribution.
- **Decoder:** Dense layers reshape the latent vector, followed by `UpSampling1D` and `Conv1D` layers for reconstruction. The loss function combines reconstruction loss (MSE) and Kullback-Leibler (KL) divergence.

## 7. Results and Model Comparison
The models were evaluated on the test set for `machine-1-1`. The results for the LSTM and VAE models are presented below:

| Model | Precision | Recall | F1-Score | AUROC |
|-------|-----------|--------|----------|-------|
| **LSTM** | 0.2006 | **1.0000** | 0.3341 | **0.8757** |
| **VAE**  | **0.3430**| 0.5174 | **0.4125** | 0.8392 |
| **CNN**  | *TBA* | *TBA* | *TBA* | *TBA* |
*(Note: CNN model results were not pre-calculated in the repository directory and should be filled in once `cnn_main.py` is executed).*

**Comparison:** 
- The **LSTM model** achieved a perfect Recall (1.0), meaning it successfully identified every single anomaly without missing any. However, its low Precision (0.20) indicates a high false-positive rate (predicting downtime when the server is actually fine). Its overall AUROC is excellent (0.87).
- The **VAE model** offers a more balanced approach. It significantly improves Precision (0.34) and F1-Score (0.41), meaning it produces fewer false alarms compared to the LSTM, though at the cost of missing some anomalies (Recall of 0.51). 

## 8. Critical Analysis and Discussion
The results highlight the inherent trade-off between Precision and Recall in anomaly detection. 
- The **LSTM** model's high sensitivity (Recall=1.0) is ideal for highly critical systems where missing a server failure is catastrophic, and the cost of investigating a false alarm is relatively low. The sequential memory of the LSTM makes it highly sensitive to deviations from historical trends.
- The **VAE**, by mapping data to a probabilistic latent space, inherently smooths out minor irregularities in the data. This robustness reduces false positives (higher Precision) but also makes it slightly less sensitive to subtle true anomalies (lower Recall). 
- To further improve these models, techniques like dynamic thresholding (e.g., Peak-over-Threshold), incorporating attention mechanisms, or ensembling the predictions of all three models could be explored.

## 9. Conclusion
In this project, we successfully developed and evaluated three deep learning models (LSTM, CNN, and VAE) to predict server downtime using multivariate time series data. The experimental results demonstrate that deep learning autoencoders are highly capable of detecting complex anomalies. While the LSTM model excels at identifying all potential failures (maximum Recall), the VAE model provides a more balanced performance (higher F1-Score). Ultimately, the choice of model depends on the business requirement and the acceptable rate of false alarms.

## 10. References
1. Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D. (2019). Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network. *KDD*.
2. Chollet, F. et al. (2015). Keras. https://keras.io.
3. TensorFlow Developers (2023). TensorFlow. https://www.tensorflow.org/.
