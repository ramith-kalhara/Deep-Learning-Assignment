import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, UpSampling1D, Dense, Flatten, Reshape, Layer
from tensorflow.keras.callbacks import EarlyStopping

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAELossLayer(Layer):
    def call(self, inputs):
        orig_inputs, outputs, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(orig_inputs, outputs), axis=(1,)))
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        self.add_loss(reconstruction_loss + kl_loss)
        return outputs

def load_data(file_path):
    return np.loadtxt(file_path, delimiter=',', dtype=np.float32)

# Removed create_windows to use tf.keras.utils.timeseries_dataset_from_array instead

def create_label_windows(labels, window_length, step_size):
    windows = [1 if np.any(labels[i : i + window_length]) else 0 for i in range(0, len(labels) - window_length + 1, step_size)]
    return np.array(windows)

def build_vae(window_length, features, latent_dim=16):
    inputs = Input(shape=(window_length, features))
    x = Conv1D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = Input(shape=(latent_dim,))
    initial_length = window_length // 4
    x = Dense(initial_length * 64, activation="relu")(latent_inputs) 
    x = Reshape((initial_length, 64))(x)
    x = Conv1D(64, 3, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    x = UpSampling1D(2)(x)
    outputs = Conv1D(features, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")

    outputs = decoder(z)
    outputs = VAELossLayer()([inputs, outputs, z_mean, z_log_var])
    model = Model(inputs, outputs, name="VAE")
    model.compile(optimizer='adam')
    return model

def run_vae_pipeline():
    print("\n--- Starting VAE Pipeline ---")
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'archive', 'ServerMachineDataset')
    
    machine_id = 'machine-1-1.txt'
    window_length = 60
    
    print("Loading data...")
    train_data = load_data(os.path.join(data_dir, 'train', machine_id))
    test_data = load_data(os.path.join(data_dir, 'test', machine_id))
    test_labels = load_data(os.path.join(data_dir, 'test_label', machine_id))
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    split_idx = int(len(train_scaled) * 0.85)
    train_data_split = train_scaled[:split_idx]
    val_data_split = train_scaled[split_idx:]
    
    train_dataset = tf.keras.utils.timeseries_dataset_from_array(train_data_split, None, sequence_length=window_length, sequence_stride=1, batch_size=32)
    val_dataset = tf.keras.utils.timeseries_dataset_from_array(val_data_split, None, sequence_length=window_length, sequence_stride=1, batch_size=32)
    test_dataset = tf.keras.utils.timeseries_dataset_from_array(test_scaled, None, sequence_length=window_length, sequence_stride=1, batch_size=32)
    
    y_test = create_label_windows(test_labels, window_length, 1)
    
    model = build_vae(window_length, 38)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Training VAE...")
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping], verbose=1)
    
    print("Evaluating VAE...")
    val_mae = []
    for batch in val_dataset:
        preds = model.predict_on_batch(batch)
        mae = np.mean(np.abs(batch.numpy() - preds), axis=(1,2))
        val_mae.extend(mae)
    val_mae = np.array(val_mae)
    threshold = np.mean(val_mae) + 3 * np.std(val_mae)
    
    test_mae = []
    for batch in test_dataset:
        preds = model.predict_on_batch(batch)
        mae = np.mean(np.abs(batch.numpy() - preds), axis=(1,2))
        test_mae.extend(mae)
    test_mae = np.array(test_mae)
    
    preds = (test_mae > threshold).astype(int)
    
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auroc = roc_auc_score(y_test, test_mae)
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    np.savez(os.path.join(save_dir, 'vae_results.npz'), test_mae=test_mae, y_test=y_test, metrics=np.array([precision, recall, f1, auroc]))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('VAE Learning Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, test_mae)
    plt.plot(fpr, tpr, label=f'AUC={auroc:.2f}')
    plt.title('VAE ROC')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'VAE_chart.png'))
    plt.close()
    print(f"VAE Pipeline Finished. Results saved in {save_dir}")

if __name__ == "__main__":
    run_vae_pipeline()
