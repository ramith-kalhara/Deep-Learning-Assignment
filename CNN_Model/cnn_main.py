import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping

def load_data(file_path):
    return np.loadtxt(file_path, delimiter=',')

def create_windows(data, window_length, step_size):
    windows = [data[i : i + window_length] for i in range(0, len(data) - window_length + 1, step_size)]
    return np.array(windows)

def create_label_windows(labels, window_length, step_size):
    windows = [1 if np.any(labels[i : i + window_length]) else 0 for i in range(0, len(labels) - window_length + 1, step_size)]
    return np.array(windows)

def build_cnn(window_length, features):
    input_layer = Input(shape=(window_length, features))
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling1D(size=2)(x)
    decoded = Conv1D(filters=features, kernel_size=3, activation='sigmoid', padding='same')(x)
    model = Model(input_layer, decoded, name="CNN_Autoencoder")
    model.compile(optimizer='adam', loss='mae')
    return model

def run_cnn_pipeline():
    print("\n--- Starting CNN Pipeline ---")
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
    
    X_train = create_windows(train_scaled, window_length, 1)
    X_test = create_windows(test_scaled, window_length, 1)
    y_test = create_label_windows(test_labels, window_length, 1)
    
    model = build_cnn(window_length, 38)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    print("Training CNN...")
    history = model.fit(X_train, X_train, epochs=50, batch_size=64, validation_split=0.15, callbacks=[early_stopping], verbose=1)
    
    print("Evaluating CNN...")
    X_val = X_train[int(len(X_train)*0.85):]
    val_mae = np.mean(np.abs(X_val - model.predict(X_val)), axis=(1,2))
    threshold = np.mean(val_mae) + 3 * np.std(val_mae)
    
    test_mae = np.mean(np.abs(X_test - model.predict(X_test)), axis=(1,2))
    preds = (test_mae > threshold).astype(int)
    
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auroc = roc_auc_score(y_test, test_mae)
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    np.savez(os.path.join(save_dir, 'cnn_results.npz'), test_mae=test_mae, y_test=y_test, metrics=np.array([precision, recall, f1, auroc]))
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('CNN Learning Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, test_mae)
    plt.plot(fpr, tpr, label=f'AUC={auroc:.2f}')
    plt.title('CNN ROC')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'CNN_chart.png'))
    plt.close()
    print(f"CNN Pipeline Finished. Results saved in {save_dir}")

if __name__ == "__main__":
    run_cnn_pipeline()
