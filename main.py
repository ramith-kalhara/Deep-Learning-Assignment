import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def main():
    print("   Starting Master Pipeline Execution    ")
    
    # 1. Run independent model scripts in their respective folders
    scripts = [
        ("CNN_Model", "cnn_main.py"),
        ("LSTM_Model", "lstm_main.py"),
        ("VAE_Model", "vae_main.py")
    ]
    
    for folder, script in scripts:
        print(f" Executing {script} in {folder}...")
        # Run the script as a separate process to guarantee 100% independence
        subprocess.run(["python", script], cwd=folder)
        print(f"<<< {script} finished.\n")
        
    print(" All independent models finished. Combining Results")
    
    # 2. Gather saved results from the folders
    results_files = {
        "CNN_AE": "CNN_Model/cnn_results.npz",
        "LSTM_AE": "LSTM_Model/lstm_results.npz",
        "VAE": "VAE_Model/vae_results.npz"
    }
    
    print("\n Assignment Reporting Metrics ")
    print(f"{'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AUROC':<10}")
    print("-" * 65)
    
    plt.figure(figsize=(10, 6))
    
    # 3. Read arrays and plot combined ROC curve
    for name, path in results_files.items():
        if os.path.exists(path):
            data = np.load(path)
            test_mae = data['test_mae']
            y_test = data['y_test']
            metrics = data['metrics']
            
            # Print table row
            print(f"{name:<15} | {metrics[0]:<10.4f} | {metrics[1]:<10.4f} | {metrics[2]:<10.4f} | {metrics[3]:<10.4f}")
            
            # Add to combined plot
            fpr, tpr, _ = roc_curve(y_test, test_mae)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics[3]:.2f})')
        else:
            print(f"Error: Results file {path} not found. Did the model run successfully?")
            
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('Combined ROC Curves Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('combined_evaluation_plots.png', dpi=300)
    print("\nSaved combined chart: combined_evaluation_plots.png")
    print("Master Pipeline Complete!")

if __name__ == "__main__":
    main()
