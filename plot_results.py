import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_learning_curves(metrics_file, output_dir):
    # Read metrics
    df = pd.read_csv(metrics_file)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot dice curves
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_dice'], label='Train Dice')
    plt.plot(df['epoch'], df['val_dice'], label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Curves')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

def main():
    # Find the most recent log directory
    log_dirs = glob.glob('logs/encoder_decoder_*')
    if not log_dirs:
        print("No log directories found")
        return
    
    latest_dir = max(log_dirs, key=os.path.getctime)
    metrics_file = os.path.join(latest_dir, 'metrics.csv')
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    plot_learning_curves(metrics_file, latest_dir)
    print(f"Learning curves saved to {latest_dir}/learning_curves.png")

if __name__ == '__main__':
    main() 