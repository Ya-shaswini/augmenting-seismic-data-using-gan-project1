import sys
import os
import time
import torch

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.gan_service import gan_service

def test_denoiser_training():
    # Use one of the K-NET files as ground truth
    data_path = r"c:\Users\yasha\Gan project\backend\data\20260115071300\HKD0382601150713.EW"
    label = "denoiser_test"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Starting denoiser training with file: {data_path}")
    # Train for a small number of epochs for verification
    success = gan_service.start_denoiser_training(data_path, label, epochs=20, batch_size=4, lr=0.001)
    
    if success:
        print("Denoiser training started successfully.")
        # Monitor progress
        while True:
            status = gan_service.get_status()
            print(f"Epoch: {status['epoch']}/{status['total_epochs']} | MSE Loss: {status['mse_loss']:.6f} | Status: {status['status']}")
            
            if status['status'] == 'completed':
                print("\nTraining completed successfully!")
                break
            elif status['status'].startswith('failed'):
                print(f"\nTraining Failed: {status['status']}")
                break
            
            time.sleep(2)
            
        # Check if weight file exists
        weight_path = os.path.join("models_weights", "denoiser.pth")
        if os.path.exists(weight_path):
            print(f"Verified: Denoiser weights saved to {weight_path}")
        else:
            print(f"Error: Denoiser weights not found at {weight_path}")
    else:
        print("Failed to start denoiser training.")

if __name__ == "__main__":
    test_denoiser_training()
