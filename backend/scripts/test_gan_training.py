import sys
import os
import time

# Add backend directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.gan_service import gan_service

def test_training():
    data_path = r"c:\Users\yasha\Gan project\backend\data\20260115071300\HKD0382601150713.EW"
    label = "knet_test"
    
    print(f"Starting training with file: {data_path}")
    success = gan_service.start_training(data_path, label, epochs=10, batch_size=4)
    
    if success:
        print("Training started successfully.")
        # Monitor for a few seconds
        for _ in range(5):
            status = gan_service.get_status()
            print(f"Status: {status}")
            if status['status'].startswith('failed'):
                print("Training Failed!")
                break
            time.sleep(1)
    else:
        print("Failed to start training.")

if __name__ == "__main__":
    test_training()
