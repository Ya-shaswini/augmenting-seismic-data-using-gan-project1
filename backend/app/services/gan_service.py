import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from app.models.gan import Generator, Discriminator
from app.services.data_parser import KNetParser
from app.models.denoiser import SeismicDenoiser
from app.services.prediction_service import prediction_service
import os
import threading

class GanService:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = 100
        self.output_length = 1024
        self.models_dir = "models_weights"
        os.makedirs(self.models_dir, exist_ok=True)
        self.is_training = False
        self.training_progress = {"epoch": 0, "total_epochs": 0, "d_loss": 0.0, "g_loss": 0.0, "mse_loss": 0.0, "status": "idle", "label": None, "type": "gan"}
        
        # Initialize Denoiser
        self.denoiser = SeismicDenoiser(input_length=self.output_length).to(self.device)
        # Note: In a real app, we'd load weights here. For this demo, we'll assume it's "ready".
    
    def _get_model_paths(self, label):
        return os.path.join(self.models_dir, f"{label}_generator.pth"), os.path.join(self.models_dir, f"{label}_discriminator.pth")

    def _train_loop(self, data_tensor, epochs, batch_size, lr, n_critic, clip_value, label):
        generator = Generator(noise_dim=self.latent_dim, output_length=self.output_length).to(self.device)
        discriminator = Discriminator(input_length=self.output_length).to(self.device)
        
        g_path, d_path = self._get_model_paths(label)
        
        # Load existing if available to continue training? Or start fresh?
        # Let's assume start fresh for now unless user wants otherwise. 
        # But for robustness, let's just train fresh or maybe add a flag later.
        
        try:
            self.training_progress["status"] = "running"
            self.training_progress["total_epochs"] = epochs
            self.training_progress["label"] = label
            
            dataset = TensorDataset(data_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
            optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
            optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
            
            for epoch in range(epochs):
                d_loss_val = 0
                g_loss_val = 0
                
                for i, (imgs,) in enumerate(dataloader):
                    
                    real_imgs = imgs.to(self.device)
                    
                    # Train Discriminator
                    optimizer_D.zero_grad()
                    z = torch.randn(imgs.shape[0], self.latent_dim).to(self.device)
                    fake_imgs = generator(z).detach()
                    loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
                    loss_D.backward()
                    optimizer_D.step()
                    
                    for p in discriminator.parameters():
                        p.data.clamp_(-clip_value, clip_value)
                        
                    if i % n_critic == 0:
                        # Train Generator
                        optimizer_G.zero_grad()
                        gen_imgs = generator(z)
                        loss_G = -torch.mean(discriminator(gen_imgs))
                        loss_G.backward()
                        optimizer_G.step()
                        g_loss_val = loss_G.item()
                    
                    d_loss_val = loss_D.item()
                
                self.training_progress["epoch"] = epoch + 1
                self.training_progress["d_loss"] = d_loss_val
                self.training_progress["g_loss"] = g_loss_val
                
                if epoch % 100 == 0:
                    torch.save(generator.state_dict(), g_path)
                    torch.save(discriminator.state_dict(), d_path)
            
            torch.save(generator.state_dict(), g_path)
            torch.save(discriminator.state_dict(), d_path)
            self.training_progress["status"] = "completed"
            
        except Exception as e:
            print(f"Training error: {e}")
            self.training_progress["status"] = f"failed: {str(e)}"
        finally:
            self.is_training = False

    def _train_denoiser_loop(self, data_tensor, epochs, batch_size, lr, label):
        self.denoiser.train()
        optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        denoiser_path = os.path.join(self.models_dir, "denoiser.pth")
        
        try:
            self.training_progress["status"] = "running"
            self.training_progress["total_epochs"] = epochs
            self.training_progress["label"] = label
            self.training_progress["type"] = "denoiser"
            
            dataset = TensorDataset(data_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
            for epoch in range(epochs):
                running_loss = 0.0
                for i, (clean_imgs,) in enumerate(dataloader):
                    clean_imgs = clean_imgs.to(self.device)
                    
                    # Create noisy version: Clean + Noise
                    # Noise level can be adjusted, let's use a random scale for robustness
                    noise_level = np.random.uniform(0.05, 0.2)
                    noise = torch.randn_like(clean_imgs) * noise_level
                    noisy_imgs = clean_imgs + noise
                    noisy_imgs = torch.clamp(noisy_imgs, -1, 1) # Keep in range
                    
                    optimizer.zero_grad()
                    outputs = self.denoiser(noisy_imgs)
                    loss = criterion(outputs, clean_imgs)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(dataloader)
                self.training_progress["epoch"] = epoch + 1
                self.training_progress["mse_loss"] = avg_loss
                
                if epoch % 50 == 0:
                    torch.save(self.denoiser.state_dict(), denoiser_path)
            
            torch.save(self.denoiser.state_dict(), denoiser_path)
            self.training_progress["status"] = "completed"
            
        except Exception as e:
            print(f"Denoiser training error: {e}")
            self.training_progress["status"] = f"failed: {str(e)}"
        finally:
            self.is_training = False

    def _preprocess_data(self, raw_data):
        """
        Slices data into chunks of 1024 and normalizes to [-1, 1]
        """
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(raw_data))
        if max_val == 0:
            return np.array([])
            
        normalized = np.array(raw_data) / max_val
        
        # Slice
        slices = []
        for i in range(0, len(normalized) - self.output_length, self.output_length): # Non-overlapping for now to maximize distinct samples? Or overlapping? Let's do stride=output_length/2
            stride = int(self.output_length / 2)
            segment = normalized[i : i + self.output_length]
            if len(segment) == self.output_length:
                slices.append(segment)
                
        # Sliding window with overlap
        processed_data = []
        stride = 256 # Higher overlap for more data
        for i in range(0, len(normalized) - self.output_length, stride):
            segment = normalized[i : i + self.output_length]
            processed_data.append(segment)
            
        return np.array(processed_data)

    def start_training(self, data, label, epochs=1000, batch_size=64):
        if self.is_training:
            return False
            
        # Parse if data is a filepath
        data = self._maybe_parse_data(data)
        if data is None:
            return False
        
        self.is_training = True
        data_tensor = torch.FloatTensor(data)
        if len(data_tensor.shape) == 2:
            data_tensor = data_tensor.unsqueeze(1)
            
        # Check if we have enough data for batch size
        real_batch_size = min(batch_size, len(data_tensor))
        
        thread = threading.Thread(target=self._train_loop, args=(data_tensor, epochs, real_batch_size, 0.00005, 5, 0.01, label))
        thread.start()
        return True

    def _maybe_parse_data(self, data):
        if isinstance(data, str) and os.path.isfile(data):
            try:
                parser = KNetParser()
                raw_values = parser.parse_file(data)
                processed_data = self._preprocess_data(raw_values)
                if len(processed_data) == 0:
                    print("No sufficient data after preprocessing")
                    return None
                return processed_data
            except Exception as e:
                print(f"Error parsing data: {e}")
                return None
        return data

    def start_denoiser_training(self, data, label, epochs=500, batch_size=32, lr=0.001):
        if self.is_training:
            return False
            
        data = self._maybe_parse_data(data)
        if data is None:
            return False
            
        self.is_training = True
        data_tensor = torch.FloatTensor(data)
        if len(data_tensor.shape) == 2:
            data_tensor = data_tensor.unsqueeze(1)
            
        real_batch_size = min(batch_size, len(data_tensor))
        
        thread = threading.Thread(target=self._train_denoiser_loop, args=(data_tensor, epochs, real_batch_size, lr, label))
        thread.start()
        return True

    def generate(self, num_samples, label):
        g_path, _ = self._get_model_paths(label)
        if not os.path.exists(g_path):
            raise ValueError(f"Model for label '{label}' not found.")
            
        generator = Generator(noise_dim=self.latent_dim, output_length=self.output_length).to(self.device)
        generator.load_state_dict(torch.load(g_path, map_location=self.device))
        generator.eval()
        
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        with torch.no_grad():
            gen_imgs = generator(z)
        return gen_imgs.squeeze(1).cpu().numpy().tolist()

    def get_status(self):
        return self.training_progress

    def denoise_and_predict(self, raw_signal):
        """
        Takes a raw signal, denoises it using the CNN Autoencoder,
        and runs earthquake prediction metrics.
        """
        # Convert to tensor
        signal_tensor = torch.FloatTensor(raw_signal).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Denoise
        self.denoiser.eval()
        with torch.no_grad():
            denoised_tensor = self.denoiser(signal_tensor)
        
        denoised_signal = denoised_tensor.squeeze().cpu().numpy()
        
        # Calculate metrics
        snr_improvement = prediction_service.calculate_snr(np.array(raw_signal), denoised_signal)
        prediction = prediction_service.predict_p_wave(denoised_signal)
        
        return {
            "denoised_data": denoised_signal.astype(float).tolist(),
            "snr_improvement": float(snr_improvement),
            "prediction": {
                "detected": bool(prediction['detected']),
                "index": int(prediction['index']),
                "confidence": float(prediction['confidence'])
            }
        }

gan_service = GanService()
