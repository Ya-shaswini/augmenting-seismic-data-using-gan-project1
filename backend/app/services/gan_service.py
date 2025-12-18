import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from app.models.gan import Generator, Discriminator
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
        self.training_progress = {"epoch": 0, "total_epochs": 0, "d_loss": 0.0, "g_loss": 0.0, "status": "idle", "label": None}
    
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

    def start_training(self, data, label, epochs=1000, batch_size=64):
        if self.is_training:
            return False
        
        self.is_training = True
        data_tensor = torch.FloatTensor(data)
        if len(data_tensor.shape) == 2:
            data_tensor = data_tensor.unsqueeze(1)
        
        thread = threading.Thread(target=self._train_loop, args=(data_tensor, epochs, batch_size, 0.00005, 5, 0.01, label))
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

gan_service = GanService()
