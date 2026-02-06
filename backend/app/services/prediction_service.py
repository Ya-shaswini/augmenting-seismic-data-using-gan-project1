import numpy as np

class PredictionService:
    @staticmethod
    def calculate_snr(signal, denoised_signal):
        """
        Calculates Signal-to-Noise Ratio (SNR) improvement.
        Actually, for demo purposes, we can calculate SNR of a signal vs its noise.
        """
        noise = signal - denoised_signal
        signal_power = np.mean(denoised_signal**2)
        noise_power = np.mean(noise**2)
        if noise_power == 0:
            return 100.0
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    @staticmethod
    def predict_p_wave(signal, threshold=0.1, window_size=50):
        """
        Simple STA/LTA or threshold-based P-wave arrival prediction.
        Demonstrates that clean signals make detection more reliable.
        """
        # Simple threshold detection for demo
        abs_signal = np.abs(signal)
        for i in range(len(abs_signal)):
            if abs_signal[i] > threshold:
                return {
                    "detected": True,
                    "index": int(i),
                    "confidence": float(min(0.99, 0.5 + (float(abs_signal[i]) * 2)))
                }
        return {"detected": False, "index": -1, "confidence": 0.0}

    @staticmethod
    def generate_synthetic_earthquake(length=1024, magnitude=5.0, distance_km=50):
        """
        Generates a synthetic seismic signal with P and S wave characteristics.
        - magnitude: Affects peak amplitude (Logarithmic scaling simulation)
        - distance_km: Affects P-S arrival interval (v_s and v_p estimation)
        """
        t = np.arange(length)
        signal = np.zeros(length)
        
        # Physics approximation: 
        # v_p approx 6 km/s, v_s approx 3.5 km/s
        # arrival_time = distance / velocity
        # Scaling factor: assuming 100 samples = 1 second (100Hz)
        sample_rate = 100 
        p_arrival = int((distance_km / 6.0) * sample_rate)
        s_arrival = int((distance_km / 3.5) * sample_rate)
        
        # Clip to length
        p_arrival = min(p_arrival, length - 100)
        s_arrival = min(s_arrival, length - 50)
        if s_arrival <= p_arrival: s_arrival = p_arrival + 150

        # Magnitude scaling (Simplified: Mag 7 is ~10x Mag 6 amplitude in this demo)
        amp_scale = 10**(magnitude - 5.0) * 0.1
        
        # P-wave: High frequency, low amplitude
        p_wave = (amp_scale * 0.2) * np.sin(0.4 * t) * np.exp(-0.02 * (t - p_arrival))
        p_wave[t < p_arrival] = 0
        
        # S-wave: Lower frequency, high amplitude
        s_wave = amp_scale * np.sin(0.15 * t) * np.exp(-0.008 * (t - s_arrival))
        s_wave[t < s_arrival] = 0
        
        signal = p_wave + s_wave
        
        # Add some natural decay variability
        noise_floor = np.random.normal(0, 0.01, length)
        return signal + noise_floor

prediction_service = PredictionService()
