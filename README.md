# Augmenting Seismic Data Using GAN - Project

This is a full-stack application for augmenting and monitoring seismic data using a Generative Adversarial Network (GAN).

## Scientific Concept: The GAN
The core of this project is a **Generative Adversarial Network (GAN)** designed to solve the problem of data scarcity in earthquake engineering. Real high-magnitude earthquake records are rare, making it hard to train AI models for damage release.
*   **The Generator**: An AI model that learns to create "fake" seismic waveforms. It starts by producing noise but learns to structure it into realistic earthquake signals.
*   **The Discriminator**: A second AI model that acts as a judge. It looks at real data (K-NET) and the generator's fake data, trying to tell them apart.
*   **The Result**: As they compete, the Generator becomes so skilled that it produces synthetic earthquake data indistinguishable from real records, effectively giving us an "infinite" dataset for training structural health monitoring systems.

## Data Utilized: K-NET
We utilize strong-motion acceleration data from **K-NET (Kyoshin Network)**, a dense seismograph network in Japan.
*   **Role**: This data serves as the **Ground Truth**.
*   **Why it helps**: The Discriminator needs to know what a "Real Earthquake" looks like to grade the Generator. By feeding it high-quality, high-magnitude records (like the M5.7 event we integrated), we teach the model the physics of ground motion (P-waves, S-waves, decay). 
*   **Integration**: We parse raw ASCII `.EW/NS/UD` files, slice them into 1024-point windows, and normalize them to train the model.

## Architecture
- **Frontend**: Next.js 15 (React), Glassmorphism UI, Real-time WebSockets.
- **Backend**: FastAPI (Python), WebSocket Manager, Simulated GAN Signal Processing.

## Prerequisites
- Node.js (v18+)
- Python (v3.8+)

## Setup and Running

### 1. Backend Setup
The backend handles the data streaming and signal processing.

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
*The backend runs on http://localhost:8000*

### 2. Frontend Setup
The frontend is the WhatsApp-style dashboard.

```bash
cd frontend
npm install
npm run dev
```
*The frontend runs on http://localhost:3000*

## Features Implemented
- **Real-time Streaming**: Connects to backend via WebSockets.
- **WhatsApp-style Interface**: Chat-based timeline for events and sensor statuses.
- **Seismic Visualization**: Live rendering of seismic waveforms using Canvas.
- **GAN Denoising & Prediction**: Functional 1D CNN Autoencoder for signal synthesis and P-wave detection.
- **Glassmorphism Design**: Premium UI with blur effects and neon accents.

## Usage
1. Open the Frontend.
2. Click the **Microphone Icon** in the chat input to simulate a Seismic Event (this triggers a backend fetch).
3. View the Waveform message.
4. Alerts will appear automatically from the mock backend stream.
