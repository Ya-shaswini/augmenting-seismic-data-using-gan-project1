# Augmenting Seismic Data Using GAN - Project

This is a full-stack application for augmenting and monitoring seismic data using a Generative Adversarial Network (GAN).

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
- **GAN Simulation**: "Enhance" and "Analyze" workflows mocked for demonstration.
- **Glassmorphism Design**: Premium UI with blur effects and neon accents.

## Usage
1. Open the Frontend.
2. Click the **Microphone Icon** in the chat input to simulate a Seismic Event (this triggers a backend fetch).
3. View the Waveform message.
4. Alerts will appear automatically from the mock backend stream.
