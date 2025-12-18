# Implementation Plan - Augmenting Seismic Data Using GAN

## Project Overview
A full-stack application to ingest, enhance, and monitor seismic data from low-cost MEMS sensors. 
**Key Features**: Real-time streaming, GAN-based denoising, WhatsApp-style interface for alerts/sharing, and comprehensive dashboards.

## Technology Stack
- **Frontend**: Next.js (React), Vanilla CSS (Custom Glassmorphism), Chart.js/Recharts (for Waveforms).
- **Backend**: FastAPI (Python), WebSockets (Real-time), SQLite (Database), PyTorch/TensorFlow (GAN Model placeholder).
- **Communication**: REST API + WebSockets.

## Architecture
```
/
├── backend/                # FastAPI Server
│   ├── app/
│   │   ├── main.py         # Entry point
│   │   ├── api/            # REST Endpoints
│   │   ├── core/           # Config & Security
│   │   ├── models/         # DB Models
│   │   ├── schemas/        # Pydantic Schemas
│   │   ├── services/       # GAN, Signal Processing
│   │   └── websockets/     # Real-time connection manager
│   ├── requirements.txt
│   └── run.py
├── frontend/               # Next.js Client
│   ├── app/                # App Router
│   ├── components/         # UI Components
│   ├── lib/                # Utils (API client, WebSocket)
│   └── public/
└── README.md
```

## Step-by-Step Implementation

### Phase 1: Foundation & Backend Setup
1.  Initialize `backend` directory with FastAPI.
2.  Setup Virtual Environment and install dependencies (`fastapi`, `uvicorn`, `numpy`, `scipy`, `torch`, `websockets`, `passlib`, `jose`).
3.  Implement `GanService` (Mock/Placeholder initially) for signal denoising.
4.  Implement `SignalProcessingService` (Filtering, Spectrogram generation).
5.  Setup WebSocket manager for real-time data streaming.
6.  Create API endpoints for Auth (Login/Register) and Data retrieval.

### Phase 2: Frontend Setup & Design System
1.  Initialize `frontend` with `npx create-next-app`.
2.  Define `globals.css` with **Glassmorphism** variables (blurs, translucent backgrounds, neon accents).
3.  Create Layouts:
    *   `AuthLayout`: For Login/Signup.
    *   `DashboardLayout`: Sidebar + Main Content (Chat style).

### Phase 3: Core Features Implementation
1.  **Real-time Chat/Stream Interface**:
    *   Left panel: Sensor list / Contacts (WhatsApp style).
    *   Right panel: Live feed (Stream messages + Waveforms).
    *   Implement WebSocket hook in React.
2.  **Dashboards**:
    *   Before/After Waveform components.
    *   Spectrogram visualization.
    *   Metrics display (SNR, Pick times).
3.  **GAN Integration** (Completed):
    *   Connect Frontend "Enhance" button to Backend `enrich_signal` endpoint.
    *   Display results.
    *   Added "Stop Reading" (Connection Toggle) and "Export" (CSV Download) features.

### Phase 4: Polish & Deployment Prep
1.  Add Micro-animations (hover effects, message pop-ins).
2.  Ensure Responsive Design (Mobile friendly).
3.  Finalize "Academic Demo" mode (Pre-loaded datasets if sensors aren't active).
