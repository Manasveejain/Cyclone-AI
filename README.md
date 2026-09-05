# 🌀 Cyclone AI

AI-powered cyclone intensity prediction and track visualization for the **North Indian Ocean** basin, built for a hackathon.

It combines a **ResNet-18 deep learning model** for satellite image inference with a **FastAPI backend** and a **React + Leaflet frontend** powered by real IBTrACS storm data.

---

## Features

- **Dashboard** — basin-wide stats, top storms by wind speed (bar chart), intensity distribution (pie chart), and a full storm table
- **Cyclone Tracks** — interactive dark-mode map with per-point intensity color coding, storm track polyline, and a wind-speed-over-time chart
- **AI Predict** — upload any satellite image to get a wind speed estimate and intensity category via a ResNet-18 CNN; also supports one-click inference on bundled mock frames

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch · ResNet-18 (fine-tuned regression head) |
| Backend | FastAPI · Uvicorn |
| Data | IBTrACS v04r01 (North Indian Ocean, 2020+) |
| Frontend | React 18 · Vite · Leaflet / react-leaflet · Recharts |

---

## Project Structure

```
Cyclone-AI/
├── backend/
│   └── main.py              # FastAPI app — all REST endpoints
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Tab navigation shell
│   │   └── pages/
│   │       ├── Dashboard.jsx # Stats cards + charts + storm table
│   │       ├── TrackView.jsx # Interactive map + wind chart
│   │       └── Predict.jsx   # Image upload + AI inference UI
│   ├── package.json
│   └── vite.config.js
├── model.py                 # CNN + LSTM definitions, IBTrACS ingestion, inference pipeline
├── demo_cyclones.csv        # Pre-filtered NI basin data (generated from IBTrACS)
├── mock_cyclone_frames/     # Sample satellite images for testing (c1–c5.jpeg)
├── requirements.txt
└── ibtracs.ALL.list.v04r01.csv  # ⚠ NOT in repo (316 MB) — download separately
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### 1. Clone the repo

```bash
git clone https://github.com/Manasveejain/Cyclone-AI.git
cd Cyclone-AI
```

### 2. (Optional) Download IBTrACS data

The master dataset is too large for GitHub. If you want to regenerate `demo_cyclones.csv`:

1. Download `ibtracs.ALL.list.v04r01.csv` from [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
2. Place it in the project root
3. Run:
   ```bash
   python model.py
   ```
   This filters the North Indian Ocean basin (2020+) and writes `demo_cyclones.csv`.

> `demo_cyclones.csv` is already committed, so this step is only needed if you want fresher or different data.

### 3. Set up the Python backend

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn backend.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 4. Set up the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/cyclones` | List all storms with basic stats |
| `GET` | `/api/cyclones/{sid}/track` | Full track data (lat/lon/wind) for a storm |
| `GET` | `/api/stats` | Basin-wide aggregate statistics |
| `POST` | `/api/predict` | Upload satellite image → wind speed + category |
| `GET` | `/api/mock-frames` | List bundled test frames |
| `GET` | `/api/mock-frames/{filename}/predict` | Run inference on a mock frame |

---

## Intensity Scale

| Category | Wind Speed |
|---|---|
| Tropical Depression | < 34 knots |
| Tropical Storm | 34 – 47 knots |
| Severe Cyclonic Storm | 48 – 63 knots |
| Very Severe Cyclonic Storm | 64 – 95 knots |
| Extremely Severe Cyclonic Storm | 96 – 119 knots |
| Super Cyclonic Storm | ≥ 120 knots |

---

## ML Architecture

**CycloneIntensityCNN** — ResNet-18 backbone with a custom regression head:
```
ResNet-18 (pretrained) → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 1)
```
Output: continuous wind speed in knots.

**CycloneTrajectoryLSTM** — 2-layer LSTM for 24-hour track forecasting:
```
Input(lat, lon, wind) → LSTM(64 hidden, 2 layers) → Linear(64, 2) → (ΔLat, ΔLon)
```

> The model runs without pre-trained weights (random init) for the hackathon demo. Swap in your own `.pth` file via the `model_weights_path` argument in `run_inference_pipeline()`.

---

## License

MIT
