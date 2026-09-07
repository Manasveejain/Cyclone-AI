# 🌀 Cyclone AI

AI-powered cyclone intensity prediction and track visualization for the **North Indian Ocean** basin, built for a hackathon.

Combines a **ResNet-18 deep learning model** trained on 4-channel satellite data with a **FastAPI backend** and a **React + Leaflet frontend** driven by real IBTrACS storm data.

---

## Features

- **Dashboard** — basin-wide stats, top storms by wind speed (bar chart), intensity distribution (pie chart), and a full storm table
- **Cyclone Tracks** — interactive dark-mode map with per-point intensity color coding, storm track polyline, and a wind-speed-over-time chart
- **AI Predict** — upload any satellite image to get a wind speed estimate and intensity category via a trained ResNet-18 CNN; also supports one-click inference on bundled mock frames

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | PyTorch · ResNet-18 (4-channel regression head) |
| Backend | FastAPI · Uvicorn |
| Data | IBTrACS v04r01 (North Indian Ocean, 2000+) |
| Frontend | React 18 · Vite · Leaflet / react-leaflet · Recharts |

---

## Project Structure

```
Cyclone-AI/
├── backend/
│   └── main.py                      # FastAPI app — all REST endpoints
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Tab navigation shell
│   │   ├── colors.js                # Shared intensity color palette
│   │   └── pages/
│   │       ├── Dashboard.jsx        # Stats cards + charts + storm table
│   │       ├── TrackView.jsx        # Interactive map + wind chart
│   │       └── Predict.jsx          # Image upload + AI inference UI
│   ├── package.json
│   └── vite.config.js
├── model.py                         # CNN definition, IBTrACS ingestion, training + inference pipeline
├── demo_cyclones.csv                # Pre-filtered NI basin data (generated from IBTrACS)
├── mock_cyclone_frames/             # Sample satellite images for testing
├── outputs/
│   ├── cyclone_cnn_best.pth         # Trained model weights (4-channel ResNet-18, 42.97 MB)
│   └── predicted_vs_actual.png      # Validation scatter plot
├── requirements.txt
└── ibtracs.NI.list.v04r01.csv       # ⚠ NOT in repo — download separately
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

The master dataset is not included in the repo. If you need to regenerate `demo_cyclones.csv`:

1. Download `ibtracs.NI.list.v04r01.csv` from [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) (North Indian Ocean basin file)
2. Place it in the project root
3. Run:
   ```bash
   python -c "from model import prepare_ibtracs_dataset; prepare_ibtracs_dataset()"
   ```
   This filters the NI basin (year 2000+) and writes `demo_cyclones.csv`.

> `demo_cyclones.csv` is already committed, so this step is only needed if you want to refresh the data.

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

API live at `http://localhost:8000` · Interactive docs at `http://localhost:8000/docs`

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
ResNet-18 (pretrained ImageNet, 4-channel input) → GlobalAvgPool
  → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 1)
```

Output: continuous wind speed in knots.

The first conv layer is extended from 3 → 4 channels to accept multi-source satellite input:

| Channel | Source |
|---|---|
| 0 | Infrared (IR) |
| 1 | Water Vapor (WV) |
| 2 | Passive Microwave (PMW) |
| 3 | Visible (VIS) |

When running inference on standard RGB satellite images, the backend automatically converts the 3-channel image to 4-channel by duplicating the red channel as a proxy for the PMW band.

**Saved weights:** `outputs/cyclone_cnn_best.pth` — 124 layers, 42.97 MB, trained with `in_channels=4`.

---

## Training (Optional)

To train on synthetic data (no downloads needed):

```bash
python model.py --synthetic --epochs 10
```

To train on real TCIR data:

```bash
python model.py --data_path TCIR-CPAC_IO_SH.h5 --epochs 30
```

To limit samples for a quick CPU run:

```bash
python model.py --data_path TCIR-CPAC_IO_SH.h5 --max_samples 500 --epochs 5
```

Weights are saved to `outputs/cyclone_cnn_best.pth` and a prediction plot to `outputs/predicted_vs_actual.png`.

### Training on Kaggle (free GPU)

1. Go to [kaggle.com/datasets](https://kaggle.com/datasets) → New Dataset → upload `TCIR-CPAC_IO_SH.h5`
2. Create a new notebook, attach the dataset, enable GPU accelerator
3. Run:
   ```bash
   !python model.py --data_path /kaggle/input/your-dataset/TCIR-CPAC_IO_SH.h5 --epochs 30
   ```
4. Download `outputs/cyclone_cnn_best.pth` and place it back in your local `outputs/` folder

---

## License

MIT
