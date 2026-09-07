import os
import sys
import io
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Add parent dir so model.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import CycloneIntensityCNN, run_inference_pipeline

import torch
from torchvision import transforms

# ─── Model weights path ───────────────────────────────────────────────────────
WEIGHTS_PATH = Path(__file__).parent.parent / "outputs" / "cyclone_cnn_best.pth"

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cyclone AI API",
    description="Backend for North Indian Ocean cyclone intensity prediction and track analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "demo_cyclones.csv"
FRAMES_DIR = Path(__file__).parent.parent / "mock_cyclone_frames"

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Strip whitespace-only strings before numeric conversion (matches model.py cleaning)
    for col in ("WMO_WIND", "LAT", "LON"):
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────
def classify_intensity(wind_knots: float) -> dict:
    """Return category name and colour for a given wind speed.
    Colors kept in sync with frontend src/colors.js."""
    if wind_knots < 34:
        return {"category": "Tropical Depression",             "color": "#22d3ee"}
    elif wind_knots < 48:
        return {"category": "Tropical Storm",                  "color": "#86efac"}
    elif wind_knots < 64:
        return {"category": "Severe Cyclonic Storm",           "color": "#fde047"}
    elif wind_knots < 96:
        return {"category": "Very Severe Cyclonic Storm",      "color": "#fb923c"}
    elif wind_knots < 120:
        return {"category": "Extremely Severe Cyclonic Storm", "color": "#f87171"}
    else:
        return {"category": "Super Cyclonic Storm",            "color": "#e879f9"}


def run_model_on_image(image: Image.Image) -> dict:
    """Run the ResNet18 model on a PIL Image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycloneIntensityCNN(in_channels=4, pretrained=False)

    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=device))
    else:
        print(f"[WARNING] No weights found at {WEIGHTS_PATH} — using random weights (demo mode)")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Convert RGB to 4-channel by duplicating the red channel as a proxy
    # for the 4th satellite band (PMW), matching the saved weights shape [64, 4, 7, 7]
    rgb_tensor = transform(image.convert("RGB"))  # (3, H, W)
    extra_channel = rgb_tensor[0:1]               # duplicate R channel as 4th
    tensor = torch.cat([rgb_tensor, extra_channel], dim=0).unsqueeze(0).to(device)  # (1, 4, H, W)

    # Per-channel normalization (same as training)
    tensor = (tensor - tensor.mean(dim=(2, 3), keepdim=True)) / (
        tensor.std(dim=(2, 3), keepdim=True) + 1e-6
    )

    with torch.no_grad():
        wind_speed = model(tensor).item()

    # If weights are loaded, output is a real prediction; clamp to physical range
    wind_speed = float(np.clip(wind_speed, 10, 200))

    intensity = classify_intensity(wind_speed)
    return {
        "wind_speed_knots": round(wind_speed, 1),
        **intensity,
        "model_mode": "trained" if WEIGHTS_PATH.exists() else "demo",
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Cyclone AI API is running"}


@app.get("/api/cyclones")
def get_cyclones():
    """Return list of unique cyclone names with basic stats."""
    df = load_data()
    storms = []
    for sid, grp in df.groupby("SID"):
        name = grp["NAME"].iloc[0]
        max_wind = grp["WMO_WIND"].max()   # NaN if ALL rows are NaN
        start_time = grp["ISO_TIME"].min()
        end_time = grp["ISO_TIME"].max()

        if pd.isna(max_wind):
            # No wind data at all for this storm — mark unclassified
            category = "Unclassified"
            color    = "#6b7a99"
        else:
            info     = classify_intensity(float(max_wind))
            category = info["category"]
            color    = info["color"]

        storms.append({
            "sid": sid,
            "name": name,
            "max_wind_knots": round(float(max_wind), 1) if not pd.isna(max_wind) else None,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "record_count": len(grp),
            "category": category,
            "color": color,
        })
    storms.sort(key=lambda x: x["start_time"], reverse=True)
    return {"cyclones": storms, "total": len(storms)}


@app.get("/api/cyclones/{sid}/track")
def get_track(sid: str):
    """Return full track (lat/lon/wind) for a specific cyclone SID."""
    df = load_data()
    grp = df[df["SID"] == sid]
    if grp.empty:
        raise HTTPException(status_code=404, detail=f"Cyclone {sid} not found")

    track = []
    for _, row in grp.iterrows():
        wind = row["WMO_WIND"]
        intensity = classify_intensity(float(wind) if not np.isnan(wind) else 0)
        track.append({
            "time": str(row["ISO_TIME"]),
            "lat": float(row["LAT"]),
            "lon": float(row["LON"]),
            "wind_knots": float(wind) if not np.isnan(wind) else None,
            **intensity,
        })

    name = grp["NAME"].iloc[0]
    max_wind = grp["WMO_WIND"].max()
    return {
        "sid": sid,
        "name": name,
        "max_wind_knots": round(float(max_wind), 1) if not np.isnan(max_wind) else None,
        "track": track,
    }


@app.get("/api/stats")
def get_stats():
    """Aggregate stats across the whole dataset."""
    df = load_data()

    total = df["SID"].nunique()
    named = df[df["NAME"] != "UNNAMED"]["SID"].nunique()
    max_wind = df["WMO_WIND"].max()

    # Classify each STORM by its peak wind, not each row
    dist: dict[str, int] = {}
    for sid, grp in df.groupby("SID"):
        peak = grp["WMO_WIND"].max()
        if pd.isna(peak):
            cat = "Unclassified"
        else:
            cat = classify_intensity(float(peak))["category"]
        dist[cat] = dist.get(cat, 0) + 1

    return {
        "total_storms": int(total),
        "named_storms": int(named),
        "max_recorded_wind_knots": round(float(max_wind), 1) if not pd.isna(max_wind) else None,
        "intensity_distribution": dist,
        "basin": "NI (North Indian Ocean)",
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a satellite image → receive wind speed prediction + intensity category.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    result = run_model_on_image(image)

    # Also return a base64 thumbnail so the frontend can display it
    thumb = image.copy()
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        **result,
        "filename": file.filename,
        "thumbnail": f"data:image/jpeg;base64,{b64}",
    }


@app.get("/api/mock-frames")
def list_mock_frames():
    """List available mock cyclone satellite frames."""
    frames = []
    for f in sorted(FRAMES_DIR.glob("*.jpeg")):
        frames.append({"filename": f.name, "path": str(f)})
    return {"frames": frames}


@app.get("/api/mock-frames/{filename}/predict")
def predict_mock_frame(filename: str):
    """Run inference on one of the bundled mock frames."""
    fpath = FRAMES_DIR / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    result = run_inference_pipeline(str(fpath))
    intensity = classify_intensity(result["wind_speed_knots"])
    return {
        "filename": filename,
        "wind_speed_knots": result["wind_speed_knots"],
        **intensity,
    }
