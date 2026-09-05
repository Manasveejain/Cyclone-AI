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
    df["WMO_WIND"] = pd.to_numeric(df["WMO_WIND"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────
def classify_intensity(wind_knots: float) -> dict:
    """Return category name and colour for a given wind speed."""
    if wind_knots < 34:
        return {"category": "Tropical Depression", "color": "#4ade80"}
    elif wind_knots < 48:
        return {"category": "Tropical Storm", "color": "#facc15"}
    elif wind_knots < 64:
        return {"category": "Severe Cyclonic Storm", "color": "#fb923c"}
    elif wind_knots < 96:
        return {"category": "Very Severe Cyclonic Storm", "color": "#f87171"}
    elif wind_knots < 120:
        return {"category": "Extremely Severe Cyclonic Storm", "color": "#c084fc"}
    else:
        return {"category": "Super Cyclonic Storm", "color": "#e11d48"}


def run_model_on_image(image: Image.Image) -> dict:
    """Run the ResNet18 model on a PIL Image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycloneIntensityCNN()
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        raw = model(tensor).item()

    # Raw output is unbounded; map it to a plausible wind range via sigmoid scaling
    wind_speed = abs(raw) % 160 + 20  # clamp to 20-180 knots for demo

    intensity = classify_intensity(wind_speed)
    return {
        "wind_speed_knots": round(wind_speed, 1),
        **intensity,
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
        max_wind = grp["WMO_WIND"].max()
        start_time = grp["ISO_TIME"].min()
        end_time = grp["ISO_TIME"].max()
        intensity = classify_intensity(max_wind if not np.isnan(max_wind) else 0)
        storms.append({
            "sid": sid,
            "name": name,
            "max_wind_knots": round(float(max_wind), 1) if not np.isnan(max_wind) else None,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "record_count": len(grp),
            **intensity,
        })
    # Sort by start time descending
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

    named = df[~df["NAME"].isin(["UNNAMED"])]["SID"].nunique()
    total = df["SID"].nunique()
    max_wind = df["WMO_WIND"].max()

    # Intensity distribution
    def cat(w):
        if np.isnan(w):
            return "Unknown"
        return classify_intensity(w)["category"]

    df["category"] = df["WMO_WIND"].apply(cat)
    dist = df.groupby("category")["SID"].nunique().to_dict()

    return {
        "total_storms": int(total),
        "named_storms": int(named),
        "max_recorded_wind_knots": round(float(max_wind), 1) if not np.isnan(max_wind) else None,
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
