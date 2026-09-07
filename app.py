"""
Cyclone AI — Streamlit App
===========================
Single-file app that replicates all three pages:
  • Dashboard  — stats, charts, storm table
  • Track View — interactive map + wind chart
  • AI Predict — upload image → wind speed prediction
"""

import io
import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# ── Path setup so model.py is importable ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from model import CycloneIntensityCNN

# ── Constants ────────────────────────────────────────────────────────────────
WEIGHTS_PATH = ROOT / "outputs" / "cyclone_cnn_best.pth"
DATA_PATH    = ROOT / "demo_cyclones.csv"
FRAMES_DIR   = ROOT / "mock_cyclone_frames"

INTENSITY_ORDER = [
    "Tropical Depression",
    "Tropical Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
    "Unclassified",
]

INTENSITY_COLORS = {
    "Tropical Depression":             "#22d3ee",
    "Tropical Storm":                  "#86efac",
    "Severe Cyclonic Storm":           "#fde047",
    "Very Severe Cyclonic Storm":      "#fb923c",
    "Extremely Severe Cyclonic Storm": "#f87171",
    "Super Cyclonic Storm":            "#e879f9",
    "Unclassified":                    "#6b7a99",
}


def classify(wind_knots: float) -> str:
    if wind_knots < 34:  return "Tropical Depression"
    if wind_knots < 48:  return "Tropical Storm"
    if wind_knots < 64:  return "Severe Cyclonic Storm"
    if wind_knots < 96:  return "Very Severe Cyclonic Storm"
    if wind_knots < 120: return "Extremely Severe Cyclonic Storm"
    return "Super Cyclonic Storm"


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in ("WMO_WIND", "LAT", "LON"):
        if col in df.columns:
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    return df


@st.cache_data
def get_storm_list(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, grp in df.groupby("SID"):
        name     = grp["NAME"].iloc[0]
        max_wind = grp["WMO_WIND"].max()
        start    = grp["ISO_TIME"].min()
        end      = grp["ISO_TIME"].max()
        cat      = classify(float(max_wind)) if not pd.isna(max_wind) else "Unclassified"
        rows.append({
            "sid":           sid,
            "name":          name,
            "max_wind_knots": round(float(max_wind), 1) if not pd.isna(max_wind) else None,
            "category":      cat,
            "color":         INTENSITY_COLORS[cat],
            "start":         str(start)[:10],
            "end":           str(end)[:10],
            "records":       len(grp),
        })
    return pd.DataFrame(rows).sort_values("start", ascending=False).reset_index(drop=True)


# ── Model inference (cached per session) ────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CycloneIntensityCNN(in_channels=4, pretrained=False)
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=device))
        mode = "trained"
    else:
        mode = "demo (random weights)"
    model.to(device).eval()
    return model, device, mode


def predict_image(image: Image.Image) -> dict:
    from torchvision import transforms as T
    model, device, mode = load_model()

    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    rgb    = transform(image.convert("RGB"))       # (3, H, W)
    extra  = rgb[0:1]                               # duplicate R → 4th channel
    tensor = torch.cat([rgb, extra], dim=0).unsqueeze(0).to(device)
    tensor = (tensor - tensor.mean(dim=(2, 3), keepdim=True)) / (
        tensor.std(dim=(2, 3), keepdim=True) + 1e-6
    )
    with torch.no_grad():
        wind = float(np.clip(model(tensor).item(), 10, 200))

    cat = classify(wind)
    return {
        "wind_speed_knots": round(wind, 1),
        "category":         cat,
        "color":            INTENSITY_COLORS[cat],
        "model_mode":       mode,
    }


# ── Page: Dashboard ──────────────────────────────────────────────────────────
def page_dashboard():
    st.title("🌀 Cyclone AI — Dashboard")
    st.caption("North Indian Ocean basin · IBTrACS v04r01 · 2000+")

    df     = load_data()
    storms = get_storm_list(df)

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Storms",  df["SID"].nunique())
    c2.metric("Named Storms",  df[df["NAME"] != "UNNAMED"]["SID"].nunique())
    peak = df["WMO_WIND"].max()
    c3.metric("Peak Wind",     f"{round(float(peak), 1)} kts" if not pd.isna(peak) else "—")
    c4.metric("Basin",         "NI (North Indian Ocean)")

    st.divider()

    col_bar, col_pie = st.columns(2)

    # Bar chart — top 15
    with col_bar:
        st.subheader("Top 15 by Max Wind Speed")
        top15 = storms.dropna(subset=["max_wind_knots"]).head(15)
        fig = px.bar(
            top15,
            x="name", y="max_wind_knots",
            color="category",
            color_discrete_map=INTENSITY_COLORS,
            labels={"max_wind_knots": "Max Wind (kts)", "name": "Storm"},
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False, margin=dict(t=10, b=60), xaxis_tickangle=-38)
        st.plotly_chart(fig, use_container_width=True)

    # Pie chart — intensity distribution
    with col_pie:
        st.subheader("Intensity Distribution")
        dist = storms["category"].value_counts().reset_index()
        dist.columns = ["category", "count"]
        dist["color"] = dist["category"].map(INTENSITY_COLORS)
        fig2 = px.pie(
            dist, names="category", values="count",
            color="category", color_discrete_map=INTENSITY_COLORS,
            template="plotly_dark",
        )
        fig2.update_traces(textposition="inside", textinfo="percent")
        fig2.update_layout(margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("All Storms")

    display = storms.copy()
    display["max_wind_knots"] = display["max_wind_knots"].fillna("—").astype(str)
    st.dataframe(
        display[["name", "sid", "max_wind_knots", "category", "start", "records"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "name":           st.column_config.TextColumn("Name"),
            "sid":            st.column_config.TextColumn("SID"),
            "max_wind_knots": st.column_config.TextColumn("Max Wind (kts)"),
            "category":       st.column_config.TextColumn("Category"),
            "start":          st.column_config.TextColumn("Start Date"),
            "records":        st.column_config.NumberColumn("Records"),
        }
    )


# ── Page: Track View ─────────────────────────────────────────────────────────
def page_track():
    st.title("🗺️ Cyclone Track Viewer")

    df     = load_data()
    storms = get_storm_list(df)

    storm_names = storms["name"] + " (" + storms["sid"] + ")"
    choice = st.selectbox("Select a storm", storm_names)
    if not choice:
        return

    idx = storm_names[storm_names == choice].index[0]
    sid = storms.loc[idx, "sid"]

    track_df = df[df["SID"] == sid].dropna(subset=["LAT", "LON"]).copy()
    track_df["category"]  = track_df["WMO_WIND"].apply(
        lambda w: classify(float(w)) if not pd.isna(w) else "Unclassified"
    )
    track_df["color"]     = track_df["category"].map(INTENSITY_COLORS)
    track_df["wind_label"] = track_df["WMO_WIND"].apply(
        lambda w: f"{round(float(w), 1)} kts" if not pd.isna(w) else "—"
    )

    st.caption(
        f"**{storms.loc[idx, 'name']}** · Peak wind: "
        f"{storms.loc[idx, 'max_wind_knots']} kts · "
        f"{storms.loc[idx, 'start']} → {storms.loc[idx, 'end']}"
    )

    # Map using plotly scatter_mapbox
    fig_map = go.Figure()

    # Track polyline
    fig_map.add_trace(go.Scattermapbox(
        lat=track_df["LAT"].tolist(),
        lon=track_df["LON"].tolist(),
        mode="lines",
        line=dict(width=1.5, color="#0ea5e9"),
        name="Track",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Points colored by intensity
    for cat in INTENSITY_ORDER:
        sub = track_df[track_df["category"] == cat]
        if sub.empty:
            continue
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["LAT"].tolist(),
            lon=sub["LON"].tolist(),
            mode="markers",
            marker=dict(size=8, color=INTENSITY_COLORS[cat]),
            name=cat,
            text=sub.apply(
                lambda r: f"{str(r['ISO_TIME'])[:16]}<br>{r['wind_label']}<br>{cat}", axis=1
            ).tolist(),
            hoverinfo="text",
        ))

    center_lat = track_df["LAT"].mean()
    center_lon = track_df["LON"].mean()
    fig_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        legend=dict(
            bgcolor="rgba(20,20,30,0.8)",
            font=dict(color="white", size=10),
        ),
        paper_bgcolor="#0f1117",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Wind speed chart
    wind_df = track_df.dropna(subset=["WMO_WIND"]).copy()
    if not wind_df.empty:
        st.subheader(f"Wind Speed Over Time — {storms.loc[idx, 'name']}")
        fig_wind = px.line(
            wind_df,
            x="ISO_TIME", y="WMO_WIND",
            labels={"ISO_TIME": "Time", "WMO_WIND": "Wind Speed (kts)"},
            template="plotly_dark",
        )
        fig_wind.add_hline(y=64, line_dash="dash", line_color="#fde047",
                           annotation_text="Severe", annotation_position="top left")
        fig_wind.add_hline(y=96, line_dash="dash", line_color="#f87171",
                           annotation_text="Very Severe", annotation_position="top left")
        fig_wind.update_layout(margin=dict(t=10), height=200)
        st.plotly_chart(fig_wind, use_container_width=True)


# ── Page: AI Predict ─────────────────────────────────────────────────────────
def page_predict():
    st.title("🛰️ AI Intensity Prediction")
    st.caption("Upload a satellite image → ResNet-18 estimates wind speed and intensity category.")

    _, _, mode = load_model()
    st.info(f"Model mode: **{mode}**  |  Weights: `outputs/cyclone_cnn_best.pth`")

    tab_upload, tab_mock = st.tabs(["Upload Image", "Bundled Test Frames"])

    # ── Upload tab ──
    with tab_upload:
        uploaded = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            col_img, col_res = st.columns(2)
            with col_img:
                st.image(image, caption=uploaded.name, use_column_width=True)
            with col_res:
                with st.spinner("Running model…"):
                    result = predict_image(image)
                cat   = result["category"]
                color = result["color"]
                wind  = result["wind_speed_knots"]

                st.markdown(f"### {wind} knots")
                st.markdown(
                    f'<span style="background:{color};color:#000;padding:4px 12px;'
                    f'border-radius:12px;font-weight:600;font-size:0.9rem">{cat}</span>',
                    unsafe_allow_html=True,
                )
                st.divider()
                st.metric("Wind Speed", f"{wind} kts")
                st.metric("Category",   cat)

    # ── Mock frames tab ──
    with tab_mock:
        frames = sorted(FRAMES_DIR.glob("*.jpeg"))
        if not frames:
            st.warning("No frames found in mock_cyclone_frames/")
        else:
            cols = st.columns(3)
            for i, fpath in enumerate(frames):
                with cols[i % 3]:
                    img = Image.open(fpath)
                    st.image(img, caption=fpath.name, use_column_width=True)
                    if st.button("Predict", key=fpath.name):
                        with st.spinner("Running…"):
                            res = predict_image(img)
                        st.markdown(
                            f'<span style="background:{res["color"]};color:#000;'
                            f'padding:3px 10px;border-radius:10px;font-size:0.8rem;font-weight:600">'
                            f'{res["wind_speed_knots"]} kts · {res["category"]}</span>',
                            unsafe_allow_html=True,
                        )


# ── App shell ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyclone AI",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🌀 Cyclone AI")
st.sidebar.caption("North Indian Ocean · IBTrACS + ResNet-18")
page = st.sidebar.radio("Navigate", ["Dashboard", "Track View", "AI Predict"])
st.sidebar.divider()
st.sidebar.markdown("**Stack**")
st.sidebar.markdown("PyTorch · ResNet-18 · FastAPI · Streamlit")
st.sidebar.markdown("**Data**")
st.sidebar.markdown("IBTrACS v04r01 · TCIR satellite")

if page == "Dashboard":
    page_dashboard()
elif page == "Track View":
    page_track()
elif page == "AI Predict":
    page_predict()
