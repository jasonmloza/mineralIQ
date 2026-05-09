title: MineralIQ
emoji: ⛏️
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: true
---

# ⛏️ MineralIQ — Gold Exploration AI

> **AMD Hackathon submission** · Built by Jason Mloza

MineralIQ turns satellite imagery into an explainable AI system that estimates gold mineralisation potential at any clicked location on Earth — no fieldwork required.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/JasonMloza/MineralIQ)
[![GitHub](https://img.shields.io/badge/GitHub-jasonmloza%2FmineralIQ-black?logo=github)](https://github.com/jasonmloza/mineralIQ)

---

## 🌍 What It Does

1. **Click anywhere on Earth** on an interactive global map
2. **Sentinel-2 satellite imagery** is automatically queried for that location (2023 median composite, <20% cloud)
3. **8 spectral mineral proxies** are extracted from the multispectral bands
4. **Normalised anomaly score** is computed across the 20 km region of interest
5. **Colour heatmap** (green → blue → red) is overlaid on the satellite image
6. **Confidence score** (0–100%) quantifies the mineralisation signal strength
7. **AI geological explanation** interprets each spectral index with domain-expert reasoning
8. **XGBoost classifier** (optional) predicts gold occurrence from terrain features

---

## 🔬 The 8-Feature Spectral Pipeline

| Feature | Formula | Geological Meaning |
|---|---|---|
| Iron Oxide Index | B4 ÷ B2 | Hydrothermal Fe-enrichment (primary gold pathfinder) |
| Clay Mineral Index | B11 ÷ B8 | Argillic alteration (kaolinite, sericite, chlorite) |
| NDVI | (B8−B4)÷(B8+B4) | Vegetation cover — bare ground = exposed alteration |
| SAVI | Soil-adjusted NDVI | Removes soil-brightness noise from vegetation signal |
| RVI | B4 ÷ B3 | Iron staining — gossan development over sulphide ore |
| NDII | (B8−B11)÷(B8+B11) | Moisture / clay moisture proxy |
| MGI | B9 ÷ B11 | Mg-alteration (chlorite, dolomite — orogenic gold indicator) |
| Thermal SWIR | B12 ÷ B11 | Silicification / residual hydrothermal heat signature |

All 8 bands are summed and normalised with `unitScale(min, max)` within the ROI to produce a spatially comparable anomaly score.

---

## 🤖 AI Explanation Layer

The app generates expert geological interpretation for each spectral index with thresholded grades (Strong / Moderate / Weak) and a final verdict:

- **High (≥70%)** — Priority exploration target; recommend ground-truth sampling and geophysics
- **Medium (40–70%)** — Secondary target; investigate with regional geology context
- **Low (<40%)** — Weak signal; redirect to higher-anomaly zones

---

## 🏗️ Architecture

```
User click (lat/lon)
        │
        ▼
Google Earth Engine
  Sentinel-2 SR Harmonized
  2023 Median Composite
  20 km ROI Buffer
        │
        ▼
feature_extractor.py
  8-band spectral indices
        │
        ▼
compute_anomaly_score()
  Sum → unitScale normalisation
        │
   ┌────┴────┐
   ▼         ▼
Heatmap     Confidence
Tile Layer  Score (%)
   │         │
   └────┬────┘
        ▼
 AI Explanation Panel
 (per-feature interpretation
  + overall verdict)
        │
        ▼ (optional)
 XGBoost Classifier
 (terrain features → deposit prediction)
```

---

## 🗂️ Project Structure

```
mineraliq/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── README.md
│
├── model/
│   ├── __init__.py
│   ├── feature_extractor.py        # 8-band GEE spectral feature extraction
│   ├── predict.py                  # XGBoost inference
│   ├── train.py                    # Model training script
│   ├── satellite_fetch.py          # Sentinel Hub direct fetch (optional fallback)
│   └── gold_model.pkl              # Trained XGBoost model
│
├── data/
│   └── training_data.csv           # 60-sample labelled deposit dataset
│
└── .streamlit/
    ├── config.toml                 # Dark theme + server config
    └── secrets.toml.template       # Credentials template (do not commit real secrets)
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/jasonmloza/mineralIQ.git
cd mineralIQ
pip install -r requirements.txt
earthengine authenticate          # one-time GEE auth
streamlit run app.py
```

### Retrain the XGBoost model

```bash
python model/train.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Folium |
| Satellite data | Google Earth Engine (Sentinel-2 SR) |
| Spectral analysis | Earth Engine Python API |
| ML classifier | XGBoost |
| Data processing | Pandas, NumPy |
| Model persistence | Joblib |
| Deployment | Hugging Face Spaces |

---

## 📡 Data Sources

- **Sentinel-2 Surface Reflectance** — ESA / Copernicus via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`)
- **Training labels** — Gold deposit coordinates in the Central African gold belt (Malawi / Zimbabwe)

---

## ⚠️ Disclaimer

MineralIQ uses rule-based spectral heuristics derived from multispectral satellite imagery. Results are indicative only and must be validated against field geology, geochemical sampling, and regional structural data. This tool is not a substitute for professional geological survey and should not be used as the sole basis for mining or investment decisions.

---

## 👤 Author

**Jason Mloza**
- GitHub: [github.com/jasonmloza](https://github.com/jasonmloza)
- HF: [huggingface.co/spaces/JasonMloza/MineralIQ](https://huggingface.co/spaces/JasonMloza/MineralIQ)