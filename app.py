import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model.feature_extractor import extract_spectral_features, compute_anomaly_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MineralIQ - Gold Exploration AI",
    layout="wide",
    page_icon="⛏️"
)

st.markdown("""
<style>
body, .main { background-color: #0e1117; }
.block-container { padding-top: 1.2rem; }

.legend-box {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 18px 14px;
    color: white;
    height: 100%;
}
.legend-title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 14px;
    color: #D4A017;
    letter-spacing: 0.04em;
}
.legend-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 14px;
    font-size: 0.88rem;
}
.legend-swatch {
    width: 28px; height: 28px;
    border-radius: 6px;
    margin-right: 12px;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.15);
}
.legend-label { font-weight: 600; color: white; }
.legend-desc  { color: #aaa; font-size: 0.78rem; margin-top: 2px; }
.meta-block   { font-size: 0.8rem; color: #aaa; line-height: 1.6; margin-top: 14px; }
.meta-block b { color: #e8c96e; }

.ai-panel {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    border-radius: 14px;
    padding: 24px 28px;
    color: white;
    margin-top: 1.5rem;
    border: 1px solid rgba(212,160,23,0.25);
}
.ai-panel-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #D4A017;
    margin-bottom: 18px;
    letter-spacing: 0.03em;
}
.signal-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}
.signal-card {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 14px 16px;
    border-left: 4px solid #D4A017;
}
.signal-name  { font-weight: 600; font-size: 0.85rem; color: #e8c96e; margin-bottom: 4px; }
.signal-value { font-size: 1.35rem; font-weight: 700; color: white; }
.signal-grade { font-size: 0.85rem; }
.signal-interp { color: #aaa; font-size: 0.8rem; margin-top: 6px; line-height: 1.5; }

.verdict-box  {
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 8px;
}
.verdict-title { font-weight: 700; font-size: 1rem; margin-bottom: 8px; }
.verdict-text  { color: #ddd; font-size: 0.9rem; line-height: 1.7; }

.disclaimer {
    margin-top: 18px;
    padding: 12px 16px;
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    font-size: 0.8rem;
    color: #888;
    line-height: 1.6;
}
.disclaimer b { color: #aaa; }

.conf-label {
    font-size: 0.82rem;
    color: #888;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Earth Engine init ─────────────────────────────────────────────────────────
@st.cache_resource
def init_ee():
    import json
    import tempfile

    key_json = os.environ.get("EE_CREDENTIALS")

    if key_json:
        try:
            key_data = json.loads(key_json)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(key_data, f)
                key_path = f.name
            credentials = ee.ServiceAccountCredentials(
                email=key_data["client_email"],
                key_file=key_path
            )
            ee.Initialize(credentials, project="mineraliq-ai")
            return True, "Authenticated via service account"
        except Exception as e:
            return False, f"Service account auth failed: {e}"
    else:
        try:
            ee.Initialize(project="mineraliq-ai")
            return True, "Authenticated via default credentials"
        except Exception as e:
            try:
                ee.Authenticate()
                ee.Initialize(project="mineraliq-ai")
                return True, "Authenticated via OAuth"
            except Exception as e2:
                return False, f"All auth methods failed: {e2}"


ee_ok, ee_msg = init_ee()

if not ee_ok:
    st.error(f"❌ Earth Engine failed to initialise: {ee_msg}")
    st.stop()


# ── Helper: safe mean extraction ─────────────────────────────────────────────
def safe_mean(image, band, roi, scale=100):
    try:
        result = image.select(band).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).get(band).getInfo()
        return round(float(result), 4) if result is not None else 0.0
    except Exception:
        return 0.0


def safe_mean_noband(image, roi, scale=100):
    try:
        band = image.bandNames().getInfo()[0]
        result = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).get(band).getInfo()
        return round(float(result), 4) if result is not None else 0.0
    except Exception:
        return 0.0


# ── Interpretation helpers ────────────────────────────────────────────────────
def grade(val, high_thresh, med_thresh, high_label, med_label, low_label,
          high_color="#ff5555", med_color="#ffcc00", low_color="#55cc77"):
    if val >= high_thresh:
        return high_label, high_color
    elif val >= med_thresh:
        return med_label, med_color
    else:
        return low_label, low_color


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#D4A017;font-size:2.2rem;font-weight:700;margin-bottom:0;">'
    '⛏️ MineralIQ — Gold Exploration AI</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color:#888;font-size:1rem;margin-top:4px;">'
    'Click anywhere on Earth to analyse gold mineralisation potential '
    'using 8-band Sentinel-2 spectral signatures</p>',
    unsafe_allow_html=True
)
st.info("👇 Click any location on the map below to begin analysis")


# ── Global base map ───────────────────────────────────────────────────────────
m = folium.Map(location=[0, 20], zoom_start=2, tiles="OpenStreetMap")
map_data = st_folium(m, width="100%", height=500, key="global_map")


# ── On click ──────────────────────────────────────────────────────────────────
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"📍 Selected: **{lat:.5f}°, {lon:.5f}°**")

    with st.spinner("🛰️ Querying Sentinel-2 · Extracting 8 spectral features · Computing anomaly score…"):

        try:
            point = ee.Geometry.Point([lon, lat])
            roi   = point.buffer(20000).bounds()

            # ── Sentinel-2 2023 median composite ─────────────────────────────
            image = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate("2023-01-01", "2023-12-31")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                .median()
                .clip(roi)
            )

            # ── 8-feature extraction via feature_extractor.py ────────────────
            features     = extract_spectral_features(image)
            anomaly_norm = compute_anomaly_score(features, roi)

            # ── Per-feature mean values for XAI ──────────────────────────────
            iron_mean    = safe_mean(features, 'iron_oxide',    roi)
            clay_mean    = safe_mean(features, 'clay',          roi)
            ndvi_mean    = safe_mean(features, 'ndvi',          roi)
            savi_mean    = safe_mean(features, 'savi',          roi)
            rvi_mean     = safe_mean(features, 'rvi',           roi)
            ndii_mean    = safe_mean(features, 'ndii',          roi)
            mgi_mean     = safe_mean(features, 'mgi',           roi)
            thermal_mean = safe_mean(features, 'thermal_ratio', roi)

            confidence   = safe_mean_noband(anomaly_norm, roi) * 100

            # ── Tile URLs ─────────────────────────────────────────────────────
            rgb_tile = image.getMapId({
                "bands": ["B4", "B3", "B2"], "min": 0, "max": 3000
            })["tile_fetcher"].url_format

            heatmap_tile = anomaly_norm.getMapId({
                "min": 0, "max": 1, "palette": ["00cc44", "0066ff", "ff1a1a"]
            })["tile_fetcher"].url_format

        except Exception as e:
            st.error(f"❌ EE query failed: {e}")
            st.stop()

    # ── Map + Legend side by side ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ Satellite Imagery & Gold Probability Heatmap")
    map_col, legend_col = st.columns([4, 1])

    with map_col:
        sat_map = folium.Map(location=[lat, lon], zoom_start=10)
        folium.TileLayer(
            tiles=rgb_tile,
            attr="Google Earth Engine / Sentinel-2",
            name="Satellite (True Colour)",
            overlay=True, show=True
        ).add_to(sat_map)
        folium.TileLayer(
            tiles=heatmap_tile,
            attr="MineralIQ",
            name="Gold Probability Heatmap",
            overlay=True, show=True, opacity=0.75
        ).add_to(sat_map)
        folium.LayerControl(collapsed=False).add_to(sat_map)
        folium.Marker(
            location=[lat, lon],
            popup=f"{lat:.5f}, {lon:.5f}",
            tooltip="Selected location",
            icon=folium.Icon(color="orange", icon="star", prefix="fa")
        ).add_to(sat_map)
        st_folium(sat_map, width="100%", height=620, key="result_map")

    with legend_col:
        st.markdown(
            '<div class="legend-box">'
            '<div class="legend-title">🎨 Heatmap Legend</div>'

            '<div class="legend-row">'
            '<div class="legend-swatch" style="background:#ff1a1a;"></div>'
            '<div><div class="legend-label">High probability</div>'
            '<div class="legend-desc">Strong iron-oxide &amp; clay anomaly — prime target zone</div></div>'
            '</div>'

            '<div class="legend-row">'
            '<div class="legend-swatch" style="background:#0066ff;"></div>'
            '<div><div class="legend-label">Medium probability</div>'
            '<div class="legend-desc">Moderate spectral anomaly — worth investigating</div></div>'
            '</div>'

            '<div class="legend-row">'
            '<div class="legend-swatch" style="background:#00cc44;"></div>'
            '<div><div class="legend-label">Low probability</div>'
            '<div class="legend-desc">Weak signal — unlikely mineralisation</div></div>'
            '</div>'

            '<hr style="border-color:rgba(255,255,255,0.1);margin:16px 0;">'
            '<div class="meta-block">'
            '<b>Data source</b><br>'
            'Sentinel-2 SR<br>'
            'COPERNICUS/S2_SR_HARMONIZED<br>'
            '2023 · &lt;20% cloud · 20 km ROI<br><br>'
            '<b>Features used</b><br>'
            'Iron oxide: B4÷B2<br>'
            'Clay: B11÷B8<br>'
            'NDVI: (B8−B4)÷(B8+B4)<br>'
            'SAVI: soil-adj. veg.<br>'
            'RVI: B4÷B3<br>'
            'NDII: (B8−B11)÷(B8+B11)<br>'
            'MGI: B9÷B11<br>'
            'Thermal: B12÷B11'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Confidence metrics ────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🎯 Confidence",       f"{confidence:.1f}%")
    m2.metric("🔴 Iron Oxide",       iron_mean)
    m3.metric("🔵 Clay Index",       clay_mean)
    m4.metric("🌿 NDVI",             ndvi_mean)
    m5.metric("🔥 Thermal Ratio",    thermal_mean)
    m6.metric("💧 NDII",             ndii_mean)

    st.markdown('<div class="conf-label">Overall mineralisation anomaly signal strength</div>', unsafe_allow_html=True)
    st.progress(min(int(confidence), 100))

    # ── AI Explanation Panel ──────────────────────────────────────────────────
    # — Iron oxide
    if iron_mean > 1.3:
        io_grade, io_col = "🔥 Strong", "#ff5555"
        io_interp = (
            "Elevated Fe-oxide ratio (>1.3) indicates strong hydrothermal fluid-rock interaction. "
            "This is a primary geochemical pathfinder for epithermal and orogenic gold systems. "
            "Ferricrete and gossanous outcrops commonly overlie gold-bearing reefs."
        )
    elif iron_mean > 1.1:
        io_grade, io_col = "⚠️ Moderate", "#ffcc00"
        io_interp = (
            "Moderate iron-oxide signal (1.1–1.3). Partial hydrothermal alteration or lateritic "
            "weathering of iron-rich minerals. Consistent with mineralised corridors requiring "
            "closer investigation."
        )
    else:
        io_grade, io_col = "✅ Weak", "#55cc77"
        io_interp = (
            "Iron-oxide levels near background (<1.1). Limited evidence of hydrothermal iron "
            "enrichment. Low epithermal gold potential from this proxy alone."
        )

    # — Clay
    if clay_mean > 0.85:
        cl_grade, cl_col = "🔥 Strong", "#ff5555"
        cl_interp = (
            "High SWIR/NIR ratio (>0.85) reveals significant phyllosilicate (kaolinite, sericite, "
            "chlorite) development — classic argillic alteration halos surrounding gold-bearing veins "
            "in both epithermal low-sulphidation and orogenic settings."
        )
    elif clay_mean > 0.65:
        cl_grade, cl_col = "⚠️ Moderate", "#ffcc00"
        cl_interp = (
            "Moderate argillic alteration (0.65–0.85). Secondary clay presence through silicate "
            "weathering. Such zones are known gold hosts in epithermal environments at appropriate "
            "structural settings."
        )
    else:
        cl_grade, cl_col = "✅ Weak", "#55cc77"
        cl_interp = (
            "Low clay signature (<0.65). Minimal argillic alteration detected, reducing likelihood "
            "of gold-hosting phyllosilicate assemblages at this location."
        )

    # — NDVI
    if ndvi_mean < 0.2:
        veg_interp = (
            "Very sparse vegetation (NDVI < 0.2). Consistent with exposed mineralised or heavily "
            "altered ground — bare/rocky terrain typical near outcropping ore zones. Spectral "
            "signatures are unmasked and highly reliable."
        )
    elif ndvi_mean < 0.4:
        veg_interp = (
            "Low-moderate vegetation (NDVI 0.2–0.4). Partial cover; spectral indices are not "
            "significantly attenuated. Alteration mapping remains valid."
        )
    else:
        veg_interp = (
            "Dense vegetation (NDVI > 0.4). Canopy may partially attenuate SWIR/NIR indices. "
            "Consider dry-season imagery or SAR data to improve confidence in vegetated terrain."
        )

    # — Thermal
    if thermal_mean > 1.1:
        th_interp = (
            "Elevated SWIR thermal ratio (B12/B11 > 1.1). Possible residual hydrothermal heat "
            "signature or silicified bedrock — silicification is a key alteration type in "
            "gold-bearing epithermal systems."
        )
    else:
        th_interp = (
            "Normal SWIR thermal ratio. No significant silicification or hydrothermal heat anomaly. "
            "This does not rule out mineralisation at depth."
        )

    # — RVI
    rvi_interp = (
        "Elevated RVI supports Fe-staining from hydrothermal fluids — associated with gossan "
        "development over sulphide ore bodies."
        if rvi_mean > 1.2 else
        "RVI within normal range. No significant iron-staining detected from this proxy."
    )

    # — NDII
    ndii_interp = (
        "Low NDII indicates dry, possibly silicified or chloritised rock with reduced moisture "
        "retention — consistent with hydrothermal alteration mineralogy."
        if ndii_mean < 0.1 else
        "Moderate NDII. Some soil moisture present; terrain may include alluvial cover masking "
        "underlying geology."
    )

    # — MGI
    mgi_interp = (
        "Elevated MGI (B9/B11 > 1.0) suggests Mg-rich alteration assemblage (chlorite, "
        "dolomite). Mg-metasomatism is a recognised pathfinder in orogenic gold systems."
        if mgi_mean > 1.0 else
        "MGI within background range. No anomalous Mg-alteration detected."
    )

    # — Overall verdict
    if confidence >= 70:
        vrd_grade  = "HIGH"
        vrd_colour = "#ff4444"
        vrd_bg     = "#ff444418"
        vrd_border = "#ff444466"
        vrd_text = (
            "Multiple independent spectral proxies converge on this zone (confidence "
            + str(round(confidence, 1)) + "%). "
            "Iron-oxide index of " + str(iron_mean) + " combined with clay index of " + str(clay_mean)
            + " and anomalous SAVI/NDII signatures is consistent with an active or fossil hydrothermal "
            "system. The spectral signature matches known gold-bearing alteration assemblages in "
            "Archaean greenstone belts and Proterozoic shear zones. This location is a "
            "<strong>priority exploration target</strong> — recommend ground-truth geological "
            "mapping, rock-chip and soil geochemical sampling, and IP/resistivity geophysics."
        )
    elif confidence >= 40:
        vrd_grade  = "MODERATE"
        vrd_colour = "#4488ff"
        vrd_bg     = "#4488ff18"
        vrd_border = "#4488ff66"
        vrd_text = (
            "Partial spectral anomaly detected (confidence " + str(round(confidence, 1)) + "%). "
            "Iron-oxide at " + str(iron_mean) + " and clay at " + str(clay_mean)
            + " suggest localised alteration but the signal is not uniformly strong across all 8 indices. "
            "This zone is a <strong>secondary-priority target</strong> — investigate if regional "
            "geology includes favourable host lithologies (BIFs, shear zones, intrusive contacts). "
            "Stream-sediment geochemistry over the broader catchment is recommended."
        )
    else:
        vrd_grade  = "LOW"
        vrd_colour = "#22bb55"
        vrd_bg     = "#22bb5518"
        vrd_border = "#22bb5566"
        vrd_text = (
            "Weak spectral anomaly (confidence " + str(round(confidence, 1)) + "%). "
            "All 8 spectral indices are near background levels — iron-oxide " + str(iron_mean)
            + ", clay " + str(clay_mean) + ". "
            "This location is a <strong>low-priority target</strong> under the current model. "
            "Consider exploring adjacent areas with higher heatmap intensity, or apply ASTER SWIR "
            "data (which has superior clay discrimination at 30 m resolution) for a second opinion."
        )

    # — Build HTML panel
    def card(name, value, grade_lbl, col, interp):
        return (
            '<div class="signal-card" style="border-left-color:' + col + ';">'
            '<div class="signal-name">' + name + '</div>'
            '<div class="signal-value">' + str(value)
            + ' <span class="signal-grade" style="color:' + col + ';">' + grade_lbl + '</span></div>'
            '<div class="signal-interp">' + interp + '</div>'
            '</div>'
        )

    ai_html = (
        '<div class="ai-panel">'
        '<div class="ai-panel-title">🤖 AI Geological Explanation — 8-Band Spectral Analysis</div>'
        '<div class="signal-grid">'
        + card("🔴 Iron Oxide  (B4÷B2)",       iron_mean,    io_grade,              io_col,    io_interp)
        + card("🔵 Clay Mineral  (B11÷B8)",     clay_mean,    cl_grade,              cl_col,    cl_interp)
        + card("🌿 Vegetation  (NDVI)",          ndvi_mean,    "Index",               "#aaaaff", veg_interp)
        + card("🔥 Thermal SWIR  (B12÷B11)",    thermal_mean, "Index",               "#ff9955", th_interp)
        + card("🩸 Iron Staining  (RVI B4÷B3)", rvi_mean,     "Strong" if rvi_mean > 1.2 else "Normal", "#ff7766", rvi_interp)
        + card("💧 Moisture  (NDII)",            ndii_mean,    "Low" if ndii_mean < 0.1 else "Normal",   "#55aaff", ndii_interp)
        + card("🪨 Mg Alteration  (MGI B9÷B11)",mgi_mean,     "Elevated" if mgi_mean > 1.0 else "Normal","#cc88ff", mgi_interp)
        + card("🌱 Soil Adj. Veg.  (SAVI)",      savi_mean,    "Index",               "#88dd88", "SAVI removes soil-brightness noise from vegetation index — low values confirm sparse cover and reliable spectral alteration mapping.")
        + '</div>'
        '<div class="verdict-box" style="border:1px solid ' + vrd_border + ';background:' + vrd_bg + ';">'
        '<div class="verdict-title" style="color:' + vrd_colour + ';">⚖️ Overall Verdict: ' + vrd_grade + ' MINERALISATION POTENTIAL</div>'
        '<div class="verdict-text">' + vrd_text + '</div>'
        '</div>'
        '<div class="disclaimer">'
        '<b>⚠️ Disclaimer:</b> MineralIQ uses spectral heuristics derived from 8-band Sentinel-2 '
        'multispectral imagery. Results are indicative only and must be validated against field '
        'geology, geochemical sampling, and regional structural data. This tool is not a substitute '
        'for professional geological survey and should not be used as the sole basis for exploration '
        'investment decisions.'
        '</div>'
        '</div>'
    )

    st.markdown(ai_html, unsafe_allow_html=True)

    # ── XGBoost model panel (if pkl available) ────────────────────────────────
    try:
        import joblib
        import pandas as pd
        model = joblib.load(os.path.join(os.path.dirname(__file__), "model", "gold_model.pkl"))

        st.markdown("---")
        st.subheader("🤖 XGBoost Deposit Classifier")
        st.caption("Predicts gold occurrence from geographic features using a trained XGBoost model.")

        col_e, col_s, col_d = st.columns(3)
        elevation = col_e.number_input("Elevation (m)",        min_value=0,   max_value=5000, value=700,  step=50)
        slope     = col_s.number_input("Slope (°)",            min_value=0,   max_value=90,   value=12,   step=1)
        distance  = col_d.number_input("Distance to deposit (km)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)

        input_df = pd.DataFrame([{
            "latitude":          lat,
            "longitude":         lon,
            "elevation":         elevation,
            "slope":             slope,
            "distance_to_gold":  distance
        }])

        pred      = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[0][1] * 100

        r1, r2 = st.columns(2)
        r1.metric("Gold Occurrence",  "✅ YES" if pred == 1 else "❌ NO")
        r2.metric("Model Probability", f"{pred_prob:.1f}%")

    except Exception:
        pass  # pkl not present — skip silently

else:
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#888;">'
        '<div style="font-size:4rem;margin-bottom:16px;">🌍</div>'
        '<div style="font-size:1.2rem;font-weight:600;color:#aaa;margin-bottom:8px;">'
        'Click anywhere on the map to start analysis</div>'
        '<div style="font-size:0.9rem;line-height:1.7;max-width:500px;margin:0 auto;">'
        'MineralIQ will query Sentinel-2 satellite imagery, extract 8 spectral mineral proxies '
        '(iron-oxide, clay, NDVI, SAVI, RVI, NDII, MGI, thermal), compute a normalised anomaly '
        'score, render a colour heatmap, and generate a geological AI interpretation report.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )