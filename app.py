import streamlit as st
import folium
from streamlit_folium import st_folium
import ee

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MineralIQ - Gold Exploration AI",
    layout="wide",
    page_icon="⛏️"
)

st.markdown("""
<style>
.legend-box {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 18px 14px;
    color: white;
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
    width: 28px;
    height: 28px;
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
}
.signal-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}
.signal-card {
    flex: 1;
    min-width: 220px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 14px 16px;
    border-left: 4px solid #D4A017;
}
.signal-name  { font-weight: 600; font-size: 0.95rem; color: #e8c96e; margin-bottom: 4px; }
.signal-value { font-size: 1.4rem; font-weight: 700; color: white; }
.signal-grade { font-size: 0.9rem; color: #e8c96e; }
.signal-interp { color: #aaa; font-size: 0.83rem; margin-top: 6px; line-height: 1.5; }

.verdict-box  {
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 4px;
}
.verdict-title { font-weight: 700; font-size: 1rem; margin-bottom: 8px; }
.verdict-text  { color: #ddd; font-size: 0.9rem; line-height: 1.6; }

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
</style>
""", unsafe_allow_html=True)

# ── Earth Engine init ─────────────────────────────────────────────────────────
@st.cache_resource
def init_ee():
    ee.Initialize(project="mineraliq-ai")

init_ee()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="color:#D4A017; font-size:2.2rem; font-weight:700; margin-bottom:0;">⛏️ MineralIQ — Gold Exploration AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#888; font-size:1rem; margin-top:4px;">Click anywhere on Earth to analyse gold mineralisation potential using satellite spectral signatures</p>', unsafe_allow_html=True)
st.info("👇 Click any location on the map below to begin analysis")

# ── Global base map ───────────────────────────────────────────────────────────
m = folium.Map(location=[0, 20], zoom_start=2, tiles="OpenStreetMap")
map_data = st_folium(m, width="100%", height=500, key="global_map")

# ── On click ─────────────────────────────────────────────────────────────────
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"📍 Selected: **{lat:.5f}°, {lon:.5f}°**")

    with st.spinner("🛰️ Fetching Sentinel-2 imagery and computing spectral indices…"):

        point = ee.Geometry.Point([lon, lat])
        roi   = point.buffer(20000).bounds()

        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate("2023-01-01", "2023-12-31")
            .median()
            .clip(roi)
        )

        iron_oxide = image.select("B4").divide(image.select("B2"))
        clay       = image.select("B11").divide(image.select("B8"))
        gold_score = iron_oxide.add(clay).rename("gold_score")

        stats     = gold_score.reduceRegion(reducer=ee.Reducer.minMax(), geometry=roi, scale=100, maxPixels=1e9)
        gold_norm = gold_score.unitScale(
            ee.Number(stats.get("gold_score_min")),
            ee.Number(stats.get("gold_score_max"))
        ).rename("gold_norm")

        mean_val   = gold_norm.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=100, maxPixels=1e9).get("gold_norm").getInfo()
        confidence = round(float(mean_val) * 100, 1) if mean_val is not None else 0.0

        iron_mean = round(float(
            iron_oxide.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=100, maxPixels=1e9).get("B4").getInfo()
        ), 3)
        clay_mean = round(float(
            clay.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, scale=100, maxPixels=1e9).get("B11").getInfo()
        ), 3)

        rgb_tile = image.getMapId({
            "bands": ["B4", "B3", "B2"], "min": 0, "max": 3000
        })["tile_fetcher"].url_format

        heatmap_tile = gold_norm.getMapId({
            "min": 0, "max": 1, "palette": ["00cc44", "0066ff", "ff1a1a"]
        })["tile_fetcher"].url_format

    # ── Map + Legend ──────────────────────────────────────────────────────────
    st.subheader("🗺️ Satellite Imagery & Gold Probability Heatmap")
    map_col, legend_col = st.columns([4, 1])

    with map_col:
        sat_map = folium.Map(location=[lat, lon], zoom_start=10)
        folium.TileLayer(
            tiles=rgb_tile, attr="Google Earth Engine / Sentinel-2",
            name="Satellite (True Colour)", overlay=True, show=True
        ).add_to(sat_map)
        folium.TileLayer(
            tiles=heatmap_tile, attr="MineralIQ",
            name="Gold Probability Heatmap", overlay=True, show=True, opacity=0.75
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
            '<hr style="border-color:rgba(255,255,255,0.1); margin:16px 0;">'
            '<div class="meta-block">'
            '<b>Data source</b><br>'
            'Sentinel-2 Surface Reflectance<br>'
            'COPERNICUS/S2_SR_HARMONIZED<br>'
            '2023 · 20 km ROI buffer<br><br>'
            '<b>Indices used</b><br>'
            'Iron oxide: B4 ÷ B2<br>'
            'Clay mineral: B11 ÷ B8<br>'
            'Gold score: sum → normalised'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Confidence Score",   f"{confidence:.1f}%")
    m2.metric("🔴 Iron Oxide Index",   iron_mean)
    m3.metric("🔵 Clay Mineral Index", clay_mean)
    m4.metric("📊 Gold Score (norm.)", f"{round(float(mean_val) * 100, 1)}%")
    st.caption("Overall mineralisation signal strength")
    st.progress(int(confidence))

    # ── AI interpretation logic ───────────────────────────────────────────────
    if iron_mean > 1.3:
        iron_grade  = "🔥 Strong"
        iron_interp = ("Elevated iron-oxide ratio strongly suggests hydrothermal alteration — "
                       "a hallmark of gold-bearing systems. Values above 1.3 indicate significant "
                       "ferric iron enrichment from fluid-rock interaction.")
    elif iron_mean > 1.1:
        iron_grade  = "⚠️ Moderate"
        iron_interp = ("Iron-oxide ratio is moderately elevated. This may indicate partial hydrothermal "
                       "alteration or lateritic weathering of iron-rich minerals. Consistent with mineralised corridors.")
    else:
        iron_grade  = "✅ Weak"
        iron_interp = ("Iron-oxide levels are near background. Hydrothermal alteration signal is weak here, "
                       "suggesting limited epithermal gold potential from this proxy alone.")

    if clay_mean > 0.85:
        clay_grade  = "🔥 Strong"
        clay_interp = ("High clay index reveals significant weathering of silicates into phyllosilicates "
                       "(kaolinite, sericite, chlorite). These clay minerals are classic pathfinders for "
                       "orogenic and epithermal gold deposits.")
    elif clay_mean > 0.65:
        clay_grade  = "⚠️ Moderate"
        clay_interp = ("Moderate clay presence indicates secondary mineralisation through weathering. "
                       "Argillic alteration zones of this strength are known to host gold in epithermal environments.")
    else:
        clay_grade  = "✅ Weak"
        clay_interp = ("Low clay mineral signature. Minimal argillic alteration detected. "
                       "This reduces the likelihood of gold-hosting phyllosilicate assemblages at this location.")

    if confidence >= 70:
        verdict_grade  = "HIGH"
        verdict_colour = "#ff4444"
        verdict_text   = (
            "This location shows a strong mineralisation signal (confidence: " + str(confidence) + "%). "
            "The combination of elevated iron-oxide (" + str(iron_mean) + ") and clay indices (" + str(clay_mean) + ") "
            "is consistent with hydrothermal alteration systems that host orogenic or epithermal gold deposits. "
            "Field follow-up and soil geochemistry sampling are recommended. Consider this a priority exploration target."
        )
    elif confidence >= 40:
        verdict_grade  = "MEDIUM"
        verdict_colour = "#4488ff"
        verdict_text   = (
            "This location shows a moderate mineralisation signal (confidence: " + str(confidence) + "%). "
            "Spectral anomalies are present but not dominant — iron-oxide at " + str(iron_mean) + " and clay at " + str(clay_mean) + " "
            "suggest localised alteration. This zone warrants secondary-priority investigation, "
            "particularly if regional geology is favourable."
        )
    else:
        verdict_grade  = "LOW"
        verdict_colour = "#22bb55"
        verdict_text   = (
            "This location shows a weak mineralisation signal (confidence: " + str(confidence) + "%). "
            "Iron-oxide (" + str(iron_mean) + ") and clay (" + str(clay_mean) + ") indices are near background levels. "
            "This location is a low-priority target under the current spectral proxy model. "
            "Consider exploring adjacent areas or applying ASTER SWIR data for deeper clay discrimination."
        )

    # ── AI panel — built with string concatenation, no f-strings inside HTML ─
    ai_html = (
        '<div class="ai-panel">'
          '<div class="ai-panel-title">🤖 AI Geological Explanation</div>'
          '<div class="signal-row">'

            '<div class="signal-card" style="border-left-color:#ff6b35;">'
              '<div class="signal-name">🔴 Iron Oxide Index (B4 ÷ B2)</div>'
              '<div class="signal-value">' + str(iron_mean) + ' <span class="signal-grade">' + iron_grade + '</span></div>'
              '<div class="signal-interp">' + iron_interp + '</div>'
            '</div>'

            '<div class="signal-card" style="border-left-color:#4488ff;">'
              '<div class="signal-name">🔵 Clay Mineral Index (B11 ÷ B8)</div>'
              '<div class="signal-value">' + str(clay_mean) + ' <span class="signal-grade">' + clay_grade + '</span></div>'
              '<div class="signal-interp">' + clay_interp + '</div>'
            '</div>'

          '</div>'

          '<div class="verdict-box" style="border:1px solid ' + verdict_colour + '66; background:' + verdict_colour + '18;">'
            '<div class="verdict-title" style="color:' + verdict_colour + ';">⚖️ Overall Verdict: ' + verdict_grade + ' MINERALISATION POTENTIAL</div>'
            '<div class="verdict-text">' + verdict_text + '</div>'
          '</div>'

          '<div class="disclaimer">'
            '<b>⚠️ Disclaimer:</b> MineralIQ uses rule-based spectral heuristics derived from Sentinel-2 multispectral imagery. '
            'Results are indicative only and should be validated against field geology, geochemical sampling, and regional structural data. '
            'This tool is not a substitute for professional geological survey.'
          '</div>'
        '</div>'
    )

    st.markdown(ai_html, unsafe_allow_html=True)

else:
    st.markdown(
        '<div style="text-align:center; padding:40px; color:#888;">'
        '<div style="font-size:3rem; margin-bottom:12px;">🌍</div>'
        '<div style="font-size:1.1rem; font-weight:600; color:#aaa;">Click anywhere on the map to start analysis</div>'
        '<div style="font-size:0.9rem; margin-top:8px; line-height:1.6;">'
        'MineralIQ will query Sentinel-2 satellite data, compute spectral mineral proxies,'
        ' and generate a gold probability heatmap with AI geological interpretation.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )