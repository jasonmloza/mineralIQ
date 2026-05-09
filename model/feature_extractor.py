import ee

def extract_spectral_features(image):

    features = []

    # Iron oxide (red / blue)
    iron_oxide = image.select('B4').divide(image.select('B2')).rename('iron_oxide')

    # Clay minerals (SWIR / NIR)
    clay = image.select('B11').divide(image.select('B8')).rename('clay')

    # Vegetation stress (NDVI)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')

    # SAVI (soil-adjusted vegetation index)
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': image.select('B8'), 'RED': image.select('B4')}
    ).rename('savi')

    # Red ratio (iron staining proxy)
    rvi = image.select('B4').divide(image.select('B3')).rename('rvi')

    # Moisture / clay proxy
    ndii = image.normalizedDifference(['B8', 'B11']).rename('ndii')

    # Magnesium / alteration proxy
    mgi = image.select('B9').divide(image.select('B11')).rename('mgi')

    # Thermal proxy (FIXED: B10 removed → use B12)
    thermal_ratio = image.select('B12').divide(image.select('B11')).rename('thermal_ratio')

    features = [
        iron_oxide,
        clay,
        ndvi,
        savi,
        rvi,
        ndii,
        mgi,
        thermal_ratio
    ]

    return ee.Image.cat(features)
def compute_anomaly_score(feature_image, roi):

    anomaly = feature_image.reduce(ee.Reducer.sum())

    stats = anomaly.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=30,
        bestEffort=True
    )

    min_v = ee.Number(stats.get('sum_min'))
    max_v = ee.Number(stats.get('sum_max'))

    anomaly_normalized = ee.Image(
        ee.Algorithms.If(
            max_v.neq(min_v),
            anomaly.unitScale(min_v, max_v),
            anomaly.multiply(0)
        )
    )

    return anomaly_normalized