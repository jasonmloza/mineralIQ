from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    CRS,
    BBox
)

import numpy as np


def fetch_satellite(lat, lon):
    config = SHConfig()

    # Add your Sentinel Hub credentials here
    config.sh_client_id = "YOUR_CLIENT_ID"
    config.sh_client_secret = "YOUR_CLIENT_SECRET"

    bbox = BBox(
        bbox=[lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01],
        crs=CRS.WGS84
    )

    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: ["B04", "B03", "B02"],
        output: { bands: 3 }
      };
    }

    function evaluatePixel(sample) {
      return [sample.B04, sample.B03, sample.B02];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.PNG)
        ],
        bbox=bbox,
        size=(256, 256),
        config=config
    )

    image = request.get_data()[0]
    return image