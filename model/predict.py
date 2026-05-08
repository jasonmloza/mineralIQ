import joblib
import pandas as pd

# load model
model = joblib.load("model/gold_model.pkl")

def predict_gold(lat, lon, elevation, slope, distance):
    data = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "elevation": elevation,
        "slope": slope,
        "distance_to_gold": distance
    }])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return prediction, probability

# test example
if __name__ == "__main__":
    result, prob = predict_gold(-13.9, 33.8, 700, 12, 0.5)
    print("Gold Found:", result)
    print("Probability:", prob)