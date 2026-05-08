import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# load dataset
data = pd.read_csv("data/training_data.csv")

# features (what AI looks at)
X = data[["latitude", "longitude", "elevation", "slope", "distance_to_gold"]]

# label (what we want to predict)
y = data["gold"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# test accuracy
print("Accuracy:", model.score(X_test, y_test))

# save model
joblib.dump(model, "model/gold_model.pkl")