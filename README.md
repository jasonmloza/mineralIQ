# MineralIQ - Gold Probability AI

MineralIQ is a machine learning web application that estimates the probability of gold occurrence based on geological and geographic land features.

## Live Demo
Try the app here: https://huggingface.co/spaces/JasonMloza/MineralIQ

## Features
- Predicts gold occurrence probability
- Streamlit web interface
- Machine learning classification model
- Public deployment on Hugging Face Spaces

## Input Parameters
Users provide:

- Latitude
- Longitude
- Elevation
- Slope
- Distance to known gold deposits

The model then predicts:

- Gold Found: YES / NO
- Probability score (%)

## Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Joblib

## Project Structure

```bash
MineralIQ/
│── streamlit_app.py
│── requirements.txt
│── README.md
│
├── data/
│   └── training_data.csv
│
└── model/
    ├── train.py
    ├── predict.py
    ├── feature_extractor.py
    └── gold_model.pkl
```

## Installation

Clone repository:

```bash
git clone https://github.com/jasonmloza/mineralIQ.git
cd mineralIQ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run streamlit_app.py
```

## Model Workflow
1. Training data is loaded from CSV
2. Features are extracted
3. Model is trained using XGBoost
4. Trained model is saved as `.pkl`
5. App loads model for live predictions

## Future Improvements
- Satellite imagery integration
- Geological survey datasets
- Interactive maps
- Probability calibration improvements
- Multi-mineral prediction support

## Disclaimer
This project is an experimental prototype for educational and research purposes.
Predictions should not be used as sole evidence for mining or investment decisions.

## Author
Jason Mloza

GitHub: https://github.com/jasonmloza