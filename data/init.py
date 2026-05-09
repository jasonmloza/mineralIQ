"""
MineralIQ model package
"""
from .feature_extractor import extract_spectral_features, compute_anomaly_score
from .predict import predict_gold, load_model

__all__ = [
    "extract_spectral_features",
    "compute_anomaly_score",
    "predict_gold",
    "load_model",
]