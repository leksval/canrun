"""
End-to-End Offline Data Science Pipeline for CanRun FPS Prediction.

Modules:
- collect_and_audit: Multi-source benchmark & Steam Store spec ingestion + 4 physical invariant gates
- features: 18 leak-free continuous domain features for hardware, resolution, and game requirements
- train_and_evaluate: Two-stage Physics-Informed Log-Ridge Prior + Monotonic Huber GBDT + GroupKFold CV + Split-Conformal calibration + Pure-JSON export
"""
