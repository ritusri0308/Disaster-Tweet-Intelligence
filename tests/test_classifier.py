"""
Unit tests for classifier module.
"""

import tempfile
from pathlib import Path
import pandas as pd
import joblib
from src.classifier import (
    build_model_pipeline,
    evaluate_model,
    predict_with_confidence,
    train_and_compare_models
)

def test_build_and_train_pipeline():
    train_df = pd.DataFrame({
        "clean_text": [
            "sos trapped people injured in building",
            "need medical supplies emergency help",
            "bridge collapsed road closed infrastructure",
            "heavy rain warning stay safe",
            "please donate to red cross hurricane campaign",
            "watching hurricane news at home"
        ],
        "target": ["Emergency", "Help Request", "Damage Report", "Information/Update", "Information/Update", "General"]
    })

    pipe = build_model_pipeline("logistic_regression")
    pipe.fit(train_df["clean_text"], train_df["target"])

    preds = predict_with_confidence(pipe, ["trapped inside collapsed building near Houston"])
    assert len(preds) == 1
    cat, conf = preds[0]
    assert cat in train_df["target"].unique()
    assert 0.0 <= conf <= 1.0

def test_linear_svc_calibrated_confidence():
    train_df = pd.DataFrame({
        "clean_text": [
            "people trapped under rubble near Houston",
            "sos trapped injured trapped inside building",
            "urgent food supplies needed in shelter",
            "need medical supplies urgently please help",
            "road damage report",
            "bridge collapsed on highway road closed",
            "weather advisory notice",
            "caution heavy rain expected tomorrow",
            "donations welcome for relief fund",
            "please donate to red cross campaign",
            "random tweet about coffee",
            "just watching news coverage at home"
        ],
        "target": [
            "Emergency", "Emergency",
            "Help Request", "Help Request",
            "Damage Report", "Damage Report",
            "Information/Update", "Information/Update",
            "Information/Update", "Information/Update",
            "General", "General"
        ]
    })

    pipe = build_model_pipeline("linear_svm")
    pipe.fit(train_df["clean_text"], train_df["target"])

    preds = predict_with_confidence(pipe, ["sos emergency trapped!"])
    assert len(preds) == 1
    cat, conf = preds[0]
    assert 0.0 <= conf <= 1.0

def test_model_save_load_round_trip():
    train_df = pd.DataFrame({
        "clean_text": ["trapped injured", "road blocked", "donation drive", "weather update"],
        "target": ["Emergency", "Damage Report", "Information/Update", "Information/Update"]
    })
    pipe = build_model_pipeline("naive_bayes")
    pipe.fit(train_df["clean_text"], train_df["target"])

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        joblib.dump(pipe, tmp.name)
        loaded_pipe = joblib.load(tmp.name)
        
        preds_orig = loaded_pipe.predict(["trapped injured"])
        assert len(preds_orig) == 1

def test_evaluate_model():
    df = pd.DataFrame({
        "clean_text": ["sos emergency", "road damage", "weather advice", "weather advice"],
        "target": ["Emergency", "Damage Report", "Information/Update", "Information/Update"]
    })
    pipe = build_model_pipeline("naive_bayes")
    pipe.fit(df["clean_text"], df["target"])
    
    metrics = evaluate_model(pipe, df["clean_text"], df["target"], split_name="dev")
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
