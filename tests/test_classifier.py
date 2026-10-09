"""
Unit tests for classifier module.
"""

import tempfile
import pandas as pd
import joblib
from src.classifier import (
    build_model_pipeline,
    evaluate_model,
    predict_with_confidence
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

def test_train_and_compare_models_small(tmp_path):
    from src.classifier import train_and_compare_models
    train_df = pd.DataFrame({
        "clean_text": [
            "sos emergency trapped", "people trapped injured",
            "need rescue food", "urgent food needed please",
            "bridge collapsed", "road broken infrastructure damaged",
            "weather forecast rain", "storm advisory notice",
            "praying for safety", "watching news"
        ],
        "target": [
            "Emergency", "Emergency",
            "Help Request", "Help Request",
            "Damage Report", "Damage Report",
            "Information/Update", "Information/Update",
            "General", "General"
        ]
    })
    dev_df = train_df.copy()
    test_df = train_df.copy()

    # Explicit output paths: training must never write to the production model/report locations.
    model_path = tmp_path / "models" / "candidate.joblib"
    res = train_and_compare_models(train_df, dev_df, test_df, model_path=model_path, reports_dir=tmp_path / "reports")
    assert "best_model_name" in res
    assert "comparison_table" in res
    assert not res["comparison_table"].empty
    assert model_path.exists()
    assert (tmp_path / "reports" / "model_comparison.csv").exists()

    # The metadata sidecar identifies the saved model by content fingerprint.
    from src.model_registry import fingerprint_file, read_model_metadata
    meta = read_model_metadata(model_path)
    assert meta["model_version"] == fingerprint_file(model_path)
    assert meta["selected_model"] == res["best_model_name"]
    assert meta["n_train"] == len(train_df)

def test_select_model_prefers_the_simpler_model_when_scores_are_within_noise():
    import pytest
    from src.classifier import select_model

    # The real retraining run: a 162 MB random forest led by 0.002 dev macro-F1. Within noise, so the simpler model wins.
    scores = {"naive_bayes": 0.7051, "logistic_regression": 0.7577, "linear_svm": 0.7594, "random_forest": 0.7598}
    assert select_model(scores) == "logistic_regression"

    # A clearly better complex model still wins
    assert select_model({"logistic_regression": 0.75, "random_forest": 0.80}) == "random_forest"
    # Tolerance is configurable; with zero tolerance the top score wins outright
    assert select_model(scores, tolerance=0.0) == "random_forest"
    # Ties among simple models resolve by preference order
    assert select_model({"naive_bayes": 0.70, "linear_svm": 0.70}) == "linear_svm"
    with pytest.raises(ValueError):
        select_model({})

def test_non_linear_models_get_no_invented_attributions():
    from src.explainability import explain_classification
    texts = ["sos trapped under rubble", "need food and water", "bridge collapsed road closed", "praying for everyone"] * 4
    labels = ["Emergency", "Help Request", "Damage Report", "General"] * 4
    forest = build_model_pipeline("random_forest").fit(texts, labels)
    explanation = explain_classification(forest, "people trapped under rubble")
    assert explanation["is_surrogate"] is True
    assert explanation["top_positive_features"] == [] and explanation["top_negative_features"] == []
    assert explanation["predicted_class"] in set(labels)

def test_char_ngram_option_trains_and_explanations_only_show_word_features():
    from src.explainability import explain_classification
    texts = ["sos trapped under rubble", "need food and water", "bridge collapsed road closed", "praying for everyone"] * 4
    labels = ["Emergency", "Help Request", "Damage Report", "General"] * 4

    word_only = build_model_pipeline("logistic_regression").fit(texts, labels)
    with_chars = build_model_pipeline("logistic_regression", use_char_ngrams=True).fit(texts, labels)
    assert len(with_chars.named_steps["tfidf"].get_feature_names_out()) > len(word_only.named_steps["tfidf"].get_feature_names_out())

    explanation = explain_classification(with_chars, "people trapped under rubble")
    assert explanation["predicted_class"] == "Emergency"
    tokens = [t for t, _ in explanation["top_positive_features"]]
    assert tokens and all("__" not in t and not t.startswith(" ") for t in tokens)
    assert "trapped" in tokens or "rubble" in tokens

def test_training_skips_models_that_cannot_fit_and_fails_clearly_if_none_can(tmp_path):
    from src.classifier import train_and_compare_models
    import pytest

    # One example per class: the calibrated SVM (2-fold CV) cannot be fitted, the others can.
    one_each = pd.DataFrame({
        "clean_text": ["sos trapped", "need food", "bridge collapsed", "advisory issued", "praying for all"],
        "target": ["Emergency", "Help Request", "Damage Report", "Information/Update", "General"],
    })
    res = train_and_compare_models(one_each, one_each, one_each,
                                   model_path=tmp_path / "m.joblib", reports_dir=tmp_path)
    trained = set(res["comparison_table"]["model_name"])
    assert "linear_svm" not in trained
    assert trained  # at least one candidate survived

    # A single class cannot train anything at all: fail loudly instead of saving nothing
    single_class = pd.DataFrame({"clean_text": ["a b", "c d"], "target": ["General", "General"]})
    with pytest.raises(ValueError):
        train_and_compare_models(single_class, single_class, single_class,
                                 model_path=tmp_path / "n.joblib", reports_dir=tmp_path)

def test_predict_with_confidence_matches_predict():
    """The single-pass prediction must agree with pipeline.predict for probabilistic models."""
    texts = ["sos trapped injured", "need food water", "bridge collapsed road closed",
             "advisory evacuation order", "praying for everyone", "volunteers donations relief"]
    labels = ["Emergency", "Help Request", "Damage Report", "Information/Update", "General", "Information/Update"]
    for model_name in ["naive_bayes", "logistic_regression", "linear_svm", "random_forest"]:
        pipe = build_model_pipeline(model_name)
        pipe.fit(texts * 3, labels * 3)
        probe = ["trapped under rubble", "water and food please", "nothing relevant here"]
        expected = list(pipe.predict(probe))
        got = predict_with_confidence(pipe, probe)
        assert [c for c, _ in got] == expected, model_name
        assert all(0.0 <= conf <= 1.0 for _, conf in got)
