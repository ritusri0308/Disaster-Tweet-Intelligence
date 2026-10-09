"""
Model Observatory page for CrisisLens.
Presents comprehensive model evaluation metrics, ROC/PR curves, reliability diagrams,
leave-one-event-out generalization, experiments, and error analysis.

Every number shown here is read from the evaluation artifacts in reports/. When an artifact is missing the
page says so; it never substitutes a made-up value.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_PATH, REPORTS_DIR
from src.model_registry import read_model_metadata
from app.ui.components import render_callout, render_empty, render_kpi_card, render_triage_detail_panel
from app.ui.theme import ACCENT_LIGHT, ACCENT_PRIMARY, NEUTRAL_LINE, apply_chart_theme

EXPERIMENT_FILES = [
    ("exp_a_ngrams.csv", "Unigrams vs unigrams + bigrams"),
    ("exp_b_text_format.csv", "Raw vs cleaned text"),
    ("exp_c_class_balancing.csv", "With vs without class balancing"),
    ("exp_e_target_schema.csv", "5-class project taxonomy vs raw 10-class labels"),
    ("exp_f_char_ngrams.csv", "Word n-grams vs word + character n-grams"),
]

def _read_json(path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

def _parse_time(value: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None

def render_page(df: pd.DataFrame, pipeline: any):
    artifacts = _read_json(REPORTS_DIR / "evaluation_artifacts.json")
    calibration = _read_json(REPORTS_DIR / "calibration_data.json")
    meta = read_model_metadata(MODEL_PATH)

    # Evaluation artifacts older than the loaded model describe a different model
    evaluated_at = _parse_time(artifacts.get("evaluation_timestamp"))
    trained_at = _parse_time(meta.get("trained_at"))
    if evaluated_at and trained_at and trained_at > evaluated_at + timedelta(minutes=1):
        render_callout(
            "The model on disk was trained after these evaluation reports were generated, so the charts below "
            "describe an earlier model. Run `python evaluate.py` to refresh them.",
            "warn", "Evaluation reports are out of date."
        )

    # 1. Provenance KPIs
    median_conf = float(df["confidence"].median()) if "confidence" in df.columns and not df.empty else None
    overall = artifacts.get("overall_metrics", {})
    macro_f1 = overall.get("macro_f1", artifacts.get("test_macro_f1"))
    test_samples = artifacts.get("test_sample_count")
    ece = calibration.get("expected_calibration_error", overall.get("expected_calibration_error"))

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Median confidence", f"{median_conf:.1%}" if median_conf is not None else "n/a", "Predicted probability")
    with k2:
        render_kpi_card("Test macro-F1", f"{macro_f1:.2f}" if macro_f1 is not None else "n/a", "Benchmark score")
    with k3:
        render_kpi_card("Test samples", f"{test_samples:,}" if test_samples is not None else "n/a", "Out-of-sample")
    with k4:
        name = str(meta.get("selected_model", "")).replace("_", " ").title() or MODEL_PATH.stem.replace("_", " ").title()
        render_kpi_card("Active model", name, f"version {meta.get('model_version', 'unknown')}")

    if meta:
        st.markdown(
            f"<div class='cl-muted' style='margin-top:0.75rem;'>Trained {meta.get('trained_at', 'unknown')[:19].replace('T', ' ')} UTC · "
            f"{meta.get('vocab_size', 0):,} vocabulary terms · {meta.get('n_train', 0):,} training tweets"
            + (f" · selection: {meta['selection_rule']}" if meta.get("selection_rule") else "")
            + "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    tab_overview, tab_curves, tab_generalization, tab_experiments, tab_errors = st.tabs([
        "Classifier performance",
        "ROC & calibration",
        "Leave-one-event-out",
        "Experiments",
        "Error analysis"
    ])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='cl-section'>Confusion Matrix</div>", unsafe_allow_html=True)
            norm_cm = st.checkbox("Normalize confusion matrix (%)", value=False)
            cm_csv = REPORTS_DIR / "best_model_confusion_matrix.csv"
            if cm_csv.exists():
                cm_df = pd.read_csv(cm_csv, index_col=0)
                mat_vals = cm_df.values
                if norm_cm:
                    row_sums = mat_vals.sum(axis=1, keepdims=True)
                    mat_vals = np.round((mat_vals / np.maximum(row_sums, 1)) * 100, 1)

                fig_cm = px.imshow(
                    mat_vals,
                    x=[c.replace("Pred_", "") for c in cm_df.columns],
                    y=[r.replace("True_", "") for r in cm_df.index],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted class", y="True class", color="Percent" if norm_cm else "Count")
                )
                apply_chart_theme(fig_cm)
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, width="stretch")
            else:
                render_empty("Run training pipeline to generate the test confusion matrix.")

        with c2:
            st.markdown("<div class='cl-section'>Model Architecture Comparison</div>", unsafe_allow_html=True)
            comp_csv = REPORTS_DIR / "model_comparison.csv"
            if comp_csv.exists():
                comp_df = pd.read_csv(comp_csv)
                fig_comp = px.bar(
                    comp_df,
                    x="model_name",
                    y=["dev_macro_f1", "test_macro_f1"],
                    barmode="group",
                    labels={"value": "Macro F1", "model_name": "Model architecture"},
                    color_discrete_sequence=[ACCENT_PRIMARY, ACCENT_LIGHT]
                )
                apply_chart_theme(fig_comp)
                fig_comp.update_layout(height=350, yaxis_range=[0, 1])
                st.plotly_chart(fig_comp, width="stretch")
                st.caption("Models within 0.005 dev macro-F1 of the best are treated as tied and the simpler one is selected.")
            else:
                render_empty("Model comparison table not found.")

        # Per-class metrics
        st.markdown("<div class='cl-section'>Per-Class Precision, Recall, and F1-Score</div>", unsafe_allow_html=True)
        rep = artifacts.get("per_class_report", {})
        class_rows = [
            {"Class": k, "Precision": v["precision"], "Recall": v["recall"], "F1-Score": v["f1-score"], "Support": v["support"]}
            for k, v in rep.items() if isinstance(v, dict) and "precision" in v
        ]
        if class_rows:
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")
        else:
            render_empty("Per-class report not found. Run `python evaluate.py`.")

    with tab_curves:
        c_roc, c_cal = st.columns(2)
        roc_file = REPORTS_DIR / "roc_pr_curves.json"

        with c_roc:
            st.markdown("<div class='cl-section'>One-vs-Rest ROC Curves</div>", unsafe_allow_html=True)
            curves = _read_json(roc_file).get("roc_curves", {})
            if curves:
                fig_roc = go.Figure()
                fig_roc.add_shape(type="line", line=dict(dash="dash", color=NEUTRAL_LINE), x0=0, x1=1, y0=0, y1=1)
                for c_name, data in curves.items():
                    fig_roc.add_trace(go.Scatter(
                        x=data["fpr"], y=data["tpr"],
                        mode="lines",
                        name=f"{c_name} (AUC={data.get('auc', 0):.2f})"
                    ))
                apply_chart_theme(fig_roc)
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
                st.plotly_chart(fig_roc, width="stretch")
            else:
                render_empty("ROC curves data not found.")

        with c_cal:
            st.markdown("<div class='cl-section'>Reliability Diagram (Calibration)</div>", unsafe_allow_html=True)
            if calibration:
                fig_cal = go.Figure()
                fig_cal.add_shape(type="line", line=dict(dash="dash", color=NEUTRAL_LINE), x0=0, x1=1, y0=0, y1=1)
                fig_cal.add_trace(go.Scatter(
                    x=calibration.get("prob_pred", []),
                    y=calibration.get("prob_true", []),
                    mode="lines+markers",
                    name="Classifier",
                    line=dict(color=ACCENT_PRIMARY)
                ))
                apply_chart_theme(fig_cal)
                fig_cal.update_layout(
                    title=f"Expected calibration error: {float(ece or 0.0):.4f}",
                    xaxis_title="Mean predicted probability",
                    yaxis_title="Observed fraction correct",
                    height=380
                )
                st.plotly_chart(fig_cal, width="stretch")
                st.caption("Points on the dashed diagonal mean the stated confidence matches how often the model is right. "
                           "Class-balanced training deliberately trades some calibration for better recall on rare classes.")
            else:
                render_empty("Calibration data not found.")

    with tab_generalization:
        st.markdown("<div class='cl-section'>Cross-Event Transfer (Leave-One-Event-Out)</div>", unsafe_allow_html=True)
        st.markdown("<div class='cl-muted' style='margin-bottom:0.75rem;'>Evaluates cross-event generalization by holding out an entire disaster event during training.</div>", unsafe_allow_html=True)

        loeo_csv = REPORTS_DIR / "exp_d_leave_one_event_out.csv"
        if loeo_csv.exists():
            loeo_df = pd.read_csv(loeo_csv)
            st.dataframe(loeo_df, hide_index=True, width="stretch")

            plot_df = loeo_df[loeo_df["held_out_event"] != "MEAN +/- STD"].copy()
            if not plot_df.empty:
                plot_df["macro_f1"] = pd.to_numeric(plot_df["macro_f1"], errors="coerce")
                fig_loeo = px.bar(
                    plot_df,
                    x="held_out_event",
                    y="macro_f1",
                    labels={"held_out_event": "Held-out disaster event", "macro_f1": "Macro-F1 score"},
                    color_discrete_sequence=[ACCENT_PRIMARY]
                )
                apply_chart_theme(fig_loeo)
                fig_loeo.update_layout(height=350)
                st.plotly_chart(fig_loeo, width="stretch")
        else:
            render_empty("Leave-one-event-out experiment results not found. Run `python train.py --experiments`.")

        per_event_csv = REPORTS_DIR / "per_event_metrics.csv"
        if per_event_csv.exists():
            st.markdown("<div class='cl-section'>Per-event test performance</div>", unsafe_allow_html=True)
            st.dataframe(pd.read_csv(per_event_csv), hide_index=True, width="stretch")

    with tab_experiments:
        st.markdown("<div class='cl-section'>Ablation experiments</div>", unsafe_allow_html=True)
        st.markdown("<div class='cl-muted' style='margin-bottom:0.75rem;'>Each experiment changes one thing and re-measures macro-F1. Generated by <code>python train.py --experiments</code>.</div>", unsafe_allow_html=True)
        found_any = False
        for filename, title in EXPERIMENT_FILES:
            path = REPORTS_DIR / filename
            if path.exists():
                found_any = True
                st.markdown(f"**{title}**")
                st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")
        if not found_any:
            render_empty("No experiment results found. Run `python train.py --experiments`.")

    with tab_errors:
        st.markdown("<div class='cl-section'>Error Analysis Laboratory</div>", unsafe_allow_html=True)
        st.markdown("<div class='cl-muted' style='margin-bottom:0.75rem;'>Inspect misclassified dispatches where true label differs from model prediction.</div>", unsafe_allow_html=True)

        sample_errors = artifacts.get("sample_misclassifications", [])
        if sample_errors:
            err_df = pd.DataFrame(sample_errors)

            c_pair, c_pick = st.columns([1.5, 3.5])
            with c_pair:
                pairs = sorted(list(set(f"{r['true_category']} -> {r['predicted_category']}" for r in sample_errors)))
                selected_pair = st.selectbox("Error transition pair", ["All Pairs"] + pairs)

            filtered_errs = err_df
            if selected_pair != "All Pairs":
                t_cat, p_cat = selected_pair.split(" -> ")
                filtered_errs = err_df[(err_df["true_category"] == t_cat) & (err_df["predicted_category"] == p_cat)]

            with c_pick:
                if not filtered_errs.empty:
                    err_idx = st.selectbox(
                        "Select misclassified report",
                        options=list(filtered_errs.index),
                        format_func=lambda i: f"#{i} | True: {filtered_errs.loc[i, 'true_category']} | Pred: {filtered_errs.loc[i, 'predicted_category']} (Conf: {filtered_errs.loc[i, 'confidence']:.0%})"
                    )
                else:
                    err_idx = None

            if err_idx is not None and not filtered_errs.empty:
                err_row = filtered_errs.loc[err_idx]
                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                render_triage_detail_panel(err_row, pipeline)
        else:
            render_empty("No error analysis records available.")
