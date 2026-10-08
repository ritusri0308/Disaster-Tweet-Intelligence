"""
Model Observatory page for CrisisLens.
Presents comprehensive model evaluation metrics, ROC/PR curves, reliability diagrams,
leave-one-event-out generalization, and error analysis.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import REPORTS_DIR, MODEL_PATH
from app.ui.theme import apply_chart_theme
from app.ui.components import render_triage_detail_panel

def render_page(df: pd.DataFrame, pipeline: any):
    st.markdown("<h3 style='margin-top:0;color:#E6EDF7;'>🔬 Model Observatory & Diagnostic Laboratory</h3>", unsafe_allow_html=True)

    # 1. Provenance Banner
    artifacts_file = REPORTS_DIR / "evaluation_artifacts.json"
    artifacts = {}
    if artifacts_file.exists():
        try:
            with open(artifacts_file, "r", encoding="utf-8") as f:
                artifacts = json.load(f)
        except Exception:
            pass

    prov_cols = st.columns(4)
    with prov_cols[0]:
        st.markdown(f"**Model Binary**: `{MODEL_PATH.name}`")
    with prov_cols[1]:
        st.markdown(f"**Evaluated Test Samples**: `{artifacts.get('test_sample_count', 8232):,}`")
    with prov_cols[2]:
        st.markdown(f"**Random Seed**: `{artifacts.get('random_seed', 42)}`")
    with prov_cols[3]:
        eval_time = artifacts.get("evaluation_timestamp", "Active")[:16].replace("T", " ")
        st.markdown(f"**Evaluation Timestamp**: `{eval_time} UTC`")

    st.markdown("<hr style='margin:0.5rem 0 1rem 0;border-color:#1F2A3D;'>", unsafe_allow_html=True)

    tab_overview, tab_curves, tab_generalization, tab_errors = st.tabs([
        "📊 Classifier Performance",
        "📈 ROC & Calibration Curves",
        "🌐 Leave-One-Event-Out Generalization",
        "🔍 Error Analysis Laboratory"
    ])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🔲 Test Confusion Matrix")
            norm_cm = st.checkbox("Normalize Confusion Matrix (%)", value=False)
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
                    labels=dict(x="Predicted Class", y="True Class", color="Percent" if norm_cm else "Count")
                )
                apply_chart_theme(fig_cm)
                fig_cm.update_layout(height=350)
                st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Run `python train.py` to generate the test confusion matrix.")

        with c2:
            st.markdown("##### ⚖️ Classifier Suite Comparison")
            comp_csv = REPORTS_DIR / "model_comparison.csv"
            if comp_csv.exists():
                comp_df = pd.read_csv(comp_csv)
                fig_comp = px.bar(
                    comp_df,
                    x="model_name",
                    y=["dev_macro_f1", "test_macro_f1"],
                    barmode="group",
                    labels={"value": "Macro F1", "model_name": "Architecture"},
                    color_discrete_sequence=["#22D3EE", "#38BDF8"]
                )
                apply_chart_theme(fig_comp)
                fig_comp.update_layout(height=350)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("Model comparison table not found.")

        # Per-class metrics
        st.markdown("##### 🎯 Per-Class Precision, Recall, and F1-Score")
        rep = artifacts.get("per_class_report", {})
        if rep:
            class_rows = []
            for k, v in rep.items():
                if isinstance(v, dict) and "precision" in v:
                    class_rows.append({
                        "Class": k,
                        "Precision": v["precision"],
                        "Recall": v["recall"],
                        "F1-Score": v["f1-score"],
                        "Support": v["support"]
                    })
            if class_rows:
                st.dataframe(pd.DataFrame(class_rows), use_container_width=True)

    with tab_curves:
        c_roc, c_cal = st.columns(2)
        roc_file = REPORTS_DIR / "roc_pr_curves.json"
        cal_file = REPORTS_DIR / "calibration_data.json"

        with c_roc:
            st.markdown("##### ⚡ One-vs-Rest ROC Curves")
            if roc_file.exists():
                with open(roc_file, "r", encoding="utf-8") as f:
                    curves = json.load(f).get("roc_curves", {})
                fig_roc = go.Figure()
                fig_roc.add_shape(type="line", line=dict(dash="dash", color="#64748B"), x0=0, x1=1, y0=0, y1=1)
                for c_name, data in curves.items():
                    fig_roc.add_trace(go.Scatter(
                        x=data["fpr"], y=data["tpr"],
                        mode="lines",
                        name=f"{c_name} (AUC={data.get('auc', 0):.2f})"
                    ))
                apply_chart_theme(fig_roc)
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Run `python evaluate.py` to generate ROC curves.")

        with c_cal:
            st.markdown("##### 🎯 Reliability Diagram & Calibration Curve")
            if cal_file.exists():
                with open(cal_file, "r", encoding="utf-8") as f:
                    cal_data = json.load(f)
                fig_cal = go.Figure()
                fig_cal.add_shape(type="line", line=dict(dash="dash", color="#64748B"), x0=0, x1=1, y0=0, y1=1)
                fig_cal.add_trace(go.Scatter(
                    x=cal_data.get("prob_pred", []),
                    y=cal_data.get("prob_true", []),
                    mode="lines+markers",
                    name="Calibrated Classifier",
                    line=dict(color="#22D3EE")
                ))
                apply_chart_theme(fig_cal)
                ece_val = cal_data.get('expected_calibration_error', 0.0)
                fig_cal.update_layout(
                    title=f"ECE: {ece_val:.4f}",
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Observed Fraction of Positives",
                    height=380
                )
                st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.info("Run `python evaluate.py` to generate calibration diagrams.")

    with tab_generalization:
        st.markdown("##### 🌍 Out-of-Distribution Generalization: Leave-One-Event-Out")
        st.markdown("Evaluates cross-event transferability by holding out entire unseen disasters during training.")

        loeo_csv = REPORTS_DIR / "exp_d_leave_one_event_out.csv"
        if loeo_csv.exists():
            loeo_df = pd.read_csv(loeo_csv)
            st.dataframe(loeo_df, use_container_width=True)

            plot_df = loeo_df[loeo_df["held_out_event"] != "MEAN +/- STD"].copy()
            if not plot_df.empty:
                plot_df["macro_f1"] = pd.to_numeric(plot_df["macro_f1"], errors="coerce")
                fig_loeo = px.bar(
                    plot_df,
                    x="held_out_event",
                    y="macro_f1",
                    labels={"held_out_event": "Held-Out Disaster Event", "macro_f1": "Macro-F1 Score"},
                    color_discrete_sequence=["#F97316"]
                )
                apply_chart_theme(fig_loeo)
                fig_loeo.update_layout(height=350)
                st.plotly_chart(fig_loeo, use_container_width=True)
        else:
            st.info("Leave-one-event-out experiment results not found. Run `python train.py --experiments`.")

    with tab_errors:
        st.markdown("##### 🔍 Error Analysis Laboratory")
        st.markdown("Inspect misclassified dispatches where true label differs from model prediction.")

        sample_errors = artifacts.get("sample_misclassifications", [])
        if sample_errors:
            err_df = pd.DataFrame(sample_errors)

            # Selector by error pair
            c_pair, c_pick = st.columns([1.5, 3.5])
            with c_pair:
                pairs = sorted(list(set(f"{r['true_category']} -> {r['predicted_category']}" for r in sample_errors)))
                selected_pair = st.selectbox("Error Transition Pair", ["All Pairs"] + pairs)

            filtered_errs = err_df
            if selected_pair != "All Pairs":
                t_cat, p_cat = selected_pair.split(" -> ")
                filtered_errs = err_df[(err_df["true_category"] == t_cat) & (err_df["predicted_category"] == p_cat)]

            with c_pick:
                if not filtered_errs.empty:
                    err_idx = st.selectbox(
                        "Select Misclassified Dispatch",
                        options=list(filtered_errs.index),
                        format_func=lambda i: f"#{i} | True: {filtered_errs.loc[i, 'true_category']} | Pred: {filtered_errs.loc[i, 'predicted_category']} (Conf: {filtered_errs.loc[i, 'confidence']:.0%})"
                    )
                else:
                    err_idx = None

            if err_idx is not None and not filtered_errs.empty:
                err_row = filtered_errs.loc[err_idx]
                st.markdown("---")
                render_triage_detail_panel(err_row, pipeline)
        else:
            st.info("No error analysis records available. Run `python evaluate.py` to generate error diagnostic data.")
