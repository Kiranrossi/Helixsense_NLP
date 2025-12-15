# app/utils/setfit_content.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def _find_label_column(df: pd.DataFrame) -> str | None:
    for c in ["label", "Label", "class", "Class", "category", "Category", "target", "PredictedCategory"]:
        if c in df.columns:
            return c
    return None


def _compute_cost_weighted_error(
    y_true: pd.Series,
    y_pred: pd.Series,
    amounts: pd.Series,
    cost_factor: float = 1.0,
) -> dict:
    mis_mask = y_true != y_pred
    n_errors = int(mis_mask.sum())
    total_mis_value = float(amounts[mis_mask].sum())
    business_cost = total_mis_value * cost_factor
    return {
        "n_errors": n_errors,
        "total_misclassified_value": total_mis_value,
        "business_cost": business_cost,
        "cost_factor": cost_factor,
    }


def _confusion_matrix_figure(y_true: pd.Series, y_pred: pd.Series):
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Purples",
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion matrix – SetFit semantic model",
    )
    fig.update_layout(
        xaxis_title="Predicted category",
        yaxis_title="True category",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _per_class_f1_comparison_figure(
    y_true: pd.Series,
    y_pred_setfit: pd.Series,
    baseline_metrics: dict | None,
):
    y_true = y_true.astype(str)
    y_pred_setfit = y_pred_setfit.astype(str)

    labels = sorted(y_true.unique())
    f1_setfit = f1_score(y_true, y_pred_setfit, average=None, labels=labels)

    df_rows = []
    for cat, f1_val in zip(labels, f1_setfit):
        df_rows.append({"Category": cat, "Model": "SetFit", "F1": f1_val})

    if baseline_metrics and "per_class_f1" in baseline_metrics:
        for cat in labels:
            if cat in baseline_metrics["per_class_f1"]:
                df_rows.append(
                    {
                        "Category": cat,
                        "Model": "Baseline",
                        "F1": baseline_metrics["per_class_f1"][cat],
                    }
                )

    df_f1 = pd.DataFrame(df_rows)
    fig = px.bar(
        df_f1,
        x="Category",
        y="F1",
        color="Model",
        barmode="group",
        text="F1",
        color_discrete_map={
            "Baseline": "#95A5A6",
            "SetFit": "#8E44AD",
        },
        title="Per‑class F1 – Baseline vs SetFit",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        yaxis_range=[0, 1.05],
        xaxis_title="Category",
        yaxis_title="F1 score",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig, dict(zip(labels, f1_setfit))


def run_setfit_and_render_tab(
    df: pd.DataFrame,
    setfit_model,
    baseline_metrics: dict | None,
) -> dict:
    """
    Evaluate the SetFit semantic model and render the Approach 2 tab.

    setfit_model is assumed to be a callable that takes a list of texts
    and returns a list/array of predicted labels. [file:160]
    """

    label_col = _find_label_column(df)
    if not label_col:
        st.info("No label column found; cannot compute SetFit metrics.")
        return {}

    X_text = df["Remarks"].astype(str).tolist()
    y_true = df[label_col].astype(str)
    y_pred = pd.Series(setfit_model(X_text), index=y_true.index).astype(str)

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # Cost‑weighted error using Debit amounts
    if "Debit" in df.columns:
        cost_info = _compute_cost_weighted_error(
            y_true=y_true,
            y_pred=y_pred,
            amounts=df["Debit"],
            cost_factor=0.1,
        )
    else:
        cost_info = {
            "n_errors": int((y_true != y_pred).sum()),
            "total_misclassified_value": None,
            "business_cost": None,
            "cost_factor": 0.0,
        }

    # Metrics table
    st.subheader("Validation metrics – SetFit")
    metrics_df = pd.DataFrame(
        {
            "metric": [
                "Accuracy",
                "F1‑Weighted (primary)",
                "F1‑Macro",
                "Misclassified records",
                "Misclassified value",
                "Estimated business cost",
            ],
            "value": [
                f"{accuracy:0.3f}",
                f"{f1_weighted:0.3f}",
                f"{f1_macro:0.3f}",
                f"{cost_info['n_errors']}",
                f"{cost_info['total_misclassified_value']:,.0f}"
                if cost_info["total_misclassified_value"] is not None
                else "N/A",
                f"{cost_info['business_cost']:,.0f}"
                if cost_info["business_cost"] is not None
                else "N/A",
            ],
        }
    )
    st.table(metrics_df)

    st.markdown(
        """
Compared with the TF‑IDF baseline, SetFit typically delivers higher **F1‑Weighted**
and lower cost‑weighted error by better handling semantic variations in the remarks. [file:160]  
This is most visible on service‑like transactions, where mis‑classification is
financially expensive. [file:160]
"""
    )

    # Per‑class report
    st.subheader("Per‑class precision/recall/F1 – SetFit")
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(report).transpose()
    st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion matrix – SetFit")
    fig_cm = _confusion_matrix_figure(y_true, y_pred)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per‑class F1 comparison vs baseline
    st.subheader("Per‑class F1 – Baseline vs SetFit")
    fig_f1, per_class_f1_setfit = _per_class_f1_comparison_figure(
        y_true, y_pred, baseline_metrics
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown(
        """
Where SetFit lifts F1 on high‑value classes (especially **Services**), the reduction
in cost‑weighted error translates directly into fewer expensive mis‑tagged transactions. [file:160]  
This is the main reason it is the preferred production model when latency and
infrastructure budgets allow a transformer‑based solution. [file:160]
"""
    )

    metrics = {
        "model_name": "SetFit semantic model",
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "n_errors": cost_info["n_errors"],
        "total_misclassified_value": cost_info["total_misclassified_value"],
        "business_cost": cost_info["business_cost"],
        "cost_factor": cost_info["cost_factor"],
        "per_class_f1": per_class_f1_setfit,
    }
    return metrics
