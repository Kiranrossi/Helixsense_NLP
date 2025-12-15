# app/utils/baseline_content.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def _find_label_column(df: pd.DataFrame) -> str | None:
    for c in [
        "label",
        "Label",
        "class",
        "Class",
        "category",
        "Category",
        "target",
        "PredictedCategory",
    ]:
        if c in df.columns:
            return c
    return None


def _compute_cost_weighted_error(
    y_true: pd.Series,
    y_pred: pd.Series,
    amounts: pd.Series,
    cost_factor: float = 1.0,
) -> dict:
    """
    Very simple cost‑weighted view:
    - Identify misclassified rows.
    - Sum their Debit amounts.
    - Optionally scale by cost_factor (e.g. 0.1 for 10% of value viewed as risk).

    Returns dict with:
    - n_errors
    - total_misclassified_value
    - business_cost
    """
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
    # Ensure consistent label types
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Greens",
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion matrix – TF‑IDF + Logistic Regression",
    )
    fig.update_layout(
        xaxis_title="Predicted category",
        yaxis_title="True category",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _per_class_f1_figure(y_true: pd.Series, y_pred: pd.Series):
    # Ensure consistent label types
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    labels = sorted(y_true.unique())
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    df_f1 = pd.DataFrame({"Category": labels, "F1": f1_per_class})
    fig = px.bar(
        df_f1,
        x="Category",
        y="F1",
        text="F1",
        color="Category",
        color_discrete_map={
            "Material": "#FF6B6B",
            "Services": "#4ECDC4",
            "Equipment": "#45B7D1",
        },
        title="Per‑class F1 – Baseline model",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        yaxis_range=[0, 1.05],
        xaxis_title="Category",
        yaxis_title="F1 score",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig, dict(zip(labels, f1_per_class))


def run_baseline_and_render_tab(
    df: pd.DataFrame,
    tfidf_vectorizer,
    tfidf_model,
) -> dict:
    """
    Train/evaluate the TF‑IDF + Logistic Regression baseline and render the tab.

    For simplicity this evaluates on the full dataset; if you later add an
    explicit train/validation split, pass only the validation subset here. [file:160]
    """

    label_col = _find_label_column(df)
    if not label_col:
        st.info("No label column found; cannot compute baseline metrics.")
        return {}

    X = tfidf_vectorizer.transform(df["Remarks"].astype(str))
    y_true = df[label_col].astype(str)  # ensure string labels
    y_pred = pd.Series(tfidf_model.predict(X), index=y_true.index).astype(str)

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
            cost_factor=0.1,  # interpret as “10% of mis‑classified value at risk”
        )
    else:
        cost_info = {
            "n_errors": int((y_true != y_pred).sum()),
            "total_misclassified_value": None,
            "business_cost": None,
            "cost_factor": 0.0,
        }

    # Metric summary table
    st.subheader("Validation metrics – Baseline")
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
- **F1‑Weighted** is the main technical metric, reflecting class imbalance and per‑class performance. [file:160]  
- The **cost‑weighted view** estimates the total value of mis‑classified transactions, scaled by a configurable risk factor, to connect model quality with business impact. [file:160]
"""
    )

    # Detailed classification report
    st.subheader("Per‑class precision/recall/F1")
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    rep_df = pd.DataFrame(report).transpose()
    st.dataframe(rep_df.style.format("{:.3f}"), use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion matrix")
    fig_cm = _confusion_matrix_figure(y_true, y_pred)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per‑class F1 bar chart
    st.subheader("Per‑class F1 scores")
    fig_f1, per_class_f1 = _per_class_f1_figure(y_true, y_pred)
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown(
        """
The baseline is fast, interpretable, and easy to deploy, but it relies purely on
n‑grams and cannot fully capture semantic nuances in the remarks. [file:160]  
The next tab evaluates a SetFit model that is designed to improve F1‑Weighted and
reduce the business cost of mis‑classified high‑value transactions. [file:160]
"""
    )

    # Return metrics dictionary for comparison and report tabs
    metrics = {
        "model_name": "TF‑IDF + Logistic Regression",
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "n_errors": cost_info["n_errors"],
        "total_misclassified_value": cost_info["total_misclassified_value"],
        "business_cost": cost_info["business_cost"],
        "cost_factor": cost_info["cost_factor"],
        "per_class_f1": per_class_f1,
    }
    return metrics
