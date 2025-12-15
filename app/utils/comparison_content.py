# app/utils/comparison_content.py

import streamlit as st
import pandas as pd
import plotly.express as px


def render_comparison_tab(
    baseline_metrics: dict,
    setfit_metrics: dict,
) -> None:
    """
    Render the Model Comparison tab.

    Expects both metrics dicts to contain:
    - model_name
    - f1_weighted
    - f1_macro
    - accuracy
    - n_errors
    - total_misclassified_value
    - business_cost
    """

    st.subheader("1. Metric overview")

    rows = []
    for m in [baseline_metrics, setfit_metrics]:
        if not m:
            continue
        rows.append(
            {
                "Model": m.get("model_name", "Model"),
                "Accuracy": m.get("accuracy"),
                "F1‑Weighted": m.get("f1_weighted"),
                "F1‑Macro": m.get("f1_macro"),
                "Misclassified records": m.get("n_errors"),
                "Misclassified value": m.get("total_misclassified_value"),
                "Estimated business cost": m.get("business_cost"),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df_display = summary_df.copy()

    for col in ["Accuracy", "F1‑Weighted", "F1‑Macro"]:
        summary_df_display[col] = summary_df_display[col].map(
            lambda x: f"{x:0.3f}" if pd.notnull(x) else "N/A"
        )

    for col in ["Misclassified value", "Estimated business cost"]:
        summary_df_display[col] = summary_df_display[col].map(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "N/A"
        )

    st.table(summary_df_display)

    st.markdown(
        """
- **F1‑Weighted** is the primary technical metric because the silver labels are moderately imbalanced across Services, Equipment, and Material. [file:160]  
- The **cost‑weighted view** converts mis‑classified transaction value into an estimated business cost, making it easier for finance stakeholders to compare models. [file:160]
"""
    )

    st.markdown("---")

    # 2. F1‑Weighted comparison
    st.subheader("2. F1‑Weighted comparison")

    df_f1 = pd.DataFrame(
        {
            "Model": [baseline_metrics["model_name"], setfit_metrics["model_name"]],
            "F1‑Weighted": [
                baseline_metrics["f1_weighted"],
                setfit_metrics["f1_weighted"],
            ],
        }
    )

    fig_f1 = px.bar(
        df_f1,
        x="Model",
        y="F1‑Weighted",
        text="F1‑Weighted",
        color="Model",
        color_discrete_map={
            baseline_metrics["model_name"]: "#95A5A6",
            setfit_metrics["model_name"]: "#8E44AD",
        },
        range_y=[0, 1],
        title="F1‑Weighted – Baseline vs SetFit",
    )
    fig_f1.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_f1.update_layout(
        xaxis_title="Model",
        yaxis_title="F1‑Weighted",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown(
        """
An uplift in **F1‑Weighted** for SetFit indicates that it is correcting more
errors across all classes, with particular gains usually on high‑value services. [file:160]
"""
    )

    st.markdown("---")

    # 3. Cost‑weighted error comparison
    st.subheader("3. Cost‑weighted error comparison")

    cost_df = pd.DataFrame(
        {
            "Model": [baseline_metrics["model_name"], setfit_metrics["model_name"]],
            "Business cost": [
                baseline_metrics["business_cost"],
                setfit_metrics["business_cost"],
            ],
        }
    )

    fig_cost = px.bar(
        cost_df,
        x="Model",
        y="Business cost",
        text="Business cost",
        color="Model",
        color_discrete_map={
            baseline_metrics["model_name"]: "#95A5A6",
            setfit_metrics["model_name"]: "#8E44AD",
        },
        title="Estimated cost from mis‑classified transactions",
    )
    fig_cost.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_cost.update_layout(
        xaxis_title="Model",
        yaxis_title="Estimated cost (currency units)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown(
        """
Here the **height of each bar** approximates the annual financial exposure from
wrong tags, given a fixed risk factor applied to the mis‑classified transaction
value. [file:160]  
A meaningfully lower bar for SetFit indicates that semantic modeling reduces
financial risk, even if the raw number of errors is similar. [file:160]
"""
    )

    st.markdown("---")

    # 4. Recommendation
    st.subheader("4. Recommended model")

    if (
        setfit_metrics["f1_weighted"] >= baseline_metrics["f1_weighted"]
        and setfit_metrics["business_cost"] <= baseline_metrics["business_cost"]
    ):
        st.markdown(
            f"""
- **Recommendation:** Promote **{setfit_metrics['model_name']}** as the primary
production model for facility‑expense tagging. [file:160]  
- **Rationale:** It achieves higher F1‑Weighted and lowers the estimated
cost‑weighted error versus the TF‑IDF baseline, which is especially important
because service transactions drive a majority of total spend. [file:160]
"""
        )
    else:
        st.markdown(
            f"""
- **Recommendation:** Keep **{baseline_metrics['model_name']}** as the default
until SetFit consistently outperforms it on both F1‑Weighted and cost‑weighted
error. [file:160]  
- **Next steps:** Collect more labelled data and iterate on the weak‑supervision
rules to strengthen the semantic model before switching. [file:160]
"""
        )
