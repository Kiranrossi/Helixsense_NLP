# app/utils/eda_content.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def _ensure_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 'PredictedCategory' column.

    If weak supervision has not yet run and the column is missing, create a
    temporary 'Unknown' category so that EDA can still render.[file:160]
    """
    if "PredictedCategory" not in df.columns:
        tmp = df.copy()
        tmp["PredictedCategory"] = "Unknown"
        return tmp
    return df


def _prepare_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction volume and value by PredictedCategory.

    Expects df to contain:
    - 'PredictedCategory' (Equipment / Services / Material / Unknown)
    - 'Debit' (numeric transaction value)
    """
    df = _ensure_category_column(df)

    grouped = (
        df.groupby("PredictedCategory")["Debit"]
        .agg(["count", "sum", "mean", "median"])
        .reset_index()
    )
    grouped.rename(
        columns={
            "count": "TransactionCount",
            "sum": "TotalSpend",
            "mean": "AvgAmount",
            "median": "MedianAmount",
        },
        inplace=True,
    )

    total_txn = grouped["TransactionCount"].sum()
    total_spend = grouped["TotalSpend"].sum()

    grouped["PctTransactions"] = (grouped["TransactionCount"] / total_txn * 100).round(1)
    grouped["PctSpend"] = (grouped["TotalSpend"] / total_spend * 100).round(1)
    grouped["ValueVolumeRatio"] = (
        grouped["PctSpend"] / grouped["PctTransactions"].replace(0, np.nan)
    ).round(2)

    return grouped


def _volume_value_dashboard(categorystats: pd.DataFrame) -> go.Figure:
    colors = {
        "Material": "#FF6B6B",
        "Services": "#4ECDC4",
        "Equipment": "#45B7D1",
        "Unknown": "#95A5A6",
    }
    colorlist = [colors.get(cat, "#95A5A6") for cat in categorystats["PredictedCategory"]]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Transaction Count by Category", "Total Spend by Category"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.12,
    )

    # Left: transaction volume
    fig.add_trace(
        go.Bar(
            x=categorystats["PredictedCategory"],
            y=categorystats["TransactionCount"],
            text=categorystats["TransactionCount"].astype(str),
            textposition="outside",
            textfont=dict(size=14, color="black", family="Arial Black"),
            marker=dict(color=colorlist, line=dict(color="black", width=1.5)),
            name="Volume",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Percentage: %{customdata:.1f}%",
            customdata=categorystats["PctTransactions"],
        ),
        row=1,
        col=1,
    )

    # Right: spend
    fig.add_trace(
        go.Bar(
            x=categorystats["PredictedCategory"],
            y=categorystats["TotalSpend"],
            text=[f"{val/1e6:0.1f}M" for val in categorystats["TotalSpend"]],
            textposition="outside",
            textfont=dict(size=14, color="black", family="Arial Black"),
            marker=dict(color=colorlist, line=dict(color="black", width=1.5)),
            name="Value",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Spend: %{y:,.0f}<br>Percentage: %{customdata:.1f}%",
            customdata=categorystats["PctSpend"],
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Category", row=1, col=1, tickfont=dict(size=12))
    fig.update_xaxes(title_text="Category", row=1, col=2, tickfont=dict(size=12))
    fig.update_yaxes(title_text="Number of Transactions", row=1, col=1, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Total Spend", row=1, col=2, tickfont=dict(size=11))

    fig.update_layout(
        title_text="Facility Expense Analysis – Volume vs Value",
        title_font=dict(size=18, color="#2C3E50", family="Arial Black"),
        bargap=0.35,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def _amount_distribution_boxplot(df: pd.DataFrame) -> go.Figure:
    df = _ensure_category_column(df)

    fig = px.box(
        df,
        x="PredictedCategory",
        y="Debit",
        color="PredictedCategory",
        color_discrete_map={
            "Material": "#FF6B6B",
            "Services": "#4ECDC4",
            "Equipment": "#45B7D1",
            "Unknown": "#95A5A6",
        },
        log_y=True,
        points="outliers",
    )
    fig.update_layout(
        title="Transaction Amount Distribution by Category (log scale)",
        xaxis_title="Category",
        yaxis_title="Debit (log scale)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _class_balance_chart(categorystats: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        categorystats,
        x="PredictedCategory",
        y="PctTransactions",
        color="PredictedCategory",
        color_discrete_map={
            "Material": "#FF6B6B",
            "Services": "#4ECDC4",
            "Equipment": "#45B7D1",
            "Unknown": "#95A5A6",
        },
        text="PctTransactions",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        title="Class Balance – Share of Transactions",
        xaxis_title="Category",
        yaxis_title="Percentage of Transactions",
        yaxis_range=[0, max(categorystats["PctTransactions"]) * 1.3],
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _confidence_distribution_chart(df: pd.DataFrame) -> go.Figure:
    if "LowConfidence" not in df.columns:
        # If weak supervision not run yet, just show a placeholder chart
        labels = ["High confidence"]
        values = [len(df)]
    else:
        counts = df["LowConfidence"].value_counts(dropna=False)
        labels = ["High confidence", "Low confidence"]
        values = [counts.get(False, 0), counts.get(True, 0)]

    fig = px.bar(
        x=labels,
        y=values,
        text=[f"{v}" for v in values],
        labels={"x": "Label confidence", "y": "Number of records"},
        title="Silver Label Confidence Distribution",
    )
    fig.update_traces(marker_color=["#2ECC71", "#E74C3C"][: len(labels)], textposition="outside")
    fig.update_layout(
        yaxis_range=[0, max(values) * 1.3 if values else 1],
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _text_complexity_heatmap(df: pd.DataFrame) -> go.Figure:
    df = _ensure_category_column(df)

    tmp = df.copy()
    tmp["remark_word_count"] = tmp["Remarks"].astype(str).str.split().apply(len)
    tmp["remark_char_count"] = tmp["Remarks"].astype(str).str.len()

    text_stats = (
        tmp.groupby("PredictedCategory")[["remark_word_count", "remark_char_count"]]
        .mean()
        .reset_index()
    )

    z = [
        text_stats["remark_word_count"].values.tolist(),
        text_stats["remark_char_count"].values.tolist(),
    ]
    x = text_stats["PredictedCategory"].tolist()
    y = ["Avg words", "Avg characters"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            showscale=True,
            hovertemplate="Metric: %{y}<br>Category: %{x}<br>Value: %{z:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Text Complexity by Category",
        xaxis_title="Category",
        yaxis_title="Metric",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def render_eda_tab(df: pd.DataFrame) -> None:
    """
    Render the full EDA tab in Streamlit.
    """

    # ------------------------------------------------------------------
    # 1. Category‑level statistics & volume–value paradox
    # ------------------------------------------------------------------
    st.subheader("1. Volume–Value Disconnect")

    categorystats = _prepare_category_stats(df)

    st.dataframe(
        categorystats[
            [
                "PredictedCategory",
                "TransactionCount",
                "PctTransactions",
                "TotalSpend",
                "PctSpend",
                "ValueVolumeRatio",
            ]
        ],
        use_container_width=True,
    )

    fig_vv = _volume_value_dashboard(categorystats)
    st.plotly_chart(fig_vv, use_container_width=True)

    max_volume_cat = categorystats.loc[
        categorystats["TransactionCount"].idxmax(), "PredictedCategory"
    ]
    max_value_cat = categorystats.loc[
        categorystats["TotalSpend"].idxmax(), "PredictedCategory"
    ]

    st.markdown(
        f"""
- **What this shows:** {max_volume_cat} has the highest number of transactions, while {max_value_cat} drives most of the total spend, creating a clear volume–value disconnect.[file:160]  
- **Key insight:** Misclassifying {max_value_cat} transactions is far more expensive, so the evaluation focuses on F1‑Weighted and cost‑weighted error rather than raw accuracy.[file:160]  
- **Model implication:** Models must prioritize precision and recall on high‑value service‑like spend, even if that means slightly lower performance on low‑value, high‑volume classes.[file:160]
"""
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. Amount distribution
    # ------------------------------------------------------------------
    st.subheader("2. Amount Distribution by Category")

    fig_box = _amount_distribution_boxplot(df)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown(
        """
- **What this shows:** Transaction amounts are highly skewed, with a long tail of very large service contracts and a dense cluster of small equipment and material purchases.[file:160]  
- **Key insight:** Using a log scale highlights that most records are low value but the financial risk sits in the right‑hand tail of the distribution.[file:160]  
- **Model implication:** Classifiers should be evaluated with cost‑weighted metrics so that rare high‑amount errors are penalized more heavily than frequent low‑amount mistakes.[file:160]
"""
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # 3. Class balance
    # ------------------------------------------------------------------
    st.subheader("3. Class Balance (Silver Labels)")

    fig_cls = _class_balance_chart(categorystats)
    st.plotly_chart(fig_cls, use_container_width=True)

    st.markdown(
        """
- **What this shows:** The silver labels are moderately imbalanced, with Equipment representing the largest share of transactions and Material clearly the minority class.[file:160]  
- **Key insight:** A simple accuracy metric would be biased toward the majority class and under‑report errors on smaller classes.[file:160]  
- **Model implication:** F1‑Weighted becomes the primary technical metric to reflect both imbalance and per‑class performance fairly.[file:160]
"""
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # 4. Label confidence
    # ------------------------------------------------------------------
    st.subheader("4. Silver Label Confidence")

    fig_conf = _confidence_distribution_chart(df)
    st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown(
        """
- **What this shows:** Around 70–75% of records have agreement from at least two labeling functions, while the rest are flagged as low‑confidence.[file:160]  
- **Key insight:** This level of label noise is realistic in production and acceptable for SetFit, which is robust to imperfect labels.[file:160]  
- **Model implication:** High‑confidence records can be trusted for training, while low‑confidence ones may be down‑weighted or targeted for human review in production.[file:160]
"""
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # 5. Text complexity
    # ------------------------------------------------------------------
    st.subheader("5. Text Complexity of Remarks")

    fig_text = _text_complexity_heatmap(df)
    st.plotly_chart(fig_text, use_container_width=True)

    st.markdown(
        """
- **What this shows:** Equipment descriptions tend to be slightly longer on average, providing richer keyword and semantic signals than shorter material descriptions.[file:160]  
- **Key insight:** Both TF‑IDF (keyword‑driven) and SetFit (semantic) models can benefit from this structure, but SetFit is particularly strong when remarks are longer and more descriptive.[file:160]  
- **Model implication:** The mixed range of text lengths reinforces the need to benchmark both approaches with F1‑Weighted and cost‑weighted views before selecting the production model.[file:160]
"""
    )
