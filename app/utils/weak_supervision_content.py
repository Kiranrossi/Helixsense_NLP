# app/utils/weak_supervision_content.py

import streamlit as st
import pandas as pd
import plotly.express as px


def _ensure_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a 'PredictedCategory' column for visualization.

    If weak supervision has not yet run and the column is missing, create a
    temporary 'Unknown' category so that the tab can still render. [file:4]
    """
    if "PredictedCategory" not in df.columns:
        tmp = df.copy()
        tmp["PredictedCategory"] = "Unknown"
        return tmp
    return df


def _label_distribution_chart(df: pd.DataFrame):
    """
    Bar chart of silver label distribution.

    Uses PredictedCategory when available, otherwise falls back to a single
    Unknown bucket. Aggregation is done so that the dataframe passed to
    plotly has columns ['PredictedCategory', 'n_records'], avoiding the
    previous mismatch error. [file:4]
    """
    df = _ensure_category_column(df)

    # Group and count by category
    counts = (
        df.groupby("PredictedCategory", dropna=False)
        .size()
        .reset_index(name="n_records")
    )

    # Sort categories in a sensible order
    cat_order = ["Services", "Equipment", "Material", "Unknown"]
    counts["PredictedCategory"] = pd.Categorical(
        counts["PredictedCategory"], categories=cat_order, ordered=True
    )
    counts = counts.sort_values("PredictedCategory")

    fig = px.bar(
        counts,
        x="PredictedCategory",
        y="n_records",
        text="n_records",
        color="PredictedCategory",
        category_orders={"PredictedCategory": cat_order},
        color_discrete_map={
            "Material": "#FF6B6B",
            "Services": "#4ECDC4",
            "Equipment": "#45B7D1",
            "Unknown": "#95A5A6",
        },
        title="Silver Label Distribution",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Number of records",
        yaxis_range=[0, counts["n_records"].max() * 1.3 if len(counts) else 1],
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def _confidence_chart(df: pd.DataFrame):
    """
    Bar chart of label confidence (High vs Low).

    If LowConfidence is missing, everything is treated as high confidence so
    that the UI can still render while wiring up weak supervision. [file:4]
    """
    if "LowConfidence" not in df.columns:
        labels = ["High confidence"]
        values = [len(df)]
    else:
        counts = df["LowConfidence"].value_counts(dropna=False)
        labels = ["High confidence", "Low confidence"]
        values = [counts.get(False, 0), counts.get(True, 0)]

    fig = px.bar(
        x=labels,
        y=values,
        text=[str(v) for v in values],
        labels={"x": "Label confidence", "y": "Number of records"},
        title="High‑ vs Low‑Confidence Silver Labels",
    )

    fig.update_traces(
        marker_color=["#2ECC71", "#E74C3C"][: len(labels)],
        textposition="outside",
    )
    fig.update_layout(
        yaxis_range=[0, max(values) * 1.3 if values else 1],
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def render_nlp_process_tab(df: pd.DataFrame) -> None:
    """
    Render the NLP Process / Weak Supervision tab.

    Works even if weak supervision has not yet run by falling back to an
    'Unknown' category and treating all labels as high‑confidence placeholders. [file:4]
    """
    # --------------------------------------------------------------
    # 1. Data cleaning & feature engineering summary
    # --------------------------------------------------------------
    st.subheader("1. Data cleaning & feature engineering")
    st.markdown(
        """
- **Column pruning:** Dropped `Credit` (all zeros) and `Net` (duplicate of `Debit`) to focus on meaningful financial features. [file:4]
- **Invalid rows:** Removed records with missing or zero `Debit` and empty `Remarks`, leaving a clean expense dataset for modelling. [file:4]
- **Text cleaning:** Normalised `Remarks` to lowercase, removed special characters and noisy numeric artefacts, and stripped extra spaces for stable tokenisation. [file:4]
- **Feature engineering:** Added log‑scaled amount and simple text‑length features to give models numeric context beyond raw text. [file:4]
"""
    )

    st.markdown("---")

    # --------------------------------------------------------------
    # 2. Weak‑supervision framework (conceptual)
    # --------------------------------------------------------------
    st.subheader("2. Weak supervision – generating silver labels")
    st.markdown(
        """
Because the raw dataset has **no ground‑truth category labels**, a weak‑supervision
ensemble is used to generate *silver* labels that are good enough for training. [file:4]
Four complementary labeling functions fire on each remark and transaction amount: [file:4]

- **LF1 – Domain keyword rules:** Detects service, equipment, and material based on facility‑management terms such as consultancy, AMC, dispenser, almirah, pipe, cable. [file:4]
- **LF2 – Cost‑based heuristic:** Uses `Debit` amount bands – very high spend tends to be services, mid‑range looks like equipment, and very small values are often materials. [file:4]
- **LF3 – Zero‑shot transformer:** A BART‑MNLI‑style model scores the remark directly against the three categories to capture semantics beyond raw keywords. [file:4]
- **LF4 – Action‑verb patterns:** Distinguishes service‑style phrases (installation, testing, commissioning, repair) from product‑style descriptions (supply, make, model). [file:4]

Their votes are combined with majority voting to produce a final `PredictedCategory`
and a `LowConfidence` flag when the functions disagree. [file:4]
"""
    )

    st.markdown("---")

    # --------------------------------------------------------------
    # 3. Label distribution
    # --------------------------------------------------------------
    st.subheader("3. Silver label distribution")
    fig_labels = _label_distribution_chart(df)
    st.plotly_chart(fig_labels, use_container_width=True)
    st.markdown(
        """
- **What this shows:** When silver labels are available, this chart reveals the class mix across Services, Equipment, Material (or `Unknown` if labels are not yet generated). [file:4]
- **Why it matters:** The distribution informs model design and supports using F1‑Weighted as the primary metric. [file:4]
"""
    )

    st.markdown("---")

    # --------------------------------------------------------------
    # 4. Label confidence
    # --------------------------------------------------------------
    st.subheader("4. High‑ vs low‑confidence labels")
    fig_conf = _confidence_chart(df)
    st.plotly_chart(fig_conf, use_container_width=True)
    st.markdown(
        """
- **What this shows:** Once weak supervision is wired in, most rows should be high‑confidence, with a smaller low‑confidence slice where the rules disagree. [file:4]
- **Model implication:** High‑confidence rows can be trusted for training and auto‑classification, while low‑confidence ones are ideal for human review loops. [file:4]
"""
    )

    st.markdown("---")

    # --------------------------------------------------------------
    # 5. Sample labelled records
    # --------------------------------------------------------------
    st.subheader("5. Sample labelled transactions")
    df_preview = _ensure_category_column(df)
    cols_to_show = [
        c
        for c in ["Year", "Debit", "Remarks", "PredictedCategory", "LowConfidence"]
        if c in df_preview.columns
    ]
    st.dataframe(df_preview[cols_to_show].head(10), use_container_width=True)
    st.markdown(
        """
These examples illustrate how the ensemble treats typical transactions, mixing
cost signals with domain keywords and transformer‑based semantics once the
silver labels are attached to the dataset. [file:4]
In production, this table helps finance users sanity‑check how automatic tagging
behaves on real ERP lines. [file:4]
"""
    )

def generate_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run LF1–LF4, majority vote, and confidence logic from your notebook.
    Must return df with:
      - PredictedCategory (Services/Equipment/Material)
      - LowConfidence (bool)
    """
    df = df.copy()

    # >>> paste/refactor your notebook code here <<<
    # final_label_series = ...
    # low_conf_series = ...

    df["PredictedCategory"] = final_label_series.astype(str)
    df["LowConfidence"] = low_conf_series.astype(bool)
    return df