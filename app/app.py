# app/app.py

import streamlit as st
import pandas as pd

from utils.load_models import load_all_models_and_data
from utils import rag_chat

# Tab content modules
from utils.eda_content import render_eda_tab
from utils.weak_supervision_content import render_nlp_process_tab
from utils.baseline_content import run_baseline_and_render_tab
from utils.setfit_content import run_setfit_and_render_tab
from utils.comparison_content import render_comparison_tab
from utils.report_generator import render_report_tab


# --------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Helixsense – Facility Expense NLP",
    layout="wide",
)


# --------------------------------------------------------------------
# Cached loading of data and models
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data, labels, and models...")
def init_resources():
    """
    Central place to load everything once and share across tabs.

    Expected from load_all_models_and_data():

        df            : pandas.DataFrame with at least
                        ['Year','Debit','Remarks','PredictedCategory','LowConfidence']
        tfidf_vec     : fitted TfidfVectorizer
        tfidf_model   : fitted LogisticRegression
        setfit_model  : callable or SetFitModel-like object
    """
    df, tfidf_vec, tfidf_model, setfit_model = load_all_models_and_data()
    return df, tfidf_vec, tfidf_model, setfit_model


df, tfidf_vectorizer, tfidf_model, setfit_model = init_resources()


# --------------------------------------------------------------------
# Shared metric container
# --------------------------------------------------------------------
if "baseline_metrics" not in st.session_state:
    st.session_state.baseline_metrics = None

if "setfit_metrics" not in st.session_state:
    st.session_state.setfit_metrics = None


# --------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------
st.sidebar.title("Helixsense NLP Demo")
st.sidebar.markdown(
    "This internal demo showcases an end‑to‑end NLP pipeline that tags "
    "facility expense transactions into **Services**, **Equipment**, and "
    "**Material** using only the free‑text remarks column.\n\n"
    "Use the **Home** tab for the chatbot and business overview. "
    "Subsequent tabs walk through EDA, NLP labeling, the two model approaches, "
    "and a final comparison and report."
)


# --------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------
tabs = st.tabs(
    [
        "Home",
        "EDA",
        "NLP Process",
        "Approach 1",
        "Approach 2",
        "Model Comparison",
        "Report",
    ]
)

# ====================================================================
# Home tab
# ====================================================================
with tabs[0]:
    st.header("Business Problem")

    st.markdown(
        """
The finance and operations teams need a reliable way to classify facility expense
transactions into **Services**, **Equipment**, and **Material** using only the free‑text
*Remarks* column in the ERP export (`data.xlsx`).  

Today this tagging is mostly manual, which makes it:
- Slow and error‑prone for analysts.
- Difficult to get a consistent view of spend by category.
- Risky when high‑value service contracts are mis‑classified into low‑value buckets.

This app demonstrates a production‑style NLP pipeline that uses weak supervision and
two model approaches to automate this tagging. F1‑Weighted is used as the primary
technical metric (because classes are imbalanced), and a cost‑weighted view translates
misclassifications into business impact.
        """
    )

    st.markdown("---")
    st.subheader("Chat with AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"""
<div style="display:flex; justify-content:flex-end; margin:4px 0;">
  <div style="
      max-width:70%;
      background-color:#4B8BF4;
      color:white;
      padding:8px 12px;
      border-radius:12px 12px 2px 12px;
      ">
    <strong>You:</strong> {msg["content"]}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div style="display:flex; justify-content:flex-start; margin:4px 0;">
  <div style="
      max-width:70%;
      background-color:#262730;
      color:white;
      padding:8px 12px;
      border-radius:12px 12px 12px 2px;
      ">
    <strong>ChatAI:</strong> {msg["content"]}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    user_msg = st.chat_input("Ask anything, or ask about this project...")

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        history_pairs = [(m["role"], m["content"]) for m in st.session_state.chat_history]

        answer = rag_chat.chat_ai(
            question=user_msg,
            history=history_pairs,
            df=df,
            tfidf_vec=tfidf_vectorizer,
            tfidf_model=tfidf_model,
            setfit_model=setfit_model,
        )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("Clear chat", key="chat_clear_button"):
        st.session_state.chat_history = []
        st.rerun()


# ====================================================================
# EDA tab
# ====================================================================
with tabs[1]:
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset snapshot")
    st.write(df.head())

    st.markdown("---")
    render_eda_tab(df)


# ====================================================================
# NLP Process tab
# ====================================================================
with tabs[2]:
    st.header("NLP Process – Weak Supervision & Labeling")

    render_nlp_process_tab(df)


# ====================================================================
# Approach 1 tab (Baseline)
# ====================================================================
with tabs[3]:
    st.header("Approach 1 – TF‑IDF Logistic Regression")

    st.markdown(
        """
This baseline uses a classic **TF‑IDF + Logistic Regression** pipeline.

- TF‑IDF converts each remark into a sparse bag‑of‑words representation.
- Logistic Regression with class weighting handles the class imbalance.
- F1‑Weighted on the validation set is the primary technical metric.
- A cost‑weighted view translates misclassifications into estimated analyst effort
  and financial impact.
        """
    )

    baseline_metrics = run_baseline_and_render_tab(
        df=df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_model=tfidf_model,
    )
    st.session_state.baseline_metrics = baseline_metrics


# ====================================================================
# Approach 2 tab (SetFit)
# ====================================================================
with tabs[4]:
    st.header("Approach 2 – SetFit Semantic Model")

    st.markdown(
        """
The second approach uses **SetFit**, a sentence‑transformer based model:

- Pretrained sentence embeddings capture semantics beyond keywords.
- Contrastive learning fine‑tunes the encoder on weakly‑supervised labels.
- A lightweight classification head predicts **Services / Equipment / Material**.
- F1‑Weighted and a cost‑weighted view are again used to evaluate performance.
        """
    )

    baseline_metrics = st.session_state.baseline_metrics

    setfit_metrics = run_setfit_and_render_tab(
        df=df,
        setfit_model=setfit_model,
        baseline_metrics=baseline_metrics,
    )
    st.session_state.setfit_metrics = setfit_metrics


# ====================================================================
# Model Comparison tab
# ====================================================================
with tabs[5]:
    st.header("Model Comparison – F1‑Weighted and Cost‑Weighted Error")

    baseline_metrics = st.session_state.baseline_metrics
    setfit_metrics = st.session_state.setfit_metrics

    if not baseline_metrics or not setfit_metrics:
        st.info(
            "Please visit **Approach 1** and **Approach 2** tabs once to compute "
            "metrics before viewing the comparison."
        )
    else:
        render_comparison_tab(
            baseline_metrics=baseline_metrics,
            setfit_metrics=setfit_metrics,
        )


# ====================================================================
# Report tab
# ====================================================================
with tabs[6]:
    st.header("Executive Report & Download")

    baseline_metrics = st.session_state.baseline_metrics
    setfit_metrics = st.session_state.setfit_metrics

    if not baseline_metrics or not setfit_metrics:
        st.info(
            "Please open **Approach 1**, **Approach 2**, and **Model Comparison** "
            "tabs first so that metrics and insights are available for the report."
        )
    else:
        render_report_tab(
            df=df,
            baseline_metrics=baseline_metrics,
            setfit_metrics=setfit_metrics,
        )
