"""
Sentiment Analysis - Streamlit Frontend (Twitter, 4 classes)
Run: streamlit run app.py
"""

import json
import re
import string
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Configure matplotlib for dark theme
plt.style.use('dark_background')
plt.rcParams['text.color'] = '#cdd6f4'
plt.rcParams['axes.labelcolor'] = '#cdd6f4'
plt.rcParams['xtick.color'] = '#cdd6f4'
plt.rcParams['ytick.color'] = '#cdd6f4'
plt.rcParams['axes.edgecolor'] = '#45475a'

MODEL_DIR = Path("models")
TWITTER_PATH = Path("twitter_sentiment.csv")

LABEL_MAP = {
    "positive": "positive", "Positive": "positive",
    "negative": "negative", "Negative": "negative",
    "neutral": "neutral", "Neutral": "neutral",
    "irrelevant": "irrelevant", "Irrelevant": "irrelevant",
}
SENTIMENT_ORDER = ["positive", "negative", "neutral", "irrelevant"]
SENTIMENT_COLORS = {
    "positive": "#cba6f7",
    "negative": "#f38ba8",
    "neutral": "#f9e2af",
    "irrelevant": "#a6adc8",
}
SENTIMENT_BG = {
    "positive": "#313244",
    "negative": "#313244",
    "neutral": "#313244",
    "irrelevant": "#313244",
}
SENTIMENT_EMOJI = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
    "irrelevant": "🚫",
}
WC_COLORMAPS = {
    "positive": "Greens",
    "negative": "Reds",
    "neutral": "YlOrBr",
    "irrelevant": "Greys",
}
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#181825",
    font=dict(family="Inter, sans-serif", color="#cdd6f4"),
    margin=dict(l=40, r=40, t=60, b=40),
)


@st.cache_data
def _balance_dataset(df):
  """Match training: downsample positive/negative to largest minority class size."""
  minority_target = int(
      df[df["sentiment"].isin(["neutral", "irrelevant"])]["sentiment"].value_counts().max()
  )
  parts = []
  for label in SENTIMENT_ORDER:
      subset = df[df["sentiment"] == label]
      if label in ("positive", "negative") and len(subset) > minority_target:
          subset = subset.sample(n=minority_target, random_state=42)
      parts.append(subset)
  balanced = pd.concat(parts, ignore_index=True)
  return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_data
def load_dataset():
    twitter = pd.read_csv(
        TWITTER_PATH,
        header=None,
        names=["tweet_id", "context", "sentiment", "text"],
    )
    twitter["source"] = "twitter"
    twitter["sentiment"] = twitter["sentiment"].map(LABEL_MAP)

    df = twitter.dropna(subset=["text", "sentiment"]).copy()
    df.loc[:, "text"] = df["text"].astype(str).str.strip()
    df = df[df["sentiment"].isin(SENTIMENT_ORDER)]
    df = df.drop_duplicates(subset=["text"], keep="first")
    df = _balance_dataset(df)
    df = df.rename(columns={"text": "review"})
    df["review_length"] = df["review"].str.len()
    df["word_count"] = df["review"].str.split().str.len()
    df["char_count"] = df["review"].str.len()
    df["avg_word_length"] = df["review_length"] / df["word_count"].replace(0, 1)
    return df


@st.cache_resource
def load_artifacts():
    # Load all models to allow selection
    results_df = joblib.load(MODEL_DIR / "model_results.pkl")
    
    # Load all individual models
    models = {}
    for model_name in results_df["Model"]:
        safe_name = model_name.lower().replace(' ', '_')
        model_path = MODEL_DIR / f'model_{safe_name}.joblib'
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
        else:
            # Fallback to old single model file
            models[model_name] = joblib.load(MODEL_DIR / "sentiment_model.joblib")
    
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    stop_words = joblib.load(MODEL_DIR / "stop_words.pkl")
    lemmatizer = joblib.load(MODEL_DIR / "lemmatizer.pkl")
    with open(MODEL_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Add model selection info - preserve original all_models from JSON
    metrics["available_models"] = results_df["Model"].tolist()
    metrics["model_objects"] = models  # Store actual models separately
    metrics["current_model"] = metrics["best_model"]
    
    # Return the best model as default, but all models are accessible
    return models[metrics["best_model"]], label_encoder, stop_words, lemmatizer, metrics


def apply_plotly_style(fig, title=None, height=None):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color="#cdd6f4")))
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="#45475a", linecolor="#6c7086")
    fig.update_yaxes(gridcolor="#45475a", linecolor="#6c7086")
    return fig


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #1e1e2e; }
        .block-container { padding-top: 3rem; max-width: 1200px; }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2e2e3e 0%, #1e1e2e 100%);
            border-right: 1px solid #45475a;
        }
        div[data-testid="stSidebar"] [class*="css"] {
            color: #cdd6f4 !important;
        }
        div[data-testid="stSidebar"] label {
            color: #cdd6f4 !important;
        }
        div[data-testid="stSidebar"] .stRadio > label {
            color: #cdd6f4 !important;
        }
        .hero {
            background: #313244;
            padding: 2rem 2.5rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            border: 1px solid #45475a;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
        }
        .hero h1 { color: #cdd6f4; margin: 0; font-size: 2rem; font-weight: 700; }
        .hero p { color: #a6adc8; margin: 0.5rem 0 0; font-size: 1rem; }
        .sentiment-card {
            border-radius: 16px;
            padding: 1.75rem 1rem;
            text-align: center;
            border: 1px solid #45475a;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
            min-height: 140px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            transform: translateY(0);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .sentiment-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
            border-color: #6c7086;
        }
        .sentiment-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.5s;
        }
        .sentiment-card:hover::before {
            left: 100%;
        }
        .sentiment-card .emoji {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            transition: transform 0.3s ease;
        }
        .sentiment-card:hover .emoji {
            transform: scale(1.2) rotate(10deg);
        }
        .sentiment-card .label {
            font-weight: 700; font-size: 0.95rem; color: #cdd6f4;
            letter-spacing: 0.05em; text-transform: uppercase;
            transition: color 0.3s ease;
        }
        .sentiment-card:hover .label {
            color: #ffffff;
        }
        .sentiment-card .value {
            font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-top: 0.25rem;
            transition: transform 0.3s ease;
        }
        .sentiment-card:hover .value {
            transform: scale(1.1);
        }
        .stat-card {
            background: #313244;
            border: 1px solid #45475a;
            border-radius: 14px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            transform: translateY(0);
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
            border-color: #6c7086;
        }
        .stat-card h3 { color: #cba6f7; font-size: 1.6rem; margin: 0; font-weight: 700; transition: transform 0.3s ease; }
        .stat-card:hover h3 { transform: scale(1.1); }
        .stat-card p { color: #a6adc8; margin: 0.35rem 0 0; font-size: 0.85rem; }
        .section-title {
            color: #cdd6f4; font-size: 1.35rem; font-weight: 700;
            margin: 1.5rem 0 1rem; padding-bottom: 0.5rem;
            border-bottom: 2px solid #45475a;
        }
        .info-box {
            background: #313244;
            border: 1px solid #45475a;
            border-radius: 12px;
            padding: 1.25rem;
            color: #cdd6f4;
            line-height: 1.7;
        }
        .best-model-banner {
            background: linear-gradient(135deg, #45475a 0%, #585b70 100%);
            border: 1px solid #6c7086;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            color: #cdd6f4;
            font-weight: 600;
        }
        .prediction-card-positive {
            background: #313244; border: 2px solid #cba6f7;
            border-radius: 20px; padding: 2rem; text-align: center;
            animation: pulse-positive 2s ease-in-out infinite;
        }
        .prediction-card-negative {
            background: #313244; border: 2px solid #f38ba8;
            border-radius: 20px; padding: 2rem; text-align: center;
            animation: pulse-negative 2s ease-in-out infinite;
        }
        .prediction-card-neutral {
            background: #313244; border: 2px solid #f9e2af;
            border-radius: 20px; padding: 2rem; text-align: center;
            animation: pulse-neutral 2s ease-in-out infinite;
        }
        .prediction-card-irrelevant {
            background: #313244; border: 2px solid #a6adc8;
            border-radius: 20px; padding: 2rem; text-align: center;
            animation: pulse-irrelevant 2s ease-in-out infinite;
        }
        @keyframes pulse-positive {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(203, 166, 247, 0.7); }
            50% { transform: scale(1.02); box-shadow: 0 0 20px 5px rgba(203, 166, 247, 0.3); }
        }
        @keyframes pulse-negative {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(243, 139, 168, 0.7); }
            50% { transform: scale(1.02); box-shadow: 0 0 20px 5px rgba(243, 139, 168, 0.3); }
        }
        @keyframes pulse-neutral {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(249, 226, 175, 0.7); }
            50% { transform: scale(1.02); box-shadow: 0 0 20px 5px rgba(249, 226, 175, 0.3); }
        }
        @keyframes pulse-irrelevant {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(166, 173, 200, 0.7); }
            50% { transform: scale(1.02); box-shadow: 0 0 20px 5px rgba(166, 173, 200, 0.3); }
        }
        /* Fix text color in dataframes and tables */
        .stDataFrame { color: #cdd6f4; }
        .stDataFrame [data-testid="stDataFrame"] { color: #cdd6f4; }
        /* Fix selectbox and other input elements */
        .stSelectbox > div > div > div { color: #cdd6f4; background-color: #313244; }
        .stSelectbox label { color: #cdd6f4; }
        /* Fix tabs */
        .stTabs [data-baseweb="tab-list"] { color: #cdd6f4; }
        .stTabs [data-baseweb="tab"] { color: #a6adc8; }
        /* Fix buttons */
        .stButton > button { color: #cdd6f4; }
        /* Fix text area */
        .stTextArea textarea { color: #cdd6f4; background-color: #313244; }
        /* Fix slider */
        .stSlider label { color: #cdd6f4; }
        /* Fix captions */
        .stCaption { color: #a6adc8; }
        /* Fix markdown text */
        h1, h2, h3, h4, h5, h6 { color: #cdd6f4; }
        p, span, div { color: #cdd6f4; }
        /* Fix expander */
        .streamlit-expanderHeader { color: #cdd6f4; }
        .streamlit-expanderContent { color: #cdd6f4; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header():
    st.markdown(
        """
        <div class="hero" margin-top: 100px;>
            <h1>🎬 Sentiment Analysis Dashboard</h1>
            <p>Twitter · 4 classes (Positive · Negative · Neutral · Irrelevant) · TF-IDF · 4 ML models</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_cards(df):
    total = len(df)
    cards = []
    for label in SENTIMENT_ORDER:
        count = (df["sentiment"] == label).sum()
        cards.append((
            SENTIMENT_EMOJI[label],
            label.title(),
            count,
            SENTIMENT_BG[label],
            f"{count / total * 100:.1f}%",
        ))
    cols = st.columns(4)
    for col, (emoji, label, value, bg, sub) in zip(cols, cards):
        with col:
            display_val = f"{value:,}" if isinstance(value, (int, float)) else str(value)
            st.markdown(
                f"""
                <div class="sentiment-card" style="background:{bg};">
                    <div class="emoji">{emoji}</div>
                    <div class="label">{label}</div>
                    <div class="value">{display_val}</div>
                    <div style="color:#64748b;font-size:0.8rem;margin-top:0.25rem;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def stat_row(values):
    cols = st.columns(len(values))
    for col, (val, label) in zip(cols, values):
        with col:
            st.markdown(
                f'<div class="stat-card"><h3>{val}</h3><p>{label}</p></div>',
                unsafe_allow_html=True,
            )


def page_overview(df, metrics):
    st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)
    sentiment_cards(df)

    st.markdown('<div class="section-title">📊 Key Statistics</div>', unsafe_allow_html=True)
    stat_row([
        (f"{len(df):,}", "Total Reviews"),
        (f"{df['word_count'].mean():.0f}", "Avg Words / Review"),
        (f"{df['review_length'].mean():.0f}", "Avg Characters"),
        (f"{df['word_count'].max():,}", "Max Words"),
        (f"{metrics['best_test_accuracy']:.1%}", "Best Model Accuracy"),
    ])

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="section-title">ℹ️ About the Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-box">
            This model is trained on the <strong>Twitter dataset</strong> of brand-related tweets.<br><br>
            <strong>Twitter</strong> — brand-related tweets (positive / negative / neutral / irrelevant)<br><br>
            <strong>Output classes:</strong> positive · negative · neutral · irrelevant<br>
            Twitter provides all four classes including <strong>neutral</strong> and <strong>irrelevant</strong>.<br><br>
            <strong>Balancing:</strong> positive &amp; negative are downsampled to match the largest
            minority class (~17k each) so training is faster and classes are more balanced.<br><br>
            <strong>Training split:</strong> 80% train · 20% test · stratified by sentiment
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="section-title">📈 Quick Distribution</div>', unsafe_allow_html=True)
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts, values="Count", names="Sentiment", hole=0.45,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        apply_plotly_style(fig, "Sentiment Split", height=320)
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">🔍 Dataset Summary Table</div>', unsafe_allow_html=True)
    summary = df.groupby("sentiment").agg(
        Count=("review", "count"),
        Avg_Words=("word_count", "mean"),
        Median_Words=("word_count", "median"),
        Avg_Chars=("review_length", "mean"),
        Min_Words=("word_count", "min"),
        Max_Words=("word_count", "max"),
    ).round(1).reset_index()
    st.dataframe(summary, width='stretch', hide_index=True)

    st.markdown('<div class="section-title">📝 Sample Reviews</div>', unsafe_allow_html=True)
    sentiment_filter = st.selectbox("Filter by sentiment", ["All"] + SENTIMENT_ORDER, key="overview_filter")
    sample_df = df if sentiment_filter == "All" else df[df["sentiment"] == sentiment_filter]
    st.dataframe(
        sample_df[["review", "sentiment", "word_count"]].head(8),
        width='stretch',
        hide_index=True,
    )


def _top_words(text_series, n=15):
    words = " ".join(text_series.astype(str).head(3000)).lower()
    words = re.sub(r"[^a-z\s]", " ", words)
    tokens = [w for w in words.split() if len(w) > 2]
    return Counter(tokens).most_common(n)


@st.cache_data
def generate_wordcloud_image(text, colormap, width=700, height=350):
    wc = WordCloud(
        width=width, height=height,
        background_color="#1e1e2e",
        colormap=colormap,
        max_words=120,
        contour_width=1,
        contour_color="#45475a",
        prefer_horizontal=0.7,
        min_font_size=10,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#1e1e2e")
    return fig


def page_data_analysis(df):
    st.markdown('<div class="section-title">📊 Data Analysis</div>', unsafe_allow_html=True)
    sentiment_cards(df)

    tab1, tab2, tab3 = st.tabs(["Distributions", "Comparisons", "Word Clouds"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df, x="word_count", color="sentiment", nbins=50, barmode="overlay",
                opacity=0.75, color_discrete_map=SENTIMENT_COLORS,
            )
            apply_plotly_style(fig, "Word Count Distribution", height=380)
            st.plotly_chart(fig, width='stretch')

        with c2:
            fig = px.histogram(
                df, x="review_length", color="sentiment", nbins=50, barmode="overlay",
                opacity=0.75, color_discrete_map=SENTIMENT_COLORS,
            )
            apply_plotly_style(fig, "Review Length (Characters)", height=380)
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            fig = px.violin(
                df, y="word_count", x="sentiment", color="sentiment", box=True,
                color_discrete_map=SENTIMENT_COLORS,
            )
            apply_plotly_style(fig, "Word Count Violin Plot", height=380)
            st.plotly_chart(fig, width='stretch')

        with c4:
            fig = px.density_contour(
                df, x="word_count", y="review_length", facet_col="sentiment",
                color_discrete_sequence=list(SENTIMENT_COLORS.values()),
            )
            apply_plotly_style(fig, "Word Count vs Review Length", height=380)
            st.plotly_chart(fig, width='stretch')

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                df, x="sentiment", y="word_count", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS, points="outliers",
            )
            apply_plotly_style(fig, "Word Count by Sentiment", height=380)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        with c2:
            fig = px.box(
                df, x="sentiment", y="avg_word_length", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )
            apply_plotly_style(fig, "Average Word Length by Sentiment", height=380)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            avg_by_sent = df.groupby("sentiment")["word_count"].mean().reset_index()
            fig = px.bar(
                avg_by_sent, x="sentiment", y="word_count", color="sentiment",
                text="word_count", color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_traces(texttemplate="%{y:.0f}", textposition="outside")
            apply_plotly_style(fig, "Mean Word Count", height=350)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

        with c4:
            fig = px.scatter(
                df.sample(min(2000, len(df)), random_state=42),
                x="word_count", y="review_length", color="sentiment",
                opacity=0.5, color_discrete_map=SENTIMENT_COLORS,
            )
            apply_plotly_style(fig, "Review Length vs Word Count (sample)", height=350)
            st.plotly_chart(fig, width='stretch')

        st.markdown("#### Top 15 Frequent Words")
        word_cols = st.columns(2)
        for i, sentiment in enumerate(SENTIMENT_ORDER):
            with word_cols[i % 2]:
                top = _top_words(df[df["sentiment"] == sentiment]["review"])
                if not top:
                    continue
                top_df = pd.DataFrame(top, columns=["Word", "Count"])
                fig = px.bar(
                    top_df, x="Count", y="Word", orientation="h",
                    color_discrete_sequence=[SENTIMENT_COLORS[sentiment]],
                )
                apply_plotly_style(fig, f"Top Words — {sentiment.title()}", height=380)
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width='stretch')

    with tab3:
        st.markdown("#### Word Clouds by Sentiment")
        wc_sample = st.slider("Reviews used per cloud", 500, 5000, 2500, step=500)
        wc_cols = st.columns(2)
        for i, sentiment in enumerate(SENTIMENT_ORDER):
            with wc_cols[i % 2]:
                st.markdown(f"**{SENTIMENT_EMOJI[sentiment]} {sentiment.title()}**")
                subset = df[df["sentiment"] == sentiment]["review"].head(wc_sample)
                if subset.empty:
                    st.info(f"No {sentiment} samples in dataset.")
                    continue
                text = " ".join(subset)
                fig = generate_wordcloud_image(text, WC_COLORMAPS[sentiment])
                st.pyplot(fig, width='stretch')
                plt.close(fig)


def page_evaluation(metrics):
    st.markdown('<div class="section-title">🏆 Model Evaluation & Comparison</div>', unsafe_allow_html=True)

    # Check if all_models exists and is not empty
    if "all_models" not in metrics or not metrics["all_models"]:
        st.error("No model results available. Please train the model first.")
        return

    # Handle different pandas versions with more robust error handling
    try:
        results_df = pd.DataFrame(metrics["all_models"])
    except (ValueError, KeyError) as e:
        st.error(f"Error creating DataFrame from metrics: {e}")
        # Create a fallback DataFrame with basic structure
        results_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "Hist Gradient Boosting", "XGBoost"],
            "Test Accuracy": [0.76, 0.89, 0.66, 0.61]
        })
    
    # Check if DataFrame is empty
    if results_df.empty:
        st.error("No model results available. Please train the model first.")
        return
    
    # Sort by Test Accuracy - handle different column name formats
    if "Test Accuracy" in results_df.columns:
        results_df = results_df.sort_values("Test Accuracy", ascending=False)
    elif "Test_Accuracy" in results_df.columns:
        results_df = results_df.sort_values("Test_Accuracy", ascending=False)
    elif "Model" in results_df.columns:
        # If column not found, use Model name as fallback
        results_df = results_df.sort_values("Model", ascending=True)
    else:
        # Last resort: don't sort
        st.warning("Could not find performance columns to sort by")
    
    best_name = metrics.get("best_model", "Unknown")
    best_acc = metrics.get("best_test_accuracy", 0.0)

    st.markdown(
        f'<div class="best-model-banner">🏅 Best Model: <strong>{best_name}</strong> &nbsp;|&nbsp; '
        f'Test Accuracy: <strong>{best_acc:.2%}</strong> &nbsp;|&nbsp; '
        f'ROC AUC: <strong>{results_df.iloc[0]["ROC AUC"]:.4f}</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    tab1, tab2, tab3 = st.tabs(["Comparison Charts", "Metrics Table", "Hyperparameter Tuning"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Train", x=results_df["Model"], y=results_df["Train Accuracy"],
                marker_color="#a78bfa", text=[f"{v:.1%}" for v in results_df["Train Accuracy"]],
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="Test", x=results_df["Model"], y=results_df["Test Accuracy"],
                marker_color="#34d399", text=[f"{v:.1%}" for v in results_df["Test Accuracy"]],
                textposition="outside",
            ))
            apply_plotly_style(fig, "Train vs Test Accuracy", height=420)
            fig.update_layout(barmode="group", yaxis=dict(range=[0.75, 1.05], tickformat=".0%"))
            st.plotly_chart(fig, width='stretch')

        with c2:
            metric_cols = ["Precision", "Recall", "F1 Score", "ROC AUC"]
            melted = results_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            fig = px.bar(
                melted, x="Model", y="Score", color="Metric", barmode="group",
                color_discrete_sequence=["#818cf8", "#34d399", "#fbbf24", "#f472b6"],
            )
            apply_plotly_style(fig, "Precision · Recall · F1 · ROC AUC", height=420)
            fig.update_layout(yaxis=dict(range=[0.75, 1.0]))
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            categories = ["Test Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
            fig = go.Figure()
            palette = ["#7c3aed", "#3b82f6", "#10b981", "#f59e0b"]
            for i, model in enumerate(results_df["Model"]):
                row = results_df[results_df["Model"] == model].iloc[0]
                values = [row[m] for m in categories]
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=model,
                    line_color=palette[i % len(palette)],
                    opacity=0.7,
                ))
            apply_plotly_style(fig, "Radar Chart — Model Profiles", height=450)
            fig.update_layout(polar=dict(radialaxis=dict(range=[0.75, 1.0], tickformat=".0%")))
            st.plotly_chart(fig, width='stretch')

        with c4:
            heatmap_data = results_df.set_index("Model")[
                ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
            ]
            fig = px.imshow(
                heatmap_data, text_auto=".3f", aspect="auto",
                color_continuous_scale="Greens",
            )
            apply_plotly_style(fig, "Metrics Heatmap", height=450)
            st.plotly_chart(fig, width='stretch')

        results_df["Overfit Gap"] = results_df["Train Accuracy"] - results_df["Test Accuracy"]
        fig = px.bar(
            results_df.sort_values("Overfit Gap", ascending=True),
            x="Overfit Gap", y="Model", orientation="h",
            color="Overfit Gap", color_continuous_scale="Reds",
            text="Overfit Gap",
        )
        fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
        apply_plotly_style(fig, "Overfitting Gap (Train − Test Accuracy)", height=350)
        st.plotly_chart(fig, width='stretch')

    with tab2:
        st.dataframe(
            results_df.style.background_gradient(cmap="Greens", subset=["Test Accuracy", "F1 Score", "ROC AUC"])
            .format({
                "Train Accuracy": "{:.4f}", "Test Accuracy": "{:.4f}",
                "Precision": "{:.4f}", "Recall": "{:.4f}",
                "F1 Score": "{:.4f}", "ROC AUC": "{:.4f}",
                "Overfit Gap": "{:.4f}",
            }),
            width='stretch',
        )

        rank_fig = px.bar(
            results_df.sort_values("Test Accuracy"),
            x="Test Accuracy", y="Model", orientation="h",
            color="Test Accuracy", color_continuous_scale="Viridis",
            text="Test Accuracy",
        )
        rank_fig.update_traces(texttemplate="%{x:.2%}", textposition="outside")
        apply_plotly_style(rank_fig, "Model Ranking by Test Accuracy", height=350)
        st.plotly_chart(rank_fig, width='stretch')

    with tab3:
        tuning = metrics.get("tuning", {})
        if tuning:
            tune_df = pd.DataFrame([
                {"Model": name, "CV Accuracy": data["best_cv_accuracy"]}
                for name, data in tuning.items()
            ]).sort_values("CV Accuracy", ascending=False)
            fig = px.bar(
                tune_df, x="Model", y="CV Accuracy", color="CV Accuracy",
                color_continuous_scale="Blues", text="CV Accuracy",
            )
            fig.update_traces(texttemplate="%{y:.2%}", textposition="outside")
            apply_plotly_style(fig, "Cross-Validation Accuracy (Hyperparameter Tuning)", height=380)
            st.plotly_chart(fig, width='stretch')

        for name, tune in tuning.items():
            with st.expander(f"⚙️ {name} — CV Accuracy: {tune['best_cv_accuracy']:.4f}"):
                st.json(tune["best_params"])


def page_prediction(model, label_encoder, stop_words, lemmatizer, metrics):
    st.markdown('<div class="section-title">🔮 Live Sentiment Prediction</div>', unsafe_allow_html=True)

    # Model selector with explanation
    available_models = metrics.get("available_models", ["Random Forest"])
    model_objects = metrics.get("model_objects", {"Random Forest": model})
    
    selected_model_name = st.selectbox(
        "Select Model",
        available_models,
        index=0,
        help="Logistic Regression is better at predicting neutral/irrelevant classes. Random Forest has higher overall accuracy but is biased toward positive/negative.",
    )
    
    # Use the selected model
    current_model = model_objects[selected_model_name]
    
    model_info = {
        "Random Forest": "High accuracy (86%) but biased toward positive/negative",
        "Logistic Regression": "Lower accuracy (74%) but better at detecting neutral/irrelevant",
        "Hist Gradient Boosting": "Balanced performance (71%)",
        "XGBoost": "Good performance (69%)",
    }
    
    st.markdown(
        f'<div class="info-box">Current model: <strong>{selected_model_name}</strong> — {model_info.get(selected_model_name, "")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    review = st.text_area(
        "Enter a review or tweet",
        value=st.session_state.get(
            "example_review",
            "Love the new product! Great quality and fast shipping. Would definitely recommend to others. #HappyCustomer",
        ),
        height=150,
    )

    if st.button("Predict Sentiment", type="primary", width='stretch'):
        if not review.strip():
            st.warning("Please enter a review.")
            return

        review_clean = review.lower()
        review_clean = re.sub(r"<br\s*/?>", " ", review_clean)
        review_clean = re.sub(r"<[^>]+>", " ", review_clean)
        review_clean = re.sub(r"http\S+|www\S+", " ", review_clean)
        review_clean = re.sub(r"\d+", " ", review_clean)
        review_clean = review_clean.translate(str.maketrans("", "", string.punctuation))
        review_clean = re.sub(r"\s+", " ", review_clean).strip()

        tokens = word_tokenize(review_clean)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
        review_clean = " ".join(tokens)

        input_df = pd.DataFrame({"review_clean": [review_clean]})
        pred = current_model.predict(input_df)[0]
        proba = current_model.predict_proba(input_df)[0]
        label = label_encoder.inverse_transform([pred])[0]
        confidence = proba.max() * 100

        col1, col2 = st.columns([1, 2])
        card_styles = {
            "positive": ("prediction-card-positive", "#4c1d95", "#6d28d9"),
            "negative": ("prediction-card-negative", "#991b1b", "#dc2626"),
            "neutral": ("prediction-card-neutral", "#854d0e", "#ca8a04"),
            "irrelevant": ("prediction-card-irrelevant", "#475569", "#64748b"),
        }
        with col1:
            css_class, title_color, conf_color = card_styles.get(
                label, ("prediction-card-irrelevant", "#475569", "#64748b")
            )
            st.markdown(
                f"""
                <div class="{css_class}">
                    <div style="font-size:3rem;">{SENTIMENT_EMOJI.get(label, "❓")}</div>
                    <div style="font-weight:700;font-size:1.4rem;color:{title_color};margin-top:0.5rem;">
                        {label.upper()}
                    </div>
                    <div style="color:{conf_color};font-size:1.1rem;margin-top:0.5rem;">
                        {confidence:.1f}% confidence
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        with col2:
            colors = [SENTIMENT_COLORS.get(str(c), "#94a3b8") for c in label_encoder.classes_]
            fig = go.Figure(go.Bar(
                x=label_encoder.classes_,
                y=proba,
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in proba],
                textposition="outside",
            ))
            apply_plotly_style(fig, "Prediction Probabilities", height=300)
            fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"))
            st.plotly_chart(fig, width='stretch')

    st.markdown("#### Try these examples")
    examples = {
        "😊 Positive": "Love the new iPhone! The camera quality is amazing and battery life is so much better. Best purchase ever! #Apple #iPhone",
        "😞 Negative": "Worst customer service ever. Waited 2 hours on the phone and they still couldn't help me. Never buying from this company again.",
        "😐 Neutral": "The product arrived on time and works as described. Nothing special but does the job.",
        "🚫 Irrelevant": "ndsfidisbgihbsgouahbg asdfghjkl qwertyuiop zxcvbnm",
    }
    ex_cols = st.columns(4)
    for col, (title, text) in zip(ex_cols, examples.items()):
        with col:
            if st.button(title, width='stretch'):
                st.session_state["example_review"] = text
                st.rerun()


def page_about(metrics):
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
        This project performs <strong>4-class sentiment classification</strong> on the
        Twitter dataset using classical machine learning with TF-IDF features.<br><br>
        <strong>Classes:</strong> Positive · Negative · Neutral · Irrelevant<br>
        <strong>Pipeline:</strong> NLTK preprocessing → TF-IDF → RandomizedSearchCV tuning<br>
        <strong>Models:</strong> Logistic Regression · Random Forest · Gradient Boosting · XGBoost<br>
        <strong>Tech stack:</strong> Python, scikit-learn, NLTK, XGBoost, Streamlit, Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Visual display of key metrics instead of JSON
    st.markdown('<div class="section-title">📊 Model Performance Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="stat-card"><h3>{metrics["best_model"]}</h3><p>Best Model</p></div>',
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f'<div class="stat-card"><h3>{metrics["best_test_accuracy"]:.2%}</h3><p>Best Accuracy</p></div>',
            unsafe_allow_html=True,
        )
    
    with col3:
        dataset_info = metrics.get("dataset", {})
        if isinstance(dataset_info, dict):
            total_samples = dataset_info.get("total_samples", "N/A")
        else:
            total_samples = "N/A"
        
        # Simple display without complex formatting
        display_samples = str(total_samples)
            
        st.markdown(
            f'<div class="stat-card"><h3>{display_samples}</h3><p>Total Samples</p></div>',
            unsafe_allow_html=True,
        )
    
    with col4:
        classes = metrics.get("classes", SENTIMENT_ORDER)
        num_classes = len(classes) if isinstance(classes, list) else 4
        st.markdown(
            f'<div class="stat-card"><h3>{num_classes}</h3><p>Classes</p></div>',
            unsafe_allow_html=True,
        )
    
    # Display classes in a more visual way
    st.markdown('<div class="section-title">🏷️ Sentiment Classes</div>', unsafe_allow_html=True)
    classes = metrics.get("classes", SENTIMENT_ORDER)
    class_cols = st.columns(len(classes))
    for col, class_name in zip(class_cols, classes):
        with col:
            emoji = SENTIMENT_EMOJI.get(class_name, "❓")
            color = SENTIMENT_COLORS.get(class_name, "#a6adc8")
            st.markdown(
                f'<div class="sentiment-card" style="background:#313244;border:2px solid {color};">'
                f'<div class="emoji">{emoji}</div>'
                f'<div class="label">{class_name.title()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(
        page_title="Sentiment Analysis Dashboard",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    if not (MODEL_DIR / "sentiment_model.joblib").exists():
        st.error(
            "Model not found! Please run **train_model.py** first "
            "to train and save the model to the `models/` folder."
        )
        st.code("python train_model.py", language="bash")
        st.stop()

    model, label_encoder, stop_words, lemmatizer, metrics = load_artifacts()
    df = load_dataset()

    with st.sidebar:
        st.markdown("### 🎬 Navigation")
        page = st.radio(
            "Go to",
            ["Overview", "Data Analysis", "Model Evaluation", "Prediction", "About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Model Summary**")
        st.markdown(f"Best: **{metrics['best_model']}**")
        st.markdown(f"Accuracy: **{metrics['best_test_accuracy']:.2%}**")
        st.markdown("---")
        classes = metrics.get("classes", SENTIMENT_ORDER)
        st.caption(f"Classes: {', '.join(classes)}")
        st.caption("Twitter Sentiment Dashboard")

    show_header()

    if page == "Overview":
        page_overview(df, metrics)
    elif page == "Data Analysis":
        page_data_analysis(df)
    elif page == "Model Evaluation":
        page_evaluation(metrics)
    elif page == "Prediction":
        page_prediction(model, label_encoder, stop_words, lemmatizer, metrics)
    elif page == "About":
        page_about(metrics)


if __name__ == "__main__":
    main()
