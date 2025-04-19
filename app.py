import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import nltk
import base64
import torch

# --- Setup ---
nltk.download('vader_lexicon')
sns.set_style('darkgrid')

# Load and trim dataset
df = pd.read_csv("amazon.csv").head(500)

# --- Load Models ---
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, roberta_model = load_models()
vader = SentimentIntensityAnalyzer()

# --- Analysis Functions ---
def analyze_vader(text):
    return vader.polarity_scores(text)

def analyze_roberta(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = roberta_model(**encoded)
    scores = softmax(output.logits[0].numpy())
    labels = ['roberta_neg', 'roberta_neu', 'roberta_pos']
    return dict(zip(labels, scores))

def get_vader_label(score):
    comp = score['compound']
    return 'pos' if comp >= 0.05 else 'neg' if comp <= -0.05 else 'neu'

def get_roberta_label(scores):
    return max(scores, key=scores.get).replace("roberta_", "")

def sentiment_label(vader_sent, roberta_sent):
    return 'Agree' if vader_sent == roberta_sent else 'Disagree'

# --- Streamlit UI ---
st.title("Amazon Sentiment Analysis")
st.subheader("📄 Preview Dataset")
st.dataframe(df.head(6))

# Download button
def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="amazon_reviews.csv">📥 Download the dataset</a>'
    return href

st.markdown(file_download(df), unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🔧 Settings")
tool_option = st.sidebar.radio("Choose Sentiment Tool", ["VADER", "ROBERTA"])
run_analysis = st.sidebar.button("🚀 Run Analysis")

# --- Run Analysis ---
if run_analysis:
    st.markdown(f"### Running {tool_option} Analysis on 500 Reviews...")
    results = []

    for _, row in df.iterrows():
        text = str(row['reviewText']) if pd.notna(row['reviewText']) else ""
        overall = row['overall']
        vader_score = analyze_vader(text)
        roberta_score = analyze_roberta(text)

        vader_sent = get_vader_label(vader_score)
        roberta_sent = get_roberta_label(roberta_score)

        results.append({
            'reviewText': text,
            'overall': overall,
            'vader_pos': vader_score['pos'],
            'vader_neu': vader_score['neu'],
            'vader_neg': vader_score['neg'],
            'vader_compound': vader_score['compound'],
            **roberta_score,
            'vader_sent': vader_sent,
            'roberta_sent': roberta_sent,
            'agreement': sentiment_label(vader_sent, roberta_sent)
        })

    final_df = pd.DataFrame(results)

    # --- Visualizations ---
    if tool_option == "VADER":
        st.subheader("📊 VADER Sentiment Analysis")
        fig, ax = plt.subplots()
        sns.barplot(data=final_df, x='overall', y='vader_compound', ax=ax)
        ax.set_title("VADER Compound Score by Star Rating")
        st.pyplot(fig)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        sns.barplot(data=final_df, x='overall', y='vader_pos', ax=axs[0])
        axs[0].set_title('VADER Positive')
        sns.barplot(data=final_df, x='overall', y='vader_neu', ax=axs[1])
        axs[1].set_title('VADER Neutral')
        sns.barplot(data=final_df, x='overall', y='vader_neg', ax=axs[2])
        axs[2].set_title('VADER Negative')
        st.pyplot(fig)

    elif tool_option == "ROBERTA":
        st.subheader("📊 ROBERTA Sentiment Analysis")
        fig, ax = plt.subplots()
        sns.barplot(data=final_df, x='overall', y='roberta_pos', ax=ax)
        ax.set_title("RoBERTa Positive Score by Star Rating")
        st.pyplot(fig)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        sns.barplot(data=final_df, x='overall', y='roberta_pos', ax=axs[0])
        axs[0].set_title('RoBERTa Positive')
        sns.barplot(data=final_df, x='overall', y='roberta_neu', ax=axs[1])
        axs[1].set_title('RoBERTa Neutral')
        sns.barplot(data=final_df, x='overall', y='roberta_neg', ax=axs[2])
        axs[2].set_title('RoBERTa Negative')
        st.pyplot(fig)

    st.session_state["results_df"] = final_df

# --- Comparison Button ---
if st.session_state.get("results_df") is not None:
    if st.sidebar.button("⚖️ Compare VADER & RoBERTa"):
        st.subheader("🤝 Agreement Between VADER & RoBERTa")
        df_comp = st.session_state["results_df"]
        fig, ax = plt.subplots()
        sns.countplot(data=df_comp, x='overall', hue='agreement', ax=ax)
        ax.set_title("VADER vs RoBERTa Sentiment Agreement")
        st.pyplot(fig)

        st.subheader("🔍 Sample Comparison Results")
        st.dataframe(df_comp[['overall', 'vader_sent', 'roberta_sent', 'agreement']].head(10))

# --- Jupyter Notebook Button ---
notebook_path = "Amazon_review.ipynb"
if st.button("📓 Open Jupyter Notebook"):
    st.markdown(f'[👉 Click here to open the notebook]({notebook_path})', unsafe_allow_html=True)
