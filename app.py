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
import numpy as np

# Load RoBERTa tokenizer and model
roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

nltk.download('vader_lexicon')
sns.set_style('darkgrid')

# Load data
df = pd.read_csv("amazon.csv").head(500)

# Load models
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, roberta_model = load_models()
vader = SentimentIntensityAnalyzer()

# Analysis functions
def analyze_vader(text):
    return vader.polarity_scores(text)

def analyze_roberta(text):
    encoded = roberta_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        output = roberta_model(**encoded)
    scores = output.logits[0].numpy()
    scores = softmax(scores)
    labels = ['roberta_neg', 'roberta_neu', 'roberta_pos']
    return dict(zip(labels, scores))

def get_roberta_label(scores):
    return max(scores, key=scores.get).replace("roberta_", "")

def get_vader_label(score):
    comp = score['compound']
    return 'pos' if comp >= 0.05 else 'neg' if comp <= -0.05 else 'neu'

def sentiment_label(vader_sent, roberta_sent):
    return 'Agree' if vader_sent == roberta_sent else 'Disagree'

# UI
st.title("Amazon_review Sentiment Analysis")

st.subheader("Dataset Preview")
st.dataframe(df.head(6))

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="amazon_reviews.csv">Download Dataset</a>'
    return href

st.markdown(file_download(df), unsafe_allow_html=True)

st.subheader("Select Sentiment Analysis Model")
col1, col2, col3 = st.columns(3)

selected_tool = st.session_state.get("tool", None)

if col1.button("VADER"):
    st.session_state["tool"] = "VADER"
    selected_tool = "VADER"
if col2.button("RoBERTa"):
    st.session_state["tool"] = "ROBERTA"
    selected_tool = "ROBERTA"

if selected_tool:
    st.markdown(f"Selected Model: {selected_tool}")
    results = []

    for _, row in df.iterrows():
        text = str(row['reviewText'])
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

    if selected_tool == "VADER":
        st.subheader("VADER Sentiment Analysis")
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

    elif selected_tool == "ROBERTA":
        st.subheader("RoBERTa Sentiment Analysis")
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

if st.session_state.get("results_df") is not None:
    if col3.button("Compare Models"):
        st.subheader("Model Agreement Comparison")
        df_comp = st.session_state["results_df"]
        fig, ax = plt.subplots()
        sns.countplot(data=df_comp, x='overall', hue='agreement', ax=ax)
        ax.set_title("Sentiment Agreement by Rating")
        st.pyplot(fig)

        st.subheader("Sample Agreement Results")
        st.dataframe(df_comp[['overall', 'vader_sent', 'roberta_sent', 'agreement']].head(10))

# Updated GitHub Notebook Link
st.markdown(
    "[View the Jupyter Notebook on GitHub](https://github.com/tanujgupta18/Sentiment-Analysis-of-Amazon-Reviews/blob/main/Amazon_review.ipynb)"
)

# Description
if st.button("Project Description"):
    st.markdown("""
### Project Overview: Amazon Sentiment Analysis

**Objective**  
To analyze sentiment in Amazon product reviews using two techniques:
- VADER (lexicon-based)
- RoBERTa (transformer-based)

**Dataset**  
- Source: amazon.csv  
- Size: 500 reviews  
- Fields: reviewText, overall

**Steps**  
1. Load and preprocess review data  
2. Analyze sentiment using VADER and RoBERTa  
3. Visualize sentiment distribution by star rating  
4. Compare agreement between models  
5. Provide results in an interactive interface using Streamlit

**Technologies**  
- NLTK for VADER  
- Hugging Face Transformers for RoBERTa  
- Matplotlib and Seaborn for visualization  
- Streamlit for web interface
""")
