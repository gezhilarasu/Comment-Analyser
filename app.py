import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load env
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Load models
sentiment_model = load_model("models/sentiment_model.h5",compile=False)
emotion_model = load_model("models/emotion_model_final.h5",compile=False)

# NLTK setup
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
custom_stop_words = {...}  # Replace with your own set

# Utility functions
def extract_text(text):
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    return review.lower().strip()

def lemmatize_text(text):
    words = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(w) for w in words if w not in set(custom_stop_words)]
    return " ".join(lemmatized)

def extract_video_id(url):
    match = re.search(r"v=([^&]+)", url)
    return match.group(1) if match else None

def fetch_comments(video_id, max_comments=500):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat='plainText'
        )
        response = request.execute()

        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comment_info = {
                'Author': snippet['authorDisplayName'],
                'Comment': snippet['textDisplay'],
                'Likes': snippet['likeCount'],
                'Published At': snippet['publishedAt']
            }
            comments.append(comment_info)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return pd.DataFrame(comments)

# Streamlit UI
st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")

st.title("🎥 YouTube Comment Sentiment & Emotion Analyzer")
video_url = st.text_input("🔗 Enter a YouTube video URL:")

max_comments = st.slider("How many comments do you want to analyze?", min_value=10, max_value=500, value=100)

if st.button("🔍 Analyze"):
    with st.spinner("Fetching comments..."):
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL")
        else:
            df = fetch_comments(video_id, max_comments=max_comments)

            # Preprocess
            df['cleaned'] = df['Comment'].apply(extract_text).apply(lemmatize_text)

            # Tokenize & pad
            max_len = 50
            sent_vocab, emo_vocab = 5000, 10000

            # Tokenizer for sentiment
            sent_tokenizer = Tokenizer(num_words=sent_vocab, oov_token="<OOV>")
            sent_tokenizer.fit_on_texts(df['cleaned'])
            sent_encoded = sent_tokenizer.texts_to_sequences(df['cleaned'])

            # Tokenizer for emotion
            emo_tokenizer = Tokenizer(num_words=emo_vocab, oov_token="<OOV>")
            emo_tokenizer.fit_on_texts(df['cleaned'])
            emo_encoded = emo_tokenizer.texts_to_sequences(df['cleaned'])

            sent_padded = pad_sequences(sent_encoded, maxlen=max_len, padding='pre')
            emo_padded = pad_sequences(emo_encoded, maxlen=max_len, padding='post')

            # Predict
            sentiment_preds = sentiment_model.predict(sent_padded)
            df['sentiment'] = (sentiment_preds > 0.5).astype(int)
            df['sentiment_label'] = df['sentiment'].map({1: 'Positive', 0: 'Negative'})

            emotion_preds = emotion_model.predict(emo_padded)
            emo_ids = np.argmax(emotion_preds, axis=1)

            emotion_map = {
                0: 'empty', 1: 'sadness', 2: 'enthusiasm', 3: 'neutral', 4: 'worry',
                5: 'surprise', 6: 'love', 7: 'fun', 8: 'hate', 9: 'happiness',
                10: 'boredom', 11: 'relief', 12: 'anger'
            }
            df['predicted_emotion'] = pd.Series(emo_ids).map(emotion_map)

            # Summary
            pos_count = (df['sentiment'] == 1).sum()
            neg_count = (df['sentiment'] == 0).sum()
            pos_likes = df[df['sentiment'] == 1]['Likes'].sum()
            neg_likes = df[df['sentiment'] == 0]['Likes'].sum()

            st.success("✅ Analysis Complete!")
            st.markdown(f"**Positive Comments**: {pos_count} | 👍 Likes: {pos_likes}")
            st.markdown(f"**Negative Comments**: {neg_count} | 👎 Likes: {neg_likes}")

            # Charts
            st.subheader("📊 Sentiment Distribution")
            sent_chart = df['sentiment_label'].value_counts()
            st.bar_chart(sent_chart)

            st.subheader("🎭 Emotion Distribution")
            emo_chart = df['predicted_emotion'].value_counts()
            st.bar_chart(emo_chart)

            # Display Data
            st.subheader("📋 Sample Comments")
            st.dataframe(df[['Author', 'Comment', 'Likes', 'sentiment_label', 'predicted_emotion']].head(20))

            # Export
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv, file_name="youtube_comment_analysis.csv", mime="text/csv")