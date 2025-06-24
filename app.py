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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="YouTube Comment Analyzer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF0000, #FF4500, #FF6347);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card h3 {
        margin-top: 0;
        color: #fff;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stats-card h2 {
        margin: 0;
        font-size: 2.5rem;
    }
    
    .stats-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .workflow-step {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: white;
        border-left: 4px solid #fff;
    }
    
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_data()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Load models (with error handling)
@st.cache_resource
def load_models():
    try:
        sentiment_model = load_model("models/sentiment_model.h5", compile=False)
        emotion_model = load_model("models/emotion_model_final.h5", compile=False)
        return sentiment_model, emotion_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

sentiment_model, emotion_model = load_models()

# NLTK setup
try:
    lemmatizer = WordNetLemmatizer()
except:
    st.error("NLTK WordNet Lemmatizer not available")
    lemmatizer = None

custom_stop_words = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
}

# Utility functions
def extract_text(text):
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    return review.lower().strip()

def lemmatize_text(text):
    if lemmatizer is None:
        return text
    try:
        words = nltk.word_tokenize(text)
        lemmatized = [lemmatizer.lemmatize(w) for w in words if w not in custom_stop_words]
        return " ".join(lemmatized)
    except:
        return text

def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"embed/([^?]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_comments(video_id, max_comments=500):
    if not API_KEY:
        st.error("YouTube API key not found. Please set YOUTUBE_API_KEY in your environment.")
        return pd.DataFrame()
    
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    next_page_token = None

    try:
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
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return pd.DataFrame()

# Main UI
st.markdown("""
<div class="main-header">
    <h1>🎥 YouTube Comment Intelligence Hub</h1>
    <p>Advanced AI-Powered Sentiment & Emotion Analysis for YouTube Comments</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with features
with st.sidebar:
    st.markdown("## 🚀 Key Features")
    
    features = [
        ("🎯", "Smart Comment Extraction", "Automatically fetch and process YouTube comments"),
        ("🧠", "AI Sentiment Analysis", "Deep learning models classify positive/negative sentiment"),
        ("😊", "Emotion Detection", "Identify 13 different emotions in comments"),
        ("📊", "Interactive Visualizations", "Beautiful charts and graphs for insights"),
        ("📈", "Engagement Analytics", "Analyze likes and engagement patterns"),
        ("🌍", "English Language Support", "Optimized for English comments with 80%+ accuracy"),
        ("📋", "Export Summary", "Download analysis summary reports")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 1rem 0; padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <div style="font-size: 1.5rem; margin-right: 1rem;">{icon}</div>
            <div>
                <strong>{title}</strong><br>
                <small style="opacity: 0.8;">{desc}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Workflow explanation
st.markdown("## 🔄 How It Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="workflow-step">
        <h4>1️⃣ Extract</h4>
        <p>Fetch comments from YouTube video using API</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="workflow-step">
        <h4>2️⃣ Process</h4>
        <p>Clean and preprocess text using NLP techniques</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="workflow-step">
        <h4>3️⃣ Analyze</h4>
        <p>Apply AI models for sentiment & emotion detection</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="workflow-step">
        <h4>4️⃣ Visualize</h4>
        <p>Present insights through interactive charts</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Input section
st.markdown("## 🔗 Video Analysis")
col1, col2 = st.columns([3, 1])

with col1:
    video_url = st.text_input(
        "Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube video URL to analyze its comments"
    )

with col2:
    max_comments = st.selectbox(
        "Comments to analyze:",
        options=[50, 100, 200, 300, 500],
        index=1,
        help="More comments = better analysis but slower processing"
    )

if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
    if not video_url:
        st.warning("Please enter a YouTube video URL first!")
    elif sentiment_model is None or emotion_model is None:
        st.error("AI models are not loaded. Please check the model files.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check the URL format.")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Fetch comments
            status_text.text("🔄 Fetching comments from YouTube...")
            progress_bar.progress(20)
            
            df = fetch_comments(video_id, max_comments=max_comments)
            
            if df.empty:
                st.error("❌ No comments found or error occurred while fetching comments.")
            else:
                # Step 2: Preprocess
                status_text.text("🧹 Cleaning and preprocessing text...")
                progress_bar.progress(40)
                
                df['cleaned'] = df['Comment'].apply(extract_text).apply(lemmatize_text)
                
                # Step 3: Tokenize
                status_text.text("🔤 Tokenizing text for AI models...")
                progress_bar.progress(60)
                
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

                # Step 4: Predict
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(80)
                
                sentiment_preds = sentiment_model.predict(sent_padded, verbose=0)
                df['sentiment'] = (sentiment_preds > 0.5).astype(int)
                df['sentiment_label'] = df['sentiment'].map({1: 'Positive', 0: 'Negative'})
                df['sentiment_confidence'] = np.where(df['sentiment'] == 1, 
                                                    sentiment_preds.flatten(), 
                                                    1 - sentiment_preds.flatten())

                emotion_preds = emotion_model.predict(emo_padded, verbose=0)
                emo_ids = np.argmax(emotion_preds, axis=1)

                emotion_map = {
                    0: 'Empty', 1: 'Sadness', 2: 'Enthusiasm', 3: 'Neutral', 4: 'Worry',
                    5: 'Surprise', 6: 'Love', 7: 'Fun', 8: 'Hate', 9: 'Happiness',
                    10: 'Boredom', 11: 'Relief', 12: 'Anger'
                }
                df['predicted_emotion'] = pd.Series(emo_ids).map(emotion_map)
                df['emotion_confidence'] = np.max(emotion_preds, axis=1)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Success message
                st.markdown("""
                <div class="success-message">
                    🎉 Analysis Complete! Here are your insights:
                </div>
                """, unsafe_allow_html=True)

                # Summary statistics
                pos_count = (df['sentiment'] == 1).sum()
                neg_count = (df['sentiment'] == 0).sum()
                pos_likes = df[df['sentiment'] == 1]['Likes'].sum()
                neg_likes = df[df['sentiment'] == 0]['Likes'].sum()
                total_likes = df['Likes'].sum()
                avg_sentiment_conf = df['sentiment_confidence'].mean()
                avg_emotion_conf = df['emotion_confidence'].mean()

                # Display summary cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h2>{pos_count}</h2>
                        <p>😊 Positive Comments</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h2>{neg_count}</h2>
                        <p>😞 Negative Comments</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h2>{total_likes}</h2>
                        <p>👍 Total Likes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    sentiment_ratio = pos_count / (pos_count + neg_count) * 100
                    st.markdown(f"""
                    <div class="stats-card">
                        <h2>{sentiment_ratio:.1f}%</h2>
                        <p>📈 Positivity Rate</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Visualizations
                st.markdown("## 📊 Interactive Analytics Dashboard")
                
                # Sentiment Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 💭 Sentiment Distribution")
                    sentiment_counts = df['sentiment_label'].value_counts()
                    
                    fig_sentiment = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        color_discrete_map={'Positive': '#00D4AA', 'Negative': '#FF6B6B'},
                        title="Comment Sentiment Breakdown"
                    )
                    fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                    fig_sentiment.update_layout(
                        font=dict(size=14),
                        showlegend=True,
                        height=400,
                        title_font_size=16
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎭 Emotion Landscape")
                    emotion_counts = df['predicted_emotion'].value_counts().head(8)
                    
                    fig_emotion = px.bar(
                        x=emotion_counts.values,
                        y=emotion_counts.index,
                        orientation='h',
                        color=emotion_counts.values,
                        color_continuous_scale='viridis',
                        title="Top Emotions Detected"
                    )
                    fig_emotion.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        height=400,
                        title_font_size=16,
                        showlegend=False
                    )
                    st.plotly_chart(fig_emotion, use_container_width=True)

                # Engagement Analysis
                st.markdown("### 💡 Engagement Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Likes by sentiment
                    likes_by_sentiment = df.groupby('sentiment_label')['Likes'].sum().reset_index()
                    
                    fig_likes = px.bar(
                        likes_by_sentiment,
                        x='sentiment_label',
                        y='Likes',
                        color='sentiment_label',
                        color_discrete_map={'Positive': '#00D4AA', 'Negative': '#FF6B6B'},
                        title="Total Likes by Sentiment"
                    )
                    fig_likes.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_likes, use_container_width=True)
                
                with col2:
                    # Top emotions by engagement
                    emotion_engagement = df.groupby('predicted_emotion')['Likes'].mean().sort_values(ascending=False).head(6)
                    
                    fig_emotion_likes = px.bar(
                        x=emotion_engagement.index,
                        y=emotion_engagement.values,
                        color=emotion_engagement.values,
                        color_continuous_scale='plasma',
                        title="Average Likes by Emotion"
                    )
                    fig_emotion_likes.update_layout(
                        xaxis_tickangle=-45,
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_emotion_likes, use_container_width=True)

                # Export section
                st.markdown("## 📥 Export Results")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Summary report
                    summary_data = {
                        'Total Comments': len(df),
                        'Positive Comments': pos_count,
                        'Negative Comments': neg_count,
                        'Positivity Rate': f"{sentiment_ratio:.1f}%",
                        'Total Likes': total_likes,
                        'Avg Sentiment Confidence': f"{avg_sentiment_conf:.3f}",
                        'Avg Emotion Confidence': f"{avg_emotion_conf:.3f}",
                        'Top Emotion': df['predicted_emotion'].mode().iloc[0]
                    }
                    
                    summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
                    summary_csv = summary_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📋 Download Summary Report",
                        data=summary_csv,
                        file_name=f"summary_report_{video_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Insights
                    st.info(f"""
                    **Quick Insights:**
                    - {sentiment_ratio:.1f}% positive sentiment
                    - Most common emotion: {df['predicted_emotion'].mode().iloc[0]}
                    - Highest engagement: {df.loc[df['Likes'].idxmax(), 'sentiment_label']} comments
                    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🚀 Built with Streamlit | 🤖 Powered by TensorFlow | 🎯 Enhanced with AI</p>
    <p>Made with ❤️ for YouTube Content Analysis</p>
</div>
""", unsafe_allow_html=True)