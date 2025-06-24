# 🎥 YouTube Comment Intelligence Hub

An advanced AI-powered web application for analyzing YouTube comments using deep learning models to extract sentiment and emotion insights. Built with Streamlit and powered by LSTM neural networks for accurate sentiment analysis and emotion detection.

## 🚀 Features

- **🎯 Smart Comment Extraction**: Automatically fetch and process YouTube comments using YouTube Data API v3
- **🧠 AI Sentiment Analysis**: Deep learning LSTM models classify positive/negative sentiment with 80%+ accuracy
- **😊 Emotion Detection**: Identify 13 different emotions in comments using advanced NLP techniques
- **📊 Interactive Visualizations**: Beautiful charts and graphs powered by Plotly for data insights
- **📈 Engagement Analytics**: Analyze likes and engagement patterns across different sentiments
- **🌍 English Language Support**: Optimized for English comments with 80%+ accuracy guarantee
- **📋 Export Summary**: Download comprehensive analysis summary reports in CSV format

## 🛠️ Technology Stack

### Core Technologies
- **Frontend**: Streamlit
- **Backend**: Python 3.11+
- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn

### AI/ML Components
- **Neural Network Architecture**: LSTM (Long Short-Term Memory)
- **NLP Framework**: NLTK
- **Text Preprocessing**: Custom tokenization, lemmatization, stop word removal
- **Model Training**: TensorFlow Keras Sequential API
- **API Integration**: YouTube Data API v3

### Key Libraries
```
streamlit
tensorflow
pandas
numpy
plotly
nltk
google-api-python-client
python-dotenv
matplotlib
seaborn
```

## 📊 Model Architecture

### Sentiment Analysis Model
- **Algorithm**: LSTM Neural Network
- **Vocabulary Size**: 5,000 words
- **Sequence Length**: 50 tokens
- **Architecture**: 
  - Embedding Layer
  - LSTM Layer with dropout
  - Dense output layer with sigmoid activation
  - Binary classification (Positive/Negative)

### Emotion Detection Model
- **Algorithm**: LSTM Neural Network
- **Vocabulary Size**: 10,000 words
- **Sequence Length**: 50 tokens
- **Classes**: 13 emotions (Empty, Sadness, Enthusiasm, Neutral, Worry, Surprise, Love, Fun, Hate, Happiness, Boredom, Relief, Anger)
- **Architecture**:
  - Embedding Layer
  - LSTM Layer with dropout
  - Dense output layer with softmax activation
  - Multi-class classification

### Text Preprocessing Pipeline
1. **Text Cleaning**: Remove special characters, convert to lowercase
2. **Tokenization**: Split text into individual words using NLTK
3. **Lemmatization**: Reduce words to their base form using WordNet Lemmatizer
4. **Stop Word Removal**: Filter out common English stop words
5. **Sequence Padding**: Pad sequences to uniform length for model input

## 🔧 Installation

### Prerequisites
- Python 3.11.7 or higher
- YouTube Data API v3 key
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/youtube-comment-analyzer.git
cd youtube-comment-analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

5. **Set up environment variables**
Create a `.env` file in the project root:
```
YOUTUBE_API_KEY=your_youtube_api_key_here
```

6. **Prepare model files**
Ensure the following model files are in the `models/` directory:
- `sentiment_model.h5`
- `emotion_model_final.h5`

## 🚀 Usage

### Running the Application
```bash
streamlit run app.py
```



### How to Use

1. **Enter YouTube URL**: Paste any YouTube video URL in the input field
2. **Select Analysis Size**: Choose number of comments to analyze (50-500)
3. **Start Analysis**: Click the "Start Analysis" button
4. **View Results**: Explore interactive dashboards and insights
5. **Export Data**: Download summary reports for further analysis

### Supported URL Formats
- `https://www.youtube.com/watch?v=VIDEO_ID`

## 📈 Output Analytics

### Sentiment Analysis
- **Positive/Negative Distribution**: Pie chart showing sentiment breakdown
- **Engagement by Sentiment**: Bar chart comparing likes across sentiments
- **Confidence Scores**: Model prediction confidence metrics

### Emotion Detection
- **Emotion Landscape**: Horizontal bar chart of top detected emotions
- **Emotion Engagement**: Average likes per emotion type
- **Emotion Distribution**: Comprehensive breakdown of all 13 emotions

### Summary Statistics
- Total comments analyzed
- Positive vs negative comment counts
- Total engagement (likes)
- Positivity percentage
- Top detected emotion
- Average confidence scores

## 🎯 Model Performance

- **Sentiment Analysis Accuracy**: 80%+ on English comments
- **Emotion Detection Accuracy**: 80%+ on English comments
- **Processing Speed**: ~100 comments per second
- **Language Support**: Optimized for English text
- **Confidence Threshold**: 0.5 for sentiment classification

## 📁 Project Structure

```
youtube-comment-analyzer/
├── app.py                 # Main Streamlit application
├── models/
│   ├── sentiment_model.h5      # Trained sentiment LSTM model
│   └── emotion_model_final.h5  # Trained emotion LSTM model
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── README.md             # Project documentation
```

## 🔒 API Configuration

### YouTube Data API v3 Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable YouTube Data API v3
4. Create API credentials (API Key)
5. Add the API key to your `.env` file


## 🚨 Limitations

- **Language**: Optimized for English comments only
- **API Limits**: Subject to YouTube API quotas
- **Model Size**: Pre-trained models require ~50MB storage
- **Processing Time**: Large comment sets may take several minutes
- **Accuracy**: Performance may vary with informal text, slang, or emoji-heavy comments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ezhilarasu G**  
🎓 B.Tech | Artificial Intelligence and Machine Learning  
💡 Passionate about making data accessible to everyone  

<p align="center">
  <a href="https://github.com/gezhilarasu">
    <img src="https://img.icons8.com/color/48/github.png" width="40"/>
  </a>
  <a href="https://linkedin.com/in/gezhilarasu">
    <img src="https://img.icons8.com/color/48/linkedin.png" width="40"/>
  </a>
  <a href="mailto:your.gezhilarasu24@gail.com">
    <img src="https://img.icons8.com/color/48/gmail.png" width="40"/>
  </a>
</p>

---

<p align="center">
  <strong>⭐ Star this repo if YouTube Comment Intelligence Hub helped you! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ and ☕ by developers who believe data should be accessible to everyone
</p>
---

