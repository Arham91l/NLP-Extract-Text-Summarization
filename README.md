# ⚡ SummarAI — Text Summarization App

TF-IDF + Cosine Similarity extractive summarizer trained on CNN/DailyMail (284K articles).

## 🚀 Run Locally

```bash
# 1. Clone / download this folder
cd text_summarizer_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at: http://localhost:8501

## 📁 File Structure

```
text_summarizer_app/
├── app.py              ← Main Streamlit app
├── summarizer.py       ← TF-IDF + Cosine summarizer logic
├── requirements.txt    ← Python dependencies
├── .streamlit/
│   └── config.toml     ← Dark theme config
└── README.md
```

## 🔬 Method

- **Vectorizer**: TF-IDF (unigrams + bigrams, max 10K features)
- **Similarity**: Cosine similarity between each sentence and full article
- **Position bias**: First 15% of article boosted ×1.3, last 15% ×1.1
- **Evaluation**: ROUGE-1: 0.3254 | ROUGE-2: 0.1231 | ROUGE-L: 0.2106
- **Dataset**: CNN/DailyMail 3.0.0 (284K train, 13K val, 11K test)

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo → select `app.py`
4. Deploy!

## App link
https://nlp-extract-text-summarization-ihqy9qsf5dicjiiwmclmet.streamlit.app/
