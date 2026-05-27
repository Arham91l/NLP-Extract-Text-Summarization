from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_tfidf_summary(article, top_n=3):
    sentences = [s.strip() for s in article.split('.')
                 if len(s.strip()) > 20]

    if len(sentences) == 0:
        return ""
    if len(sentences) <= top_n:
        return '. '.join(sentences) + '.'

    docs = sentences + [article]
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    try:
        matrix = tfidf.fit_transform(docs)
    except:
        return '. '.join(sentences[:top_n]) + '.'

    article_vec   = matrix[-1]
    sentence_vecs = matrix[:-1]
    scores        = cosine_similarity(sentence_vecs, article_vec).flatten()

    for i in range(len(scores)):
        position = i / len(sentences)
        if position < 0.15:
            scores[i] *= 1.3
        elif position > 0.85:
            scores[i] *= 1.1

    top_indices = np.argsort(scores)[::-1][:top_n]
    top_indices = sorted(top_indices)

    summary = '. '.join([sentences[i] for i in top_indices]) + '.'
    return summary


def get_sentence_scores(article):
    """Return all sentences with their cosine scores for visualization."""
    sentences = [s.strip() for s in article.split('.')
                 if len(s.strip()) > 20]

    if len(sentences) < 2:
        return [], []

    docs = sentences + [article]
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)

    try:
        matrix = tfidf.fit_transform(docs)
    except:
        return sentences, [0.5] * len(sentences)

    scores = cosine_similarity(matrix[:-1], matrix[-1]).flatten()

    for i in range(len(scores)):
        position = i / len(sentences)
        if position < 0.15:
            scores[i] *= 1.3
        elif position > 0.85:
            scores[i] *= 1.1

    return sentences, scores.tolist()


def compression_stats(article, summary):
    """Return basic stats about the compression."""
    art_words = len(article.split())
    sum_words = len(summary.split())
    art_sents = len([s for s in article.split('.') if len(s.strip()) > 20])
    sum_sents = len([s for s in summary.split('.') if len(s.strip()) > 5])
    ratio = round(sum_words / art_words * 100, 1) if art_words > 0 else 0
    return {
        'article_words': art_words,
        'summary_words': sum_words,
        'article_sentences': art_sents,
        'summary_sentences': sum_sents,
        'compression_pct': ratio
    }
