# Sentiment Analysis of product reviews using DistilBERT

from transformers import pipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of product reviews using DistilBERT"""
    def __init__(self):
        logger.info("Loading DistilBERT sentiment model.")
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # cpu(-1), for GPU its 0
        )
        logger.info("Sentiment model loaded successfully")

    def analyze_single(self, text):
        """Analyze sentiment of a single text. Returns label and confidence score"""
        result = self.classifier(text, truncation=True, max_length=512)[0]
        return {
            "label": result["label"],
            "score": round(result["score"], 4)
        }

    def analyze_reviews(self, reviews):
        """Analyze a list of review dicts from dummyjson.

        Returns summary with positive count, negative count, avg score.
        """
        if not reviews:
            return {
                "positive": 0,
                "negative": 0,
                "avg_score": 0.0,
                "summary": "No reviews"
            }
        comments = [r.get("comment", "") for r in reviews]
        results = self.classifier(comments, truncation=True, max_length=512)
        positive = sum(1 for r in results if r["label"] == "POSITIVE")
        negative = sum(1 for r in results if r["label"] == "NEGATIVE")

        # Convert to a 0-1 scale (negative scores become negative directions)
        scores = []
        for r in results:
            if r["label"] == "POSITIVE":
                scores.append(r["score"])
            else:
                scores.append(1 - r["score"])

        avg_score = round(sum(scores) / len(scores), 4)

        if positive > negative:
            summary = "Mostly positive"
        elif negative > positive:
            summary = "Mostly negative"
        else:
            summary = "Mixed"

        return {
            "positive": positive,
            "negative": negative,
            "avg_score": avg_score,
            "summary": summary
        }


"""
What this does:
Loads DistilBERT - a lightweight BERT model fine-tuned for sentiment (runs on CPU, ~260MB)
analyze_single(text) - scores a single review as POSITIVE/NEGATIVE with confidence
analyze_reviews(reviews) - takes the DummyJSON review list, classifies all of them,
and returns a summary (positive count, negative count, average score, overall sentiment)
"""
