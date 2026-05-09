"""Basic tests for ShopSmart AI core modules."""

# Config Tests
def test_config_loads():
    """Config class should load without errors."""
    from shopsmart.config import Config
    assert Config.PRODUCT_API_BASE == "https://dummyjson.com"
    assert Config.EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
    assert Config.RAG_MODEL == "llama-3.1-8b-instant"


def test_config_analytics_db_default():
    """Analytics DB should default to SQLite if env var not set."""
    from shopsmart.config import Config
    assert "sqlite" in Config.ANALYTICS_DB_URL


# Sentiment Tests
def test_sentiment_analyzer_single():
    """Sentiment analyzer should return label and score for a single text."""
    from shopsmart.sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_single("This product is amazing!")
    assert result["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0 <= result["score"] <= 1


def test_sentiment_analyzer_empty_reviews():
    """Sentiment analyzer should handle empty review list."""
    from shopsmart.sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_reviews([])
    assert result["positive"] == 0
    assert result["negative"] == 0
    assert result["summary"] == "No reviews"


def test_sentiment_analyzer_reviews():
    """Sentiment analyzer should classify multiple reviews."""
    from shopsmart.sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    reviews = [
        {"comment": "Great quality, love it!"},
        {"comment": "Terrible product, broke immediately"},
        {"comment": "Decent value for the price"},
    ]
    result = analyzer.analyze_reviews(reviews)
    assert result["positive"] + result["negative"] == 3
    assert 0 <= result["avg_score"] <= 1
    assert result["summary"] in ["Mostly positive", "Mostly negative", "Mixed"]


#Analytics Tests
def test_analytics_tracker_init():
    """Analytics tracker should initialize with SQLite."""
    from shopsmart.analytics import AnalyticsTracker
    tracker = AnalyticsTracker()
    assert tracker.get_total_queries() >= 0


def test_analytics_log_and_retrieve():
    """Should log an interaction and retrieve it."""
    from shopsmart.analytics import AnalyticsTracker
    tracker = AnalyticsTracker()
    initial_count = tracker.get_total_queries()

    tracker.log_interaction(
        session_id="test_session",
        user_query="test query",
        bot_response="test response",
        response_time_ms=100.0,
        category_detected="test"
    )

    assert tracker.get_total_queries() == initial_count + 1

    recent = tracker.get_recent_interactions(limit=1)
    assert len(recent) == 1
    assert recent[0].user_query == "test query"


# Data Converter Tests-
def test_api_fetch_products():
    """Should fetch products from DummyJSON API."""
    import requests
    resp = requests.get("https://dummyjson.com/products", params={"limit": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert "products" in data
    assert len(data["products"]) == 1


# Logger Tests
def test_logger_creation():
    """Logger should be creatable without errors."""
    from utils.logger import get_logger
    logger = get_logger("test")
    assert logger is not None
