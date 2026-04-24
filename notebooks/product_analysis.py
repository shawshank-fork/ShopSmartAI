# Product Analysis: Clustering, Embeddings Visualization, and EDA

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter
from shopsmart.data_converter import APIDataConverter
from dotenv import load_dotenv
load_dotenv()

sns.set_theme(style="dark")
plt.rcParams["figure.figsize"] = (12, 6)

def fetch_products():
    """Fetch all products from DummyJSON API."""
    print("Fetching products from API...")
    converter = APIDataConverter()
    products = converter.fetch_all_products()
    print(f"Fetched {len(products)} products")
    return products

def category_distribution(products):
    """Bar chart of product count per category."""
    categories = [p["category"] for p in products]
    cat_counts = Counter(categories)
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    names = [c[0] for c in sorted_cats]
    counts = [c[1] for c in sorted_cats]
    plt.figure(figsize=(14, 6))
    bars = plt.barh(names, counts, color=sns.color_palette("viridis", len(names)))
    plt.xlabel("Number of Products")
    plt.title("Product Distribution by Category")
    plt.tight_layout()
    plt.savefig("notebooks/category_distribution.png", dpi=150)
    plt.close()
    print("Saved: category_distribution.png")

def price_analysis(products):
    """Box plot of prices by category."""
    categories = list(set(p["category"] for p in products))
    data = {cat: [p["price"] for p in products if p["category"] == cat] for cat in categories}
    plt.figure(figsize=(16, 8))
    plt.boxplot(data.values(), labels=data.keys(), vert=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Price ($)")
    plt.title("Price Distribution by Category")
    plt.tight_layout()
    plt.savefig("notebooks/price_distribution.png", dpi=150)
    plt.close()
    print("Saved: price_distribution.png")

def rating_vs_price(products):
    """Scatter plot: rating vs price, colored by category."""
    prices = [p["price"] for p in products]
    ratings = [p.get("rating", 0) for p in products]
    categories = [p["category"] for p in products]
    unique_cats = list(set(categories))
    color_map = {cat: i for i, cat in enumerate(unique_cats)}
    colors = [color_map[c] for c in categories]
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(prices, ratings, c=colors, cmap="tab20", alpha=0.7, s=60)
    plt.xlabel("Price ($)")
    plt.ylabel("Rating")
    plt.title("Price vs Rating by Category")
    plt.colorbar(scatter, label="Category Index")
    plt.tight_layout()
    plt.savefig("notebooks/price_vs_rating.png", dpi=150)
    plt.close()
    print("Saved: price_vs_rating.png")

def sentiment_distribution(products):
    """Analyze and plot sentiment across products using DistilBERT."""
    from shopsmart.sentiment import SentimentAnalyzer
    print("Running sentiment analysis on all products...")
    analyzer = SentimentAnalyzer()
    sentiments = []
    scores = []
    for p in products:
        result = analyzer.analyze_reviews(p.get("reviews", []))
        sentiments.append(result["summary"])
        scores.append(result["avg_score"])
    # Sentiment distribution pie chart
    sent_counts = Counter(sentiments)
    plt.figure(figsize=(8, 8))
    plt.pie(sent_counts.values(), labels=sent_counts.keys(), autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c", "#f39c12"], startangle=140)
    plt.title("Customer Sentiment Distribution")
    plt.tight_layout()
    plt.savefig("notebooks/sentiment_distribution.png", dpi=150)
    plt.close()
    print("Saved: sentiment_distribution.png")
    # Sentiment score histogram
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color="#3498db", edgecolor="black", alpha=0.7)
    plt.xlabel("Sentiment Score (0=Negative, 1=Positive)")
    plt.ylabel("Number of Products")
    plt.title("Sentiment Score Distribution")
    plt.tight_layout()
    plt.savefig("notebooks/sentiment_scores.png", dpi=150)
    plt.close()
    print("Saved: sentiment_scores.png")

def product_clustering(products):
    """K-Means clustering on product features + t-SNE visualization."""
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from shopsmart.config import Config
    print("Generating embeddings for clustering (this may take a minute)...")
    embedder = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)
    
    # Create text representations
    texts = [
        f"{p['title']} {p['category']} {p.get('description', '')}"
        for p in products
    ]
    
    # Get embeddings
    embeddings = embedder.embed_documents(texts)
    X = np.array(embeddings)
    
    # K-Means clustering
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # t-SNE dimensionality reduction
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)
    # Plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap="tab10", alpha=0.7, s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Product Embeddings Visualization (t-SNE + K-Means Clustering)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig("notebooks/product_clusters.png", dpi=150)
    plt.close()
    print("Saved: product_clusters.png")
    # Print cluster summary
    print(f"\nCluster Summary ({n_clusters} clusters):")
    for i in range(n_clusters):
        cluster_products = [products[j] for j in range(len(products)) if clusters[j] == i]
        cats = Counter([p["category"] for p in cluster_products])
        top_cat = cats.most_common(1)[0][0]
        print(f"  Cluster {i}: {len(cluster_products)} products, dominant category: {top_cat}")
        
if __name__ == "__main__":
    products = fetch_products()
    print("\n--- 1. Category Distribution ---")
    category_distribution(products)
    print("\n--- 2. Price Analysis ---")
    price_analysis(products)
    print("\n--- 3. Rating vs Price ---")
    rating_vs_price(products)
    print("\n--- 4. Sentiment Distribution ---")
    sentiment_distribution(products)
    print("\n--- 5. Product Clustering (t-SNE) ---")
    product_clustering(products)
    print("\nAll analysis complete! Check the notebooks/ folder for saved charts.")   