# LangChain cant directly access CSV so we need to convert data into documents

import requests
from langchain_core.documents import Document
from shopsmart.sentiment import SentimentAnalyzer
from shopsmart.config import Config


class APIDataConverter:

    def __init__(self, categories=None):
        self.base_url = Config.PRODUCT_API_BASE
        self.categories = categories or Config.PRODUCT_CATEGORIES
        self.sentiment_analyzer = SentimentAnalyzer()

    def fetch_all_products(self):
        all_products = []
        limit = 30
        skip = 0

        while True:
            response = requests.get(
                f"{self.base_url}/products",
                params={"limit": limit, "skip": skip}
            )
            response.raise_for_status()
            data = response.json()

            all_products.extend(data["products"])

            if skip + limit >= data["total"]:
                break
            skip += limit

        return all_products

    def fetch_by_category(self, category_slug):
        response = requests.get(f"{self.base_url}/products/category/{category_slug}")
        response.raise_for_status()
        return response.json()["products"]

    def convert(self):
        """Fetch products and convert to LangChain documents."""
        if self.categories:
            products = []
            for cat in self.categories:
                products.extend(self.fetch_by_category(cat))
        else:
            products = self.fetch_all_products()

        docs = []  # converting each product into a langchain document
        for p in products:
            reviews_text = "\n".join(
                f"  - {r['reviewerName']} ({r['rating']}/5): {r['comment']}"
                for r in p.get("reviews", [])
            )

            # Sentiment analysis on reviews
            sentiment = self.sentiment_analyzer.analyze_reviews(p.get("reviews", []))

            page_content = (
                f"Product: {p['title']}\n"
                f"Brand: {p.get('brand', 'N/A')}\n"
                f"Category: {p['category']}\n"
                f"Price: ${p['price']}\n"
                f"Discount: {p.get('discountPercentage', 0)}%\n"
                f"Rating: {p.get('rating', 'N/A')}/5\n"
                f"Description: {p.get('description', '')}\n"
                f"Availability: {p.get('availabilityStatus', 'Unknown')}\n"
                f"Warranty: {p.get('warrantyInformation', 'N/A')}\n"
                f"Shipping: {p.get('shippingInformation', 'N/A')}\n"
                f"Customer Sentiment: {sentiment['summary']} "
                f"(Score: {sentiment['avg_score']})\n"
                f"Reviews:\n{reviews_text}"
            )

            docs.append(Document(
                page_content=page_content,
                metadata={
                    "product_name": p["title"],
                    "price": p["price"],
                    "category": p["category"],
                    "brand": p.get("brand", ""),
                    "rating": p.get("rating", 0),
                    "thumbnail": p.get("thumbnail", ""),
                    "sentiment_score": sentiment["avg_score"],
                    "sentiment_summary": sentiment["summary"]
                }
            ))
        return docs


# Changes:
# sentiment = self.sentiment.analyze_reviews(...) runs DistilBERT on each product's reviews
# f"Customer Sentiment: ..." adds sentiment to the document text so the LLM can see it
# sentiment_score + sentiment_summary added to metadata for analytics