## langchain cant directly acces csv so we need to cnvert data into document

import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    def __init__(self, file_path:str):
        self.file_path = file_path

    def convert(self):
        df = pd.read_csv(self.file_path)[["product_title", "price", "review"]]

        docs = [
            Document(
                page_content=f"Product: {row['product_title']}\nPrice: Rs {row['price']}\nReview: {row['review']}", 
                metadata={"product_name": row["product_title"], "price": row["price"]}
            )
            for _, row in df.iterrows()
        ]

        return docs