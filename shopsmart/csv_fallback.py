import pandas as pd
from langchain_core.documents import Document

class CSVDataConverter:
    def __init__(self,file_path: str):
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



"""Why a separate fallback file? old DataConverter logic preserved as a safety net. If the DummyJSON API is ever down, app will till work. """        