from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from shopsmart.data_converter import APIDataConverter
from shopsmart.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestor:
    def __init__(self):
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="shopsmart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    # Main method: convert docs into embeddings
    def ingest(self, load_existing=True):
        """Ingest data into the vector store.

        If load_existing is True, just return the existing store.
        Set to False to re-fetch and re-ingest all data.
        """
        if load_existing:
            logger.info("Loading existing vector store from AstraDB")
            return self.vstore

        try:
            logger.info("Fetching products from DummyJSON API....")
            docs = APIDataConverter().convert()
            logger.info(f"Fetched {len(docs)} products, ingesting into AstraDB")
        except Exception as e:
            logger.warning(f"API fetch failed ({e}), falling back to CSV..")
            from shopsmart.csv_fallback import CSVDataConverter
            docs = CSVDataConverter("data/flipkart_product_review.csv").convert()
            logger.info(f"Loaded {len(docs)} products from CSV fallback")

        self.vstore.add_documents(docs)
        logger.info("Data ingestion complete!")
        return self.vstore