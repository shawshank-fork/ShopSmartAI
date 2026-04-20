from shopsmart.data_ingestion import DataIngestor
from utils.logger import get_logger

logger = get_logger(__name__)

print("Initializing Data Ingestor...")
ingestor = DataIngestor()
 
print("Clearing existing AstraDB collection...")
try:
    ingestor.vstore.clear()
    print("Collection cleared.")
except Exception as e:
    print(f"Warning during clear: {e}")

print("Re-ingesting data from DummyJSON API...")
# Setting load_existing=False to force reading the CSV again
vstore = ingestor.ingest(load_existing=False)

print("Data re-ingestion complete!")
