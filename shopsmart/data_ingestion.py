from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from shopsmart.data_converter import DataConverter
from shopsmart.config import Config

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


        ## Main method for this class is to convert the docs into embeddings
    def ingest(self,load_existing=True): #load existing true means if vector store already has data then just load it
            if load_existing==True:          # we will set load_existing to false if we have replaced the csv file to a new file, so that if it will perform all the steps again
                return self.vstore
            
            docs = DataConverter("data/flipkart_product_review.csv").convert() #Tales csv fo;s and converts them into docs

            self.vstore.add_documents(docs) #adding docs to vector store
            return self.vstore #Returning the vector store after adding documents
        