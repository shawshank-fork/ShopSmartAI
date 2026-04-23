#RAG eval using RAGAS framework
#Measures: faithfulness, answer relevancy, context precision

from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from datasets import Dataset
from shopsmart.config import Config
from shopsmart.data_ingestion import DataIngestor
from shopsmart.rag_chain import RAGChainBuilder
from dotenv import load_dotenv

load_dotenv()

TEST_DATA = [
    {"question": "Recommend a smartphone", "reference": "A smartphone product with price, rating, and reviews"},
    {"question": "What is the best laptop?", "reference": "A laptop product with good ratings and positive reviews"},
    {"question": "Suggest kitchen accessories", "reference": "Kitchen accessory products like spatulas, knives, or utensils"},
    {"question": "Which product has the best reviews?", "reference": "A product with mostly positive customer sentiment and high ratings"},
    {"question": "Recommend something under $20", "reference": "Products priced below $20 with their details"},
    {"question": "What beauty products do you have?", "reference": "Beauty category products like mascara, lipstick, or skincare"},
    {"question": "Suggest a good pair of sunglasses", "reference": "Sunglasses products with style details and pricing"},
    {"question": "What furniture do you recommend?", "reference": "Furniture products like tables, chairs, or beds with reviews"},
]

def run_evaluation():
    print("Initializing RAG pipeline for eval")
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    questions = []
    answers = []
    contexts = []

    for d in TEST_DATA:
        q = d["question"]
        print(f"Evaluating: {q}")

        #Get RAG answer
        result = rag_chain.invoke(
            {"input": q},
            config={"configurable": {"session_id": "eval_session"}}
        )
        answer = result["answer"]

        #get retrived documents
        retrieved_docs = retriever.invoke(q)
        context_list = [doc.page_content for doc in retrieved_docs]

        questions.append(q)
        answers.append(answer)
        contexts.append(context_list)

    #Create dataset for RAGAS
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": [d["reference"] for d in TEST_DATA],
    })


    #configure RAGAS to use groq LLM and Huggingface embeddings
    eval_llm = LangchainLLMWrapper(ChatGroq(model=Config.RAG_MODEL, temperature=0))
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)
    )

    print("\nRunning RAGAS evaluation")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    print("\n" + "=" * 50)
    print("RAG EVALUATION RESULTS")
    print("=" * 50)
    print(results)
    print("=" * 50)

    return results

if __name__ == "__main__":
    run_evaluation()
    
"""
What this does:

Runs 8 test questions through the RAG chain
Measures 3 quality metrics using RAGAS:
Faithfulness — does the answer stick to retrieved context? (no hallucination)
Answer Relevancy — is the answer relevant to the question?
Context Precision — did the retriever fetch the right documents?
"""
     
