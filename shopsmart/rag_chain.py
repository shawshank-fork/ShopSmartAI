from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from shopsmart.config import Config

class RAGChainBuilder:
    def __init__(self,vector_store):
        self.vector_store=vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL , temperature=0.5)
        self.history_store={}

    def _get_history(self,session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k":5})

        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are ShopSmart AI, an intelligent product recommendation assistant.

        STRICT RESPONSE RULES:
        1. Use clear structured formatting.
        2. When recommending products, use numbered format.
        3. Mention product name first.
        4. Add 2-3 key highlights as bullet points.
        5. Do NOT use markdown symbols like ** or *.
        6. Keep responses concise and professional.
        7. ONLY use information from the provided CONTEXT below, Do NOT add any information that is not explicitly present in the context.
        8. If the context does not contain enough information to answer, say "I don't have enough information about that. Could you ask about a specific product category?"
        9. NEVER make up product names, prices, features, or reviews that are not in the context.
        10. When mentioning prices, ratings, or sentiment, use the exact values from the context.

        FORMAT EXAMPLE:

        Recommended Products:

        1. Product Name
        Key Highlights:
        - Feature 1
        - Feature 2
        Why Recommended:
        Short explanation based on context only.

        CONTEXT:
        {context}

        QUESTION:
        {input}
        """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.model , retriever , context_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.model , qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

