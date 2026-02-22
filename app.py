from flask import render_template, Flask, request, Response
from prometheus_client import Counter, generate_latest

from shopsmart.data_ingestion import DataIngestor
from shopsmart.rag_chain import RAGChainBuilder

from dotenv import load_dotenv
load_dotenv()

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")
PREDICTION_COUNT = Counter(
    "rag_prediction_total",
    "Total number of RAG predictions generated"
)

def create_app():

    app = Flask(__name__)
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")
    
    @app.route("/get", methods=["POST"])
    def get_response():

        user_input = request.form["msg"]

        response = rag_chain.invoke(
            {"input" : user_input},
            config={"configurable" : {"session_id" : "user_session"}} 
        )["answer"]

        PREDICTION_COUNT.inc()

        return response
    
    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")
    
    return app

if __name__=="__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000,debug=True)


'''📌 ShopSmart AI – Progress Summary (Today) 22-02--26
🔷 1️⃣ RAG Pipeline Completion
✔ Completed RAG integration using:

LangChain (modern structure using langchain_classic)

AstraDB (vector storage)

HuggingFace embeddings

Groq LLM (Llama 3.1)

✔ Implemented:

History-aware retriever

Context-based answer generation

RunnableWithMessageHistory

Session-based conversation memory

✔ Verified:

Follow-up questions work correctly

Model remembers earlier suggestions

Session ID properly passed

🔷 2️⃣ Flask Backend Integration
✔ Built application factory pattern:
def create_app():
✔ Implemented routes:

/ → UI page

/get → AI inference endpoint

/metrics → Prometheus metrics endpoint

✔ Connected:

Flask → RAG chain

AJAX frontend → Flask backend

🔷 3️⃣ Frontend Improvements (Major UX Upgrade)
✔ Structured Response Formatting

Improved qa_prompt

Enforced clean, numbered output

Removed markdown formatting

Added white-space: pre-wrap; in CSS

✔ Added Typing Animation

“ShopSmart AI is typing…” bubble

Removed automatically after response

Makes UI feel interactive

✔ Added Friendly Error Handling

If API fails → show user-friendly error bubble

Prevents UI crashes

✔ Added Auto-Scroll

Chat window scrolls to latest message

🔷 4️⃣ Monitoring & Observability (Prometheus)
✔ Added default HTTP counter:
REQUEST_COUNT = Counter("http_requests_total", ...)
✔ Added custom business metric:
PREDICTION_COUNT = Counter("rag_prediction_total", ...)
✔ Incremented counter on every successful RAG response:
PREDICTION_COUNT.inc()
✔ Verified:

/metrics endpoint works

Prometheus format exposed

rag_prediction_total increases correctly

System metrics (GC, Python info) visible

🔷 5️⃣ System Stability Verification
✔ Checked:

No runtime errors

No session_id errors

No LangChain config errors

Chat memory working correctly

Metrics working correctly

✔ Confirmed:

System is stable and functioning properly.

🔷 6️⃣ Architecture Achieved

Current system architecture:

User → Browser (AJAX)
       ↓
Flask Backend
       ↓
LangChain RAG
       ↓
AstraDB (Vector Store)
       ↓
Groq LLM
       ↓
Response → Frontend UI

With monitoring layer:

Prometheus ← /metrics endpoint
🔷 7️⃣ Concepts Learned Today

RAG architecture

Session-based conversational memory

Prompt engineering for structured outputs

Frontend asynchronous handling (AJAX)

Error handling best practices

Prometheus metrics integration

Custom business metrics design

AI system observability principles

🔷 8️⃣ Current Project Status

You now have:

✅ Fully working RAG system

✅ Structured conversational AI

✅ Responsive chat UI

✅ UX enhancements

✅ Monitoring-ready backend

✅ Custom inference counter

✅ Production-style app structure

This is now a full-stack AI system, not just a tutorial script.

🎤 If Asked “What Did You Implement?”

You can say:

“We completed the RAG pipeline, integrated session-aware conversational memory, enhanced the frontend UX with structured responses and asynchronous interaction, and implemented Prometheus-based monitoring including a custom inference counter for AI usage tracking.”
'''