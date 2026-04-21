
import time
from flask import render_template, Flask, request, Response, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from shopsmart.data_ingestion import DataIngestor
from shopsmart.rag_chain import RAGChainBuilder
from shopsmart.analytics import AnalyticsTracker
from utils.logger import get_logger

from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)

#Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")
PREDICTION_COUNT = Counter(
    "rag_prediction_total",
    "Total number of RAG predictions generated"
)
RESPONSE_LATENCY = Histogram(
    "rag_response_seconds",
    "TAG response latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total RAG errors",
    ["error_type"]
)

def create_app():

    app = Flask(__name__)

    #initialize components
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain = RAGChainBuilder(vector_store).build_chain()
    tracker = AnalyticsTracker()

    logger.info("ShopSmart AI intialized successfully")

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")
    
    @app.route("/get", methods=["POST"])
    def get_response():

        user_input = request.form["msg"]
        start_time = time.time()

        try:
            response = rag_chain.invoke(
                {"input" : user_input},
                config={"configurable" : {"session_id" : "user_session"}} 
            )["answer"]

            response_time_ms = (time.time() - start_time) * 1000

            PREDICTION_COUNT.inc()
            RESPONSE_LATENCY.observe(response_time_ms / 1000)

            # Log to analytics DB
            tracker.log_interaction(
                session_id="user_session",
                user_query=user_input,
                bot_response=response,
                response_time_ms=response_time_ms
            )

            return response

        except Exception as e:
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            logger.error(f"Error processing query: {e}")
            return "Sorry, something went wrong. Please try again.", 500

    
    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "service": "shopsmart-ai"})

    @app.route("/health/ready")
    def readiness():
        """Readiness probe to check if all dependencies are connected"""
        checks = {"vector_store": False, "analytics_db": False}
        try:
            vector_store.as_retriever()
            checks["vector_store"] = True
        except Exception:
            pass
        try:
            tracker.get_total_queries()
            checks["analytics_db"] = True
        except Exception:
            pass

        all_ready = all(checks.values())
        return jsonify({"ready": all_ready, "checks": checks}), 200 if all_ready else 503

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")           

    return app

if __name__=="__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000,debug=True)
