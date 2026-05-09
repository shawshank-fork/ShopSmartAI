# 🛍️ ShopSmart AI — Intelligent Product Recommendation Engine

> RAG-powered chatbot with sentiment analysis, analytics dashboard, and full MLOps/DevOps pipeline

[![CI/CD](https://github.com/shawshank-fork/ShopSmartAI/actions/workflows/ci.yml/badge.svg)](https://github.com/shawshank-fork/ShopSmartAI/actions)

## ✨ Features

| Feature | Technology |
|---------|-----------|
| 🤖 LLM-powered product recommendations | Groq Llama 3.1 + LangChain |
| 🔍 Semantic vector search | AstraDB (Cassandra) |
| 💬 Sentiment analysis on reviews | DistilBERT (HuggingFace Transformers) |
| 📊 Real-time analytics dashboard | Chart.js + Flask API |
| 📈 Prometheus + Grafana monitoring | Full observability stack |
| 🐳 Docker Compose one-command setup | Flask + Prometheus + Grafana |
| ⚡ CI/CD with GitHub Actions | Lint → Test → Docker Build |
| 🧪 RAG evaluation with RAGAS | Faithfulness, Relevancy, Precision |
| 📉 Product clustering & EDA | t-SNE, K-Means, Matplotlib |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   User UI   │────▶│  Flask App   │────▶│  Groq LLM API   │
│ (Chat/Dash) │     │  (app.py)    │     │ (Llama 3.1 8B)  │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌──────────┐ ┌──────────────┐
     │   AstraDB    │ │ SQLite   │ │  Prometheus   │
     │ Vector Store │ │ Analytics│ │   Metrics     │
     │ (Embeddings) │ │   DB     │ │               │
     └──────────────┘ └──────────┘ └──────┬───────┘
                                          ▼
                                   ┌──────────────┐
                                   │   Grafana     │
                                   │  Dashboards   │
                                   └──────────────┘
```

### Data Pipeline

```
DummyJSON API ──▶ Sentiment Analysis ──▶ Document Creation ──▶ AstraDB
  (194 products)    (DistilBERT)          (LangChain Docs)     (Vector Store)
       │
       ▼
  CSV Fallback (if API fails)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/) (free tier)
- [AstraDB Account](https://astra.datastax.com/) (free tier)
- [HuggingFace Token](https://huggingface.co/settings/tokens)

### 1. Clone & Install

```bash
git clone https://github.com/shawshank-fork/ShopSmartAI.git
cd ShopSmartAI
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
ASTRA_DB_API_ENDPOINT=your_astra_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
ASTRA_DB_KEYSPACE=default_keyspace
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

### 3. Ingest Data

```bash
python reingest_data.py
```

### 4. Run the App

```bash
python app.py
```

Visit:
- **Chat UI**: http://localhost:5000
- **Analytics Dashboard**: http://localhost:5000/dashboard
- **Health Check**: http://localhost:5000/health
- **Metrics**: http://localhost:5000/metrics

### 5. Docker Compose (Full Stack)

```bash
docker-compose up --build
```

This starts Flask (5000), Prometheus (9090), and Grafana (3000).

---

## 📊 RAG Evaluation Results

Evaluated using [RAGAS](https://docs.ragas.io/) framework on 8 test queries:

| Metric | Before Optimization | After Optimization | Change |
|--------|--------------------|--------------------|--------|
| **Faithfulness** | 0.39 | **0.80** | +106% ✅ |
| **Answer Relevancy** | 0.76 | 0.54 | Trade-off |
| **Context Precision** | 0.67 | 0.58 | Trade-off |

> Faithfulness was prioritized — a product recommendation bot should never hallucinate fake products or prices.

---

## 💬 Sentiment Analysis

Each product's reviews are analyzed using **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`):

- Classifies each review as POSITIVE or NEGATIVE
- Computes average sentiment score (0-1 scale)
- Generates summary: "Mostly positive", "Mostly negative", or "Mixed"
- Sentiment data is embedded in product documents for RAG context

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

**Test coverage includes:**
- Config validation
- Sentiment analysis (single text, batch reviews, empty input)
- Analytics DB (log + retrieve round-trip)
- API connectivity (DummyJSON)
- Logger initialization

---

## 📁 Project Structure

```
ShopSmartAI/
├── .github/workflows/ci.yml       # CI/CD pipeline
├── docker-compose.yml              # Full stack orchestration
├── Dockerfile                      # Production container
├── requirements.txt                # Dependencies
├── app.py                          # Flask application
│
├── shopsmart/                      # Core package
│   ├── config.py                   # Centralized configuration
│   ├── data_converter.py           # DummyJSON API → LangChain Documents
│   ├── data_ingestion.py           # AstraDB vector store + CSV fallback
│   ├── rag_chain.py                # RAG pipeline with chat history
│   ├── sentiment.py                # DistilBERT sentiment analyzer
│   ├── analytics.py                # SQLAlchemy interaction tracker
│   └── csv_fallback.py             # Fallback data source
│
├── evaluation/                     # ML evaluation
│   └── rag_eval.py                 # RAGAS evaluation pipeline
│
├── notebooks/                      # Data science analysis
│   └── product_analysis.py         # Clustering, EDA, sentiment viz
│
├── tests/                          # Test suite
│   └── test_basic.py               # 8 tests across all modules
│
├── templates/                      # Flask templates
│   ├── index.html                  # Chat UI
│   └── dashboard.html              # Analytics dashboard
│
├── prometheus/                     # Monitoring configs
│   ├── prometheus.yml              # Docker Compose config
│   ├── prometheus-configmap.yaml   # K8s ConfigMap
│   └── prometheus-deployment.yaml  # K8s Deployment
│
├── grafana/                        # Grafana K8s deployment
├── utils/                          # Logger + custom exceptions
└── data/                           # CSV fallback data
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq (Llama 3.1 8B Instant) |
| **Embeddings** | HuggingFace (BAAI/bge-base-en-v1.5) |
| **Vector DB** | DataStax AstraDB (Cassandra) |
| **NLP** | HuggingFace Transformers (DistilBERT) |
| **Framework** | LangChain |
| **Backend** | Flask |
| **Frontend** | HTML/CSS/JS, Chart.js |
| **Analytics** | SQLAlchemy + SQLite |
| **Monitoring** | Prometheus + Grafana |
| **CI/CD** | GitHub Actions |
| **Container** | Docker + Docker Compose |
| **ML Eval** | RAGAS |
| **Clustering** | scikit-learn (K-Means, t-SNE) |

---

## 👤 Author

**Shashank Lakhaiyar**

---

## 📄 License

This project is for educational and portfolio purposes.
