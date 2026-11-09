# Real-Time LLM Semantic Caching & Cost-Optimization Proxy

An enterprise-grade middleware proxy that intercepts LLM requests, computes prompt embeddings via , and performs vector similarity matching against a high-performance Redis cache.

## 🚀 Key Performance Highlights
- **Sub-Millisecond Retrieval:** Cuts repeated/similar query inference latency from ~1.5s down to ~15ms.
- **Cost Reduction:** Reduces redundant LLM API token consumption by up to 45%.
- **High Concurrency:** Built with FastAPI and asynchronous Redis connections for zero-downtime microservice architecture.

## 🛠️ Tech Stack
- **Python, FastAPI, Redis, Sentence-Transformers, NumPy, Docker, OCI**
