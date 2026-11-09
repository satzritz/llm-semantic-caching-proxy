import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SemanticCacheProxy')

app = FastAPI(title='LLM Semantic Caching & Cost-Optimization Proxy', version='1.0.0')

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(REDIS_URL, encoding='utf-8', decode_responses=True)

# Load lightweight embedding model for semantic similarity matching
model = SentenceTransformer('all-MiniLM-L6-v2')
SIMILARITY_THRESHOLD = 0.92

class LLMRequest(BaseModel):
    prompt: str = Field(..., description='User prompt for LLM inference')
    temperature: float = 0.7
    max_tokens: int = 500

class LLMResponse(BaseModel):
    response: str
    cached: bool
    latency_ms: float
    similarity_score: float = None

@app.post('/v1/chat/completions', response_model=LLMResponse)
async def proxy_completion(req: LLMRequest):
    import time
    start_time = time.time()
    
    prompt = req.prompt.strip()
    # Compute embedding
    embedding = model.encode(prompt).astype(np.float32).tobytes()
    
    # Vector similarity search in Redis (using vector index or cached key scan)
    # For robust enterprise proxy, we check keys or cosine similarity against stored vectors
    cached_keys = await redis_client.keys('cache:vec:*')
    best_score = 0.0
    best_response = None
    
    for key in cached_keys:
        stored_emb_b64 = await redis_client.get(key)
        if stored_emb_b64:
            stored_emb = np.frombuffer(bytes.fromhex(stored_emb_b64), dtype=np.float32)
            query_emb = np.frombuffer(embedding, dtype=np.float32)
            score = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
            if score > best_score:
                best_score = score
                best_response = await redis_client.get(key.replace('vec:', 'resp:'))

    if best_score >= SIMILARITY_THRESHOLD and best_response:
        latency = (time.time() - start_time) * 1000
        logger.info(f'Cache HIT! Similarity: {best_score:.4f} | Latency: {latency:.2f}ms')
        return LLMResponse(response=best_response, cached=True, latency_ms=latency, similarity_score=float(best_score))

    # Cache MISS - Simulate upstream LLM call (e.g. Gemini API / OpenAI)
    await asyncio.sleep(0.4) # Simulate upstream network latency ~400ms
    simulated_response = f'Enterprise AI Generated Response for: "{prompt}"'
    
    # Store embedding and response in Redis
    hex_emb = np.frombuffer(embedding, dtype=np.float32).tobytes().hex()
    cache_id = str(abs(hash(prompt)))
    await redis_client.set(f'cache:vec:{cache_id}', hex_emb, ex=86400)
    await redis_client.set(f'cache:resp:{cache_id}', simulated_response, ex=86400)
    
    latency = (time.time() - start_time) * 1000
    logger.info(f'Cache MISS. Upstream call made. Latency: {latency:.2f}ms')
    return LLMResponse(response=simulated_response, cached=False, latency_ms=latency, similarity_score=None)

@app.get('/health')
async def health():
    return {'status': 'healthy', 'service': 'semantic-cache-proxy'}
