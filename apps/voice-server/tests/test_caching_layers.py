import pytest
import numpy as np
from services.pipecat_pipeline.bot import SemanticResponseCache

def test_semantic_response_cache_hit():
    cache = SemanticResponseCache(similarity_threshold=0.85, max_size=50)
    
    # Store a canonical question: "What are your business hours?"
    emb1 = [1.0, 0.0, 0.0, 0.0]
    cache.put("What are your business hours?", emb1, "We are open 9am to 5pm Monday through Friday.")
    
    # Query with identical vector -> Exact hit
    resp, score = cache.get("What are your business hours?", [1.0, 0.0, 0.0, 0.0])
    assert resp == "We are open 9am to 5pm Monday through Friday."
    assert score >= 0.99
    
    # Query with a close semantic neighbor vector ("When do you open?")
    # Cosine similarity between [1.0, 0.0, 0.0, 0.0] and [0.95, 0.1, 0.0, 0.0] is ~0.99
    close_emb = [0.95, 0.1, 0.0, 0.0]
    resp_close, score_close = cache.get("When do you open?", close_emb)
    assert resp_close == "We are open 9am to 5pm Monday through Friday."
    assert score_close >= 0.85

def test_semantic_response_cache_miss_unrelated():
    cache = SemanticResponseCache(similarity_threshold=0.88)
    
    # Stored question vector
    emb1 = [1.0, 0.0, 0.0, 0.0]
    cache.put("What are your business hours?", emb1, "We are open 9am to 5pm.")
    
    # Orthogonal / unrelated question vector ("Can I cancel my appointment?")
    unrelated_emb = [0.0, 1.0, 0.0, 0.0]
    resp, score = cache.get("Can I cancel my appointment?", unrelated_emb)
    assert resp is None
    assert score < 0.88

def test_semantic_response_cache_lru_eviction():
    cache = SemanticResponseCache(max_size=2)
    cache.put("q1", [1.0, 0.0], "ans1")
    cache.put("q2", [0.0, 1.0], "ans2")
    assert len(cache._entries) == 2
    
    # Adding a 3rd should evict the oldest (q1)
    cache.put("q3", [0.5, 0.5], "ans3")
    assert len(cache._entries) == 2
    assert cache._entries[0]["query"] == "q2"
    assert cache._entries[1]["query"] == "q3"
