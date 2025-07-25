import redis
import json
import logging
import pickle
from typing import Any, Optional, Dict, List
import hashlib
from datetime import datetime, timedelta
import torch

from ..config.settings import config

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based caching layer for embeddings and model outputs."""
    
    def __init__(self, redis_config=None):
        self.config = redis_config or config.redis
        self.redis_client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            max_connections=self.config.max_connections,
            decode_responses=False  # Keep as bytes for pickle
        )
        
        self._test_connection()
        
    def _test_connection(self):
        """Test Redis connection."""
        try:
            self.redis_client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key with prefix and hash."""
        hashed_id = hashlib.md5(identifier.encode()).hexdigest()
        return f"{prefix}:{hashed_id}"
    
    def get_merchant_embedding(self, merchant_id: str) -> Optional[torch.Tensor]:
        """Get cached merchant embedding."""
        key = self._generate_key("merchant_embedding", merchant_id)
        
        try:
            data = self.redis_client.get(key)
            if data:
                embedding = pickle.loads(data)
                logger.debug(f"Cache hit for merchant {merchant_id}")
                return embedding
        except Exception as e:
            logger.error(f"Error retrieving merchant embedding: {e}")
            
        return None
    
    def set_merchant_embedding(self, 
                              merchant_id: str, 
                              embedding: torch.Tensor,
                              ttl: int = None) -> bool:
        """Cache merchant embedding."""
        key = self._generate_key("merchant_embedding", merchant_id)
        ttl = ttl or config.model.embedding_cache_ttl
        
        try:
            data = pickle.dumps(embedding)
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Cached merchant embedding for {merchant_id}")
            return True
        except Exception as e:
            logger.error(f"Error caching merchant embedding: {e}")
            return False
    
    def get_deal_score(self, 
                      merchant_id: str, 
                      deal_id: str) -> Optional[Dict[str, float]]:
        """Get cached deal score."""
        key = self._generate_key("deal_score", f"{merchant_id}_{deal_id}")
        
        try:
            data = self.redis_client.get(key)
            if data:
                score_data = json.loads(data.decode())
                logger.debug(f"Cache hit for deal score {merchant_id}_{deal_id}")
                return score_data
        except Exception as e:
            logger.error(f"Error retrieving deal score: {e}")
            
        return None
    
    def set_deal_score(self, 
                      merchant_id: str, 
                      deal_id: str,
                      score_data: Dict[str, float],
                      ttl: int = 300) -> bool:
        """Cache deal score with shorter TTL."""
        key = self._generate_key("deal_score", f"{merchant_id}_{deal_id}")
        
        try:
            data = json.dumps(score_data)
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Cached deal score for {merchant_id}_{deal_id}")
            return True
        except Exception as e:
            logger.error(f"Error caching deal score: {e}")
            return False
    
    def get_graph_data(self, merchant_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached graph data for merchant."""
        key = self._generate_key("graph_data", merchant_id)
        
        try:
            data = self.redis_client.get(key)
            if data:
                graph_data = pickle.loads(data)
                logger.debug(f"Cache hit for graph data {merchant_id}")
                return graph_data
        except Exception as e:
            logger.error(f"Error retrieving graph data: {e}")
            
        return None
    
    def set_graph_data(self, 
                      merchant_id: str, 
                      graph_data: Dict[str, torch.Tensor],
                      ttl: int = 1800) -> bool:
        """Cache graph data."""
        key = self._generate_key("graph_data", merchant_id)
        
        try:
            # Convert tensors to CPU before caching
            cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                       for k, v in graph_data.items()}
            data = pickle.dumps(cpu_data)
            self.redis_client.setex(key, ttl, data)
            logger.debug(f"Cached graph data for {merchant_id}")
            return True
        except Exception as e:
            logger.error(f"Error caching graph data: {e}")
            return False
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple keys."""
        try:
            values = self.redis_client.mget(keys)
            return dict(zip(keys, values))
        except Exception as e:
            logger.error(f"Error in batch get: {e}")
            return {}
    
    def batch_set(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Batch set multiple keys."""
        try:
            pipe = self.redis_client.pipeline()
            for key, value in data.items():
                serialized = pickle.dumps(value)
                pipe.setex(key, ttl, serialized)
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Error in batch set: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error deleting keys with pattern {pattern}: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'N/A'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': info.get('keyspace_hits', 0) / max(
                    info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1), 1
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def invalidate_merchant_cache(self, merchant_id: str) -> bool:
        """Invalidate all cache entries for a merchant."""
        try:
            pattern = f"*:{hashlib.md5(merchant_id.encode()).hexdigest()}"
            deleted = self.delete_pattern(pattern)
            logger.info(f"Invalidated {deleted} cache entries for merchant {merchant_id}")
            return True
        except Exception as e:
            logger.error(f"Error invalidating merchant cache: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for cache."""
        try:
            start_time = datetime.now()
            self.redis_client.ping()
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy',
                'latency_ms': latency,
                'connected': True
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connected': False
            }