import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from ..models.merchant_encoder import MerchantProfileEncoder, GeoFeatureExtractor
from ..models.contextual_scorer import ContextualDealScorer, PrototypicalMemory, RuleBasedSurrogate
from ..algorithms.risk_aware_bandit import RiskAwareBandit, Action, State, Reward
from ..cache.redis_cache import RedisCache
from ..config.settings import config
from .monitoring import MetricsCollector

logger = logging.getLogger(__name__)


# Pydantic models for API
class MerchantRequest(BaseModel):
    merchant_id: str = Field(..., description="Unique merchant identifier")
    city_id: str = Field(..., description="City identifier")
    category: str = Field(..., description="Deal category")
    timestamp: int = Field(..., description="Unix timestamp")


class DealRequest(BaseModel):
    merchant_id: str
    deal_id: str
    deal_params: Dict[str, float] = Field(..., description="Deal parameters")
    category: str
    historical_gp30: float = Field(default=0.0)
    market_demand: float = Field(default=0.5)


class PersonalizationResponse(BaseModel):
    merchant_id: str
    deal_id: str
    score: float
    risk_factor: float
    explanation: Dict[str, Any]
    attention_weights: List[float]
    feature_importance: List[float]
    processing_time_ms: float


class BatchRequest(BaseModel):
    merchant_id: str
    deals: List[DealRequest]
    top_k: int = Field(default=5, ge=1, le=10)


class BatchResponse(BaseModel):
    merchant_id: str
    recommendations: List[PersonalizationResponse]
    total_processing_time_ms: float


class PersonalizationService:
    """Main personalization service combining all components."""
    
    def __init__(self):
        self.cache = RedisCache()
        self.metrics = MetricsCollector()
        
        # Initialize models
        self.merchant_encoder = MerchantProfileEncoder()
        self.contextual_scorer = ContextualDealScorer()
        self.geo_extractor = GeoFeatureExtractor()
        self.bandit = RiskAwareBandit()
        
        # Explainability components
        self.prototypical_memory = PrototypicalMemory()
        self.rule_surrogate = RuleBasedSurrogate()
        
        # Service state
        self.is_initialized = False
        self.model_version = "v2.5.0"
        
    async def initialize(self):
        """Initialize the service."""
        if self.is_initialized:
            return
            
        logger.info("Initializing personalization service...")
        
        # Load pre-trained models (implement loading logic)
        # await self._load_models()
        
        self.is_initialized = True
        logger.info("Personalization service initialized")
        
    async def _load_models(self):
        """Load pre-trained model weights."""
        try:
            # Load merchant encoder
            encoder_path = config.model.merchant_encoder_path
            if os.path.exists(encoder_path):
                self.merchant_encoder.load_state_dict(torch.load(encoder_path))
                logger.info("Loaded merchant encoder")
            
            # Load contextual scorer
            scorer_path = config.model.contextual_scorer_path
            if os.path.exists(scorer_path):
                self.contextual_scorer.load_state_dict(torch.load(scorer_path))
                logger.info("Loaded contextual scorer")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            
    def _prepare_merchant_data(self, request: MerchantRequest) -> Dict[str, torch.Tensor]:
        """Prepare merchant data for encoding."""
        
        # Check cache first
        cached_embedding = self.cache.get_merchant_embedding(request.merchant_id)
        if cached_embedding is not None:
            return {"merchant_embedding": cached_embedding}
        
        # Extract geo features
        geo_features = self.geo_extractor.extract_features(
            request.merchant_id,
            request.city_id,
            request.category,
            request.timestamp
        )
        
        # Create dummy data for demonstration
        # In production, these would come from feature store
        deal_history = torch.randn(1, 50, 5)  # [batch, seq_len, features]
        override_sequence = torch.randn(1, 30, 64)
        graph_data = {
            'x': torch.randn(10, 10),
            'edge_index': torch.randint(0, 10, (2, 20)),
            'batch': torch.zeros(10, dtype=torch.long)
        }
        
        merchant_data = {
            'deal_history': deal_history,
            'override_sequence': override_sequence,
            'graph_data': graph_data,
            'geo_features': geo_features.unsqueeze(0)
        }
        
        return merchant_data
    
    def _prepare_deal_data(self, deal: DealRequest) -> Dict[str, torch.Tensor]:
        """Prepare deal data for scoring."""
        
        deal_data = {
            'discount_percentage': torch.tensor([deal.deal_params['discount_percentage']]),
            'price_point': torch.tensor([deal.deal_params['price_point']]),
            'duration_days': torch.tensor([deal.deal_params['duration_days']]),
            'category': torch.tensor([hash(deal.category) % 50]),
            'historical_gp30': torch.tensor([deal.historical_gp30]),
            'market_demand': torch.tensor([deal.market_demand])
        }
        
        return deal_data
    
    def _prepare_override_history(self, merchant_id: str) -> torch.Tensor:
        """Prepare override history features."""
        # In production, this would come from historical data
        # Dummy: override_rate, avg_severity, count, recent_trend, etc.
        return torch.randn(1, 10)
    
    async def score_single_deal(self, 
                               merchant_request: MerchantRequest,
                               deal: DealRequest) -> PersonalizationResponse:
        """Score a single deal for a merchant."""
        
        start_time = datetime.now()
        
        try:
            # Mock response for demonstration (replace with actual model calls)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Generate mock scores
            base_score = 0.7 + 0.1 * torch.randn(1).item()
            risk_factor = 0.2 + 0.05 * torch.randn(1).item()
            final_score = base_score * (1 - risk_factor * 0.5)
            
            response = PersonalizationResponse(
                merchant_id=merchant_request.merchant_id,
                deal_id=deal.deal_id,
                score=max(0.0, min(1.0, final_score)),
                risk_factor=max(0.0, min(1.0, risk_factor)),
                explanation={
                    "base_score": base_score,
                    "confidence": 1.0 - risk_factor,
                    "features": ["discount", "category", "location"]
                },
                attention_weights=[0.3, 0.2, 0.15, 0.1, 0.25],
                feature_importance=[0.4, 0.3, 0.2, 0.1],
                processing_time_ms=processing_time
            )
            
            # Update metrics
            self.metrics.record_latency('single_deal', processing_time)
            self.metrics.record_score(response.score)
            
            return response
            
        except Exception as e:
            logger.error(f"Error scoring deal: {e}")
            self.metrics.record_error('single_deal')
            raise HTTPException(status_code=500, detail=str(e))
    
    async def score_batch_deals(self, batch_request: BatchRequest) -> BatchResponse:
        """Score multiple deals for a merchant."""
        
        start_time = datetime.now()
        
        try:
            merchant_request = MerchantRequest(
                merchant_id=batch_request.merchant_id,
                city_id="dummy_city",  # Would come from request
                category="dummy_category",  # Would come from request
                timestamp=int(datetime.now().timestamp())
            )
            
            # Score all deals
            scored_deals = []
            for deal in batch_request.deals:
                scored_deal = await self.score_single_deal(merchant_request, deal)
                scored_deals.append(scored_deal)
            
            # Sort by score and take top-k
            scored_deals.sort(key=lambda x: x.score, reverse=True)
            top_recommendations = scored_deals[:batch_request.top_k]
            
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = BatchResponse(
                merchant_id=batch_request.merchant_id,
                recommendations=top_recommendations,
                total_processing_time_ms=total_time
            )
            
            # Update metrics
            self.metrics.record_latency('batch_deals', total_time)
            self.metrics.record_batch_size(len(batch_request.deals))
            
            return response
            
        except Exception as e:
            logger.error(f"Error in batch scoring: {e}")
            self.metrics.record_error('batch_deals')
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_personalized_recommendations(self, 
                                              merchant_id: str,
                                              num_recommendations: int = 5) -> List[DealRequest]:
        """Get personalized deal recommendations using bandit."""
        
        # Create state
        merchant_profile = torch.randn(512)  # Would be actual profile
        state = State(
            merchant_profile=merchant_profile,
            market_context={'demand': 0.7, 'competition': 0.5},
            historical_performance={'avg_gp30': 0.15, 'success_rate': 0.8},
            override_rate=0.1,
            deal_success_rate=0.75
        )
        
        # Generate candidate deals (in production, from deal pool)
        candidate_deals = []
        for i in range(10):  # Generate 10 candidate deals
            deal = DealRequest(
                merchant_id=merchant_id,
                deal_id=f"deal_{i}",
                deal_params={
                    'discount_percentage': 0.2 + 0.1 * i,
                    'price_point': 50.0 + 10.0 * i,
                    'duration_days': 7 + i
                },
                category=f"category_{i % 5}",
                historical_gp30=0.1 + 0.05 * i,
                market_demand=0.5 + 0.1 * i
            )
            candidate_deals.append(deal)
        
        # Convert to actions
        actions = []
        for deal in candidate_deals:
            action = Action(
                id=deal.deal_id,
                type='explore' if np.random.random() < 0.3 else 'safe',
                deal_params=deal.deal_params,
                risk_score=0.1 + 0.1 * np.random.random(),
                expected_gp30=deal.historical_gp30,
                category=deal.category
            )
            actions.append(action)
        
        # Select best actions
        selected_actions = []
        for _ in range(num_recommendations):
            action, metadata = self.bandit.select_action(state, actions)
            selected_actions.append(action)
            
            # Remove selected action to avoid duplicates
            actions = [a for a in actions if a.id != action.id]
        
        # Convert back to deals
        recommended_deals = []
        for action in selected_actions:
            deal = DealRequest(
                merchant_id=merchant_id,
                deal_id=action.id,
                deal_params=action.deal_params,
                category=action.category,
                historical_gp30=action.expected_gp30
            )
            recommended_deals.append(deal)
        
        return recommended_deals


# FastAPI app
app = FastAPI(title="AIDG v2.5 Personalization Service", version="2.5.0")
service = PersonalizationService()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    await service.initialize()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": service.model_version,
        "initialized": service.is_initialized
    }


class SingleDealRequest(BaseModel):
    merchant_id: str
    city_id: str
    category: str
    timestamp: int
    deal_id: str
    discount_percentage: float
    price_point: float
    duration_days: int
    historical_gp30: float = 0.0
    market_demand: float = 0.5

@app.post("/score", response_model=PersonalizationResponse)
async def score_deal(request: SingleDealRequest):
    """Score a single deal with simplified request format."""
    
    merchant_request = MerchantRequest(
        merchant_id=request.merchant_id,
        city_id=request.city_id,
        category=request.category,
        timestamp=request.timestamp
    )
    
    deal = DealRequest(
        merchant_id=request.merchant_id,
        deal_id=request.deal_id,
        deal_params={
            'discount_percentage': request.discount_percentage,
            'price_point': request.price_point,
            'duration_days': request.duration_days
        },
        category=request.category,
        historical_gp30=request.historical_gp30,
        market_demand=request.market_demand
    )
    
    return await service.score_single_deal(merchant_request, deal)


@app.post("/score/batch", response_model=BatchResponse)
async def score_batch(batch_request: BatchRequest):
    """Score batch of deals."""
    return await service.score_batch_deals(batch_request)


@app.get("/recommendations/{merchant_id}")
async def get_recommendations(merchant_id: str, num_recommendations: int = 5):
    """Get personalized deal recommendations."""
    recommendations = await service.get_personalized_recommendations(
        merchant_id, num_recommendations
    )
    return {
        "merchant_id": merchant_id,
        "recommendations": recommendations,
        "count": len(recommendations)
    }


@app.post("/test/simple")
async def test_simple():
    """Simple test endpoint for quick validation."""
    return {"status": "ok", "message": "Service is running"}


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return service.metrics.get_metrics()


@app.post("/cache/invalidate/{merchant_id}")
async def invalidate_cache(merchant_id: str):
    """Invalidate merchant cache."""
    success = service.cache.invalidate_merchant_cache(merchant_id)
    return {"merchant_id": merchant_id, "cache_invalidated": success}