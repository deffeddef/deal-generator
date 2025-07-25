# AIDG v2.5 - Groupon AI Deal Generator

A production-ready AI system for contextual deal personalization and safe exploration in Groupon's marketplace.

## Features

### 1. Contextual Deal Personalization
- **Multi-modal merchant encoding** combining deal history, override patterns, and geo-local context
- **Transformer-based sequence models** for temporal pattern recognition
- **Graph Neural Networks** for merchant-deal relationship modeling
- **Real-time feature extraction** with Redis caching

### 2. Safe Exploration & Bandits
- **Risk-aware Thompson Sampling** with safety guardrails
- **Batch-constrained Q-learning** for offline policy optimization
- **Dynamic exploration rate adjustment** based on performance
- **Comprehensive reward signal** incorporating GP30, MM approval, and risk factors

### 3. Explainability
- **Attention heatmaps** showing feature importance
- **Prototypical memory** for finding similar merchants
- **Rule-based surrogate models** for business-friendly explanations

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Gateway   │────│ Personalization  │────│  Model Serving  │
│                 │    │    Service       │    │   (TensorFlow)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                    ┌──────────────────┐    ┌─────────────────┐
                    │   Feature Store  │────│ Embedding Cache │
                    │   (Redis/Feast)  │    │    (Redis)      │
                    └──────────────────┘    └─────────────────┘
                                │                       │
                    ┌──────────────────┐    ┌─────────────────┐
                    │ Stream Processor │────│     Kafka       │
                    │   (Flink)        │    │   (Events)      │
                    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (if not running)
docker run -p 6379:6379 redis:7-alpine

# Start the service
python main.py

# Test the API
curl http://localhost:8080/health
```

### 2. Docker Development

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f aidg_service
```

### 3. Production Deployment

```bash
# Build and run with Docker
docker build -t aidg:v2.5 .
docker run -p 8080:8080 aidg:v2.5
```

## API Usage

### Score Single Deal

```bash
curl -X POST http://localhost:8080/score \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "merchant_123",
    "city_id": "nyc",
    "category": "restaurant",
    "timestamp": 1625097600,
    "deal_id": "deal_001",
    "discount_percentage": 0.25,
    "price_point": 50.0,
    "duration_days": 7
  }'
```

### Batch Scoring

```bash
curl -X POST http://localhost:8080/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "merchant_123",
    "deals": [
      {
        "deal_id": "deal_1",
        "deal_params": {
          "discount_percentage": 0.3,
          "price_point": 50.0,
          "duration_days": 7
        },
        "category": "restaurant"
      }
    ],
    "top_k": 3
  }'
```

### Get Personalized Recommendations

```bash
curl http://localhost:8080/recommendations/merchant_123?num_recommendations=5
```

## Configuration

### Environment Variables

```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8080
WORKERS=4

# Monitoring
PROMETHEUS_PORT=8000
LOG_LEVEL=INFO
```

### Model Configuration

```python
# src/config/settings.py
config = Config(
    model=ModelConfig(
        merchant_encoder_path="models/merchant_encoder.pt",
        contextual_scorer_path="models/contextual_scorer.pt",
        embedding_cache_ttl=3600
    )
)
```

## Monitoring

### Metrics

The service exposes comprehensive metrics via Prometheus:

- **Request metrics**: rate, latency, errors
- **Model metrics**: inference time, score distribution
- **Business metrics**: GP30 uplift, override rates
- **Cache metrics**: hit rate, latency
- **Bandit metrics**: exploration rate, reward distribution

### Grafana Dashboard

Access Grafana at `http://localhost:3000` (admin/admin)

### Health Checks

```bash
# Service health
curl http://localhost:8080/health

# Metrics endpoint
curl http://localhost:8080/metrics

# Cache statistics
curl http://localhost:8080/metrics
```

## Model Details

### Merchant Profile Encoder

- **Deal History**: Transformer encoder (50-sequence, 128-dim output)
- **Override Patterns**: LSTM with attention (30-sequence, 64-dim output)
- **Graph Features**: GCN with 3 layers (256-dim output)
- **Geo Context**: 48-dim city + category features
- **Final embedding**: 512-dim merchant profile

### Contextual Deal Scorer

- **Cross-attention**: Merchant-deal interaction modeling
- **Risk adjustment**: Override history-based risk factors
- **Explainability**: Attention weights + feature importance
- **Output**: Score (0-1) with explanation tokens

### Risk-Aware Bandit

- **Algorithm**: Thompson Sampling with risk constraints
- **Safety**: Override rate, GP30 baseline, category risk checks
- **Exploration**: Dynamic rate adjustment (0.01-0.3)
- **Reward**: Composite (GP30: 50%, MM approval: 30%, conversion: 15%, risk: 5%)

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Load testing
locust -f tests/load_test.py --host=http://localhost:8080
```

## Development

### Project Structure

```
AIDG_v2.5/
├── src/
│   ├── models/           # ML models
│   │   ├── merchant_encoder.py
│   │   └── contextual_scorer.py
│   ├── algorithms/       # Bandit algorithms
│   │   └── risk_aware_bandit.py
│   ├── services/         # API services
│   │   ├── personalization_service.py
│   │   └── monitoring.py
│   ├── cache/            # Redis caching
│   │   └── redis_cache.py
│   └── config/           # Configuration
│       └── settings.py
├── tests/
├── models/               # Saved model weights
├── logs/                 # Application logs
├── data/                 # Training data
├── main.py               # Entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Adding New Features

1. **New Model Components**: Add to `src/models/`
2. **New Algorithms**: Add to `src/algorithms/`
3. **New API Endpoints**: Add to `src/services/personalization_service.py`
4. **Configuration**: Update `src/config/settings.py`

## Performance

### Benchmarks

- **Single deal scoring**: ~50ms (95th percentile)
- **Batch scoring (10 deals)**: ~100ms (95th percentile)
- **Throughput**: ~1000 requests/second (with 4 workers)
- **Cache hit rate**: >95% for merchant embeddings

### Scaling

- **Horizontal**: Add more service instances behind load balancer
- **Vertical**: Increase workers or GPU resources
- **Caching**: Redis cluster for high availability
- **Model serving**: TensorFlow Serving or TorchServe for production

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

Copyright © 2025 Groupon. All rights reserved.