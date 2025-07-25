keep # Groupon AI Deal Generator (AIDG) v2.5 Design Memo

## Metadata
- **Time to draft:** 2 days
- **Difficulty:** 8/10

---

## 1. Contextual Deal Personalization

### Objective
Tailor deal suggestions to merchant personality, rep performance patterns, and geo-local nuances with modular, explainable ML.

### 1.1 Merchant Behavioral Profiles

#### Feature Engineering Pipeline
```
Raw Data → Feature Extractors → Embeddings → Context Vectors
```

**Core Components:**

**1. Deal History Encoder**
- **Sequence Length:** 50 most recent deals
- **Features:** 
  - Discount percentage (normalized 0-1)
  - Category embedding (20-dim one-hot)
  - Price point (log-normalized)
  - Redemption rate (0-1)
  - GP30 uplift (target variable)
- **Architecture:** 3-layer Transformer encoder
- **Output:** 128-dimensional deal embedding

**2. Override Pattern Analyzer**
```python
class OverridePatternAnalyzer:
    def __init__(self):
        self.lstm = LSTM(input_size=64, hidden_size=128, num_layers=2)
        self.attention = MultiHeadAttention(embed_dim=128, num_heads=8)
    
    def forward(self, override_sequence):
        # Input: [batch, seq_len, 64] - deal params → override type → outcome
        lstm_out, _ = self.lstm(override_sequence)
        attended = self.attention(lstm_out, lstm_out, lstm_out)
        return attended[:, -1, :]  # 64-dim override intent vector
```

**3. Sentiment Extractor**
- **Input:** MM feedback text + merchant notes
- **Model:** BERT-base fine-tuned on Groupon corpus
- **Fine-tuning:** 3 epochs on 100k labeled examples
- **Output:** 32-dim sentiment vector (positive/negative/neutral + intensity)

**4. Geo-Local Context**
```python
def extract_geo_features(merchant_id, city_id, category):
    # City embeddings via Node2Vec on merchant-city graph
    city_embedding = city_node2vec[city_id]  # 24-dim
    
    # Category-seasonality vectors
    seasonality = compute_category_seasonality(category)  # 16-dim
    
    # Local competition density
    competition = compute_local_competition(merchant_id)  # 8-dim
    
    return np.concatenate([city_embedding, seasonality, competition])
```

#### Representation Learning Architecture

```
Multi-Modal Merchant Encoder
├── Graph Branch (GNN)
│   ├── Merchant-Deal bipartite graph (50k nodes, 500k edges)
│   ├── GraphSAGE with 3 message-passing layers
│   ├── Edge features: deal similarity, temporal decay
│   └── Output: 256-dim merchant node embedding
├── Sequence Branch (Transformer)
│   ├── Input: chronological deal/override sequences
│   ├── Positional encoding: deal recency weighting
│   ├── 6-layer Transformer encoder
│   └── Output: 128-dim sequence embedding
├── Fusion Layer
│   └── Concatenates all embeddings → 512-dim merchant profile
```

### 1.2 Model Architecture & Explainability

**Core Model: Contextual Deal Scorer**
```python
class ContextualDealScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.merchant_encoder = MerchantProfileEncoder()  # frozen
        self.deal_encoder = DealFeatureEncoder()
        self.cross_attention = CrossAttention(embed_dim=512)
        self.explainability_layer = ExplainabilityLayer()
        self.risk_adjuster = RiskAdjustmentLayer()
        
    def forward(self, merchant_profile, deal_features):
        merchant_emb = self.merchant_encoder(merchant_profile)
        deal_emb = self.deal_encoder(deal_features)
        
        # Cross-attention for personalization
        attended = self.cross_attention(merchant_emb, deal_emb)
        
        # Explainability
        attention_weights = self.cross_attention.get_attention_weights()
        explanation = self.explainability_layer(attention_weights, merchant_profile)
        
        # Risk adjustment
        score = self.risk_adjuster(attended, merchant_profile['override_history'])
        
        return {
            'score': score,
            'explanation': explanation,
            'attention_map': attention_weights
        }
```

**Explainability Components:**

1. **Attention Heatmaps**
   - Visualize which merchant features most influenced deal scoring
   - Web dashboard showing top-5 influential features per recommendation

2. **Prototypical Memory**
   ```python
   class PrototypicalMemory:
       def __init__(self, k=100):
           self.prototypes = {}  # category → [merchant_ids]
           self.update_frequency = 24  # hours
       
       def find_similar_merchants(self, merchant_embedding, category, top_k=5):
           prototypes = self.prototypes[category]
           similarities = cosine_similarity(merchant_embedding, prototypes)
           return prototypes[similarities.argsort()[-top_k:]]
   ```

3. **Rule-based Surrogate**
   - Train decision tree on model outputs
   - Generate IF-THEN rules for MM review
   - Example: "IF merchant_category='restaurant' AND avg_discount>0.4 THEN recommend_high_gp30_deal"

### 1.3 Real-Time Personalization

**Microservice Architecture:**
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

**Caching Strategy:**
- **Redis Cluster:** Merchant embeddings (TTL: 1 hour)
- **Incremental Updates:** Streaming pipeline via Kafka → Flink → Redis
- **A/B Testing:** 5% traffic to new models, gradual rollout

---

## 2. Exploration vs. Exploitation in Deal Variants

### 2.1 Algorithmic Framework

**Contextual Bandit with Risk Constraints**

**State Representation:**
```python
State = {
    'merchant_profile': merchant_embedding,  # 512-dim
    'market_context': {
        'city_demand': float,
        'category_trend': float,
        'competition_density': float,
        'seasonality_factor': float
    },
    'historical_performance': {
        'avg_gp30': float,
        'override_rate': float,
        'conversion_rate': float,
        'deal_success_rate': float
    }
}
```

**Action Space:**
```python
Action = {
    'type': Enum['safe', 'explore', 'hybrid'],
    'deal_params': {
        'discount_percentage': float,
        'price_point': float,
        'duration_days': int,
        'category_focus': str
    },
    'risk_score': float  # 0-1, computed by risk model
}
```

**Thompson Sampling with Risk Bounds:**
```python
class RiskAwareThompsonSampling:
    def __init__(self, alpha=1.0, beta=1.0, risk_threshold=0.1):
        self.alpha = alpha  # prior success
        self.beta = beta    # prior failure
        self.risk_threshold = risk_threshold
        
    def select_action(self, state, available_actions):
        # Compute UCB for each action
        action_scores = []
        
        for action in available_actions:
            # Thompson sampling
            posterior_success = np.random.beta(
                self.alpha[action['id']], 
                self.beta[action['id']]
            )
            
            # Risk adjustment
            risk_score = self.assess_risk(action, state)
            adjusted_score = posterior_success * (1 - risk_score)
            
            action_scores.append({
                'action': action,
                'score': adjusted_score,
                'risk_score': risk_score
            })
        
        # Select best action respecting risk threshold
        valid_actions = [a for a in action_scores 
                        if a['risk_score'] <= self.risk_threshold]
        
        if not valid_actions:
            # Fallback to safest action
            return min(action_scores, key=lambda x: x['risk_score'])
        
        return max(valid_actions, key=lambda x: x['score'])
```

### 2.2 Reward Signal Design

**Composite Reward Function:**
```python
def compute_reward(deal_outcome):
    w1, w2, w3, w4 = 0.5, 0.3, 0.15, 0.05
    
    primary_reward = deal_outcome['gp30_uplift']
    secondary_reward = deal_outcome['mm_approval_rate']
    tertiary_reward = deal_outcome['conversion_proxy']
    
    risk_penalty = compute_risk_penalty(deal_outcome)
    
    return (w1 * primary_reward + 
            w2 * secondary_reward + 
            w3 * tertiary_reward - 
            w4 * risk_penalty)

def compute_risk_penalty(outcome):
    override_freq = outcome['override_frequency']
    severity = outcome['severity_score']
    return override_freq * severity * 0.1
```

### 2.3 Safe Exploration & Risk Throttling

**Safety Guard Clauses:**
```python
class SafetyGuard:
    def __init__(self):
        self.baseline_threshold = 0.8
        self.max_override_rate = 0.3
        self.high_risk_categories = {'gambling', 'tobacco', 'adult'}
    
    def validate_variant(self, variant, merchant_profile):
        # Baseline comparison
        if variant['expected_gp30'] < self.baseline_threshold * merchant_profile['baseline_gp30']:
            return False, "GP30 below baseline threshold"
        
        # Override history check
        if merchant_profile['override_rate'] > self.max_override_rate:
            return False, "High override rate detected"
        
        # Category risk assessment
        if variant.get('category') in self.high_risk_categories:
            return False, "High-risk category"
        
        return True, "Valid"

class ExplorationManager:
    def __init__(self):
        self.exploration_rate = 0.05
        self.success_threshold = 0.8
        self.ramp_up_factor = 1.5
        self.ramp_down_factor = 0.5
    
    def update_exploration_rate(self, success_rate):
        if success_rate > self.success_threshold:
            self.exploration_rate = min(0.3, self.exploration_rate * self.ramp_up_factor)
        else:
            self.exploration_rate = max(0.01, self.exploration_rate * self.ramp_down_factor)
        
        return self.exploration_rate
```

**Real-time Monitoring Dashboard:**
```yaml
Dashboard_Metrics:
  GP30_uplift:
    - Current: real-time
    - Rolling_avg: 7-day
    - Alert_threshold: -20%
  
  Override_rates:
    - Current: real-time
    - Alert_threshold: 30%
  
  Conversion_rates:
    - Current: real-time
    - Baseline_comparison: true
  
  Exploration_metrics:
    - Active_variants: count
    - Success_rate: percentage
    - Risk_score_distribution: histogram

Auto_Rollback_Triggers:
  - GP30_drop: >20% over 2 hours
  - Override_spike: >30% in any 1-hour window
  - System_errors: >5% failure rate
```

---

## Deployment & Monitoring Strategy

### 3.1 Rollout Plan

**Phase 1: Shadow Mode (Week 1)**
- Models run in parallel with existing system
- Log predictions for comparison
- No user impact

**Phase 2: A/B Testing (Week 2)**
- 5% traffic to new personalization
- 10% traffic to exploration framework
- Monitor all key metrics

**Phase 3: Gradual Rollout (Week 3-4)**
- Increase traffic by 10% every 2 days
- Continuous monitoring
- Rollback capability within 5 minutes

### 3.2 Monitoring Infrastructure

**Key Performance Indicators:**
```yaml
Business_KPIs:
  - GP30_uplift: target +15%
  - MM_override_rate: target <20%
  - Deal_acceptance_rate: target +10%
  - Time_to_generate_deal: target <2s

Technical_KPIs:
  - Model_latency_p95: <100ms
  - Cache_hit_rate: >95%
  - System_availability: >99.9%
  - Memory_usage: <80% of provisioned
```

**Alerting Strategy:**
- **P0 (Critical):** GP30 drop >20%, system down
- **P1 (High):** Override rate >30%, latency >500ms
- **P2 (Medium):** Cache hit rate <90%, minor metric deviations

---

## Assumptions & Constraints

### Technical Assumptions
- Feature store provides standardized merchant/deal features
- GPU resources available for model inference (NVIDIA T4 or better)
- Redis cluster with 99.9% availability
- Streaming pipeline (Kafka + Flink) operational

### Data Privacy & Compliance
- **GDPR Compliance:**
  - Differential privacy (ε=1.0) in merchant embeddings
  - Right to deletion: 30-day data retention
  - Consent management for data usage
- **Data Anonymization:**
  - Hash merchant IDs in logs
  - No PII in model features
  - Encrypted data at rest and in transit

### Resource Requirements
- **Compute:** 8 GPUs for training, 4 for inference
- **Storage:** 500GB for embeddings, 10TB for training data
- **Network:** 10Gbps internal bandwidth
- **Team:** 2 ML engineers, 1 data engineer, 1 MLOps engineer

### Timeline & Next Steps
- **Week 1:** Feature engineering pipeline setup
- **Week 2:** Model training and validation
- **Week 3:** A/B testing infrastructure
- **Week 4:** Gradual rollout and monitoring
- **Post-launch:** Continuous monitoring and iteration

---

*Design completed: 2025-07-22*
*Next review: 2025-07-29*