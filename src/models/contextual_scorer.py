import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .merchant_encoder import MerchantProfileEncoder


class CrossAttention(nn.Module):
    """Cross-attention layer for merchant-deal interaction."""
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, merchant_emb: torch.Tensor, deal_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            merchant_emb: [batch_size, 512]
            deal_emb: [batch_size, 512]
        Returns:
            attended_features: [batch_size, 512]
        """
        # Add sequence dimension for attention
        merchant_seq = merchant_emb.unsqueeze(0)  # [1, B, 512]
        deal_seq = deal_emb.unsqueeze(0)  # [1, B, 512]
        
        # Cross-attention
        attended, attention_weights = self.multihead_attn(
            deal_seq, merchant_seq, merchant_seq
        )
        
        # Residual connection and normalization
        attended = self.norm1(attended + deal_seq)
        
        # Feed-forward network
        ffn_out = self.ffn(attended)
        
        # Final residual connection
        output = self.norm2(ffn_out + attended)
        
        # Remove sequence dimension
        return output.squeeze(0), attention_weights


class RiskAdjustmentLayer(nn.Module):
    """Adjusts deal scores based on override history and risk factors."""
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.risk_encoder = nn.Sequential(
            nn.Linear(10, 64),  # Override history features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128)
        )
        
        self.adjustment_layer = nn.Sequential(
            nn.Linear(embed_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Risk adjustment factor 0-1
        )
        
    def forward(self, 
                deal_features: torch.Tensor,
                override_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            deal_features: [batch_size, 512]
            override_history: [batch_size, 10] - override rate, severity, etc.
        Returns:
            risk_adjusted_score: [batch_size, 1]
        """
        # Encode override history
        risk_emb = self.risk_encoder(override_history)  # [B, 128]
        
        # Combine features
        combined = torch.cat([deal_features, risk_emb], dim=1)
        
        # Compute adjustment factor
        adjustment = self.adjustment_layer(combined)  # [B, 1]
        
        return adjustment


class ExplainabilityLayer(nn.Module):
    """Generates explanations for deal recommendations."""
    
    def __init__(self, embed_dim: int = 512, num_features: int = 20):
        super().__init__()
        self.attention_analyzer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_features)
        )
        
        self.explanation_generator = nn.Sequential(
            nn.Linear(num_features + embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, 
                attention_weights: torch.Tensor,
                merchant_features: torch.Tensor,
                deal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            attention_weights: [batch_size, seq_len, seq_len]
            merchant_features: [batch_size, 512]
            deal_features: [batch_size, 512]
        Returns:
            explanation: dict with feature_importance and explanation_vector
        """
        # Analyze attention patterns
        feature_importance = self.attention_analyzer(attention_weights.mean(dim=(1, 2)))
        
        # Generate explanation
        combined = torch.cat([feature_importance, deal_features], dim=1)
        explanation_vector = self.explanation_generator(combined)
        
        return {
            'feature_importance': feature_importance,
            'explanation_vector': explanation_vector
        }


class DealFeatureEncoder(nn.Module):
    """Encodes deal-specific features into embeddings."""
    
    def __init__(self, input_dim: int = 15, embed_dim: int = 512):
        super().__init__()
        
        # Deal features: discount, price, category, etc.
        self.category_embedding = nn.Embedding(50, 64)  # 50 categories
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim - 1 + 64, 256),  # -1 for category + 64 for embedding
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim)
        )
        
    def forward(self, deal_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            deal_features: dict with numerical and categorical features
        Returns:
            deal_embedding: [batch_size, 512]
        """
        # Extract category embedding
        category_emb = self.category_embedding(deal_features['category'])
        
        # Combine numerical features
        numerical_features = torch.cat([
            deal_features['discount_percentage'].unsqueeze(1),
            deal_features['price_point'].unsqueeze(1),
            deal_features['duration_days'].unsqueeze(1),
            deal_features['historical_gp30'].unsqueeze(1),
            deal_features['market_demand'].unsqueeze(1)
        ], dim=1)
        
        # Combine all features
        combined = torch.cat([numerical_features, category_emb], dim=1)
        
        return self.feature_encoder(combined)


class ContextualDealScorer(nn.Module):
    """Complete contextual deal scoring system."""
    
    def __init__(self, 
                 merchant_encoder: Optional[MerchantProfileEncoder] = None,
                 freeze_merchant_encoder: bool = True):
        super().__init__()
        
        # Merchant encoder (can be pre-trained)
        self.merchant_encoder = merchant_encoder or MerchantProfileEncoder()
        if freeze_merchant_encoder:
            for param in self.merchant_encoder.parameters():
                param.requires_grad = False
                
        # Deal encoder
        self.deal_encoder = DealFeatureEncoder()
        
        # Cross-attention
        self.cross_attention = CrossAttention()
        
        # Risk adjustment
        self.risk_adjuster = RiskAdjustmentLayer()
        
        # Explainability
        self.explainability = ExplainabilityLayer()
        
        # Final scoring
        self.scoring_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Score between 0-1
        )
        
    def forward(self, 
                merchant_data: Dict[str, torch.Tensor],
                deal_data: Dict[str, torch.Tensor],
                override_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            merchant_data: dict with all merchant features
            deal_data: dict with deal features
            override_history: [batch_size, 10] override metrics
            
        Returns:
            results: dict with score, explanation, and metadata
        """
        # Encode merchant profile
        merchant_emb = self.merchant_encoder(**merchant_data)
        
        # Encode deal features
        deal_emb = self.deal_encoder(deal_data)
        
        # Cross-attention for personalization
        attended_features, attention_weights = self.cross_attention(
            merchant_emb, deal_emb
        )
        
        # Risk adjustment
        risk_factor = self.risk_adjuster(attended_features, override_history)
        
        # Base score
        base_score = self.scoring_layer(attended_features)
        
        # Final score with risk adjustment
        final_score = base_score * risk_factor
        
        # Generate explanations
        explanation = self.explainability(
            attention_weights,
            merchant_emb,
            attended_features
        )
        
        return {
            'score': final_score,
            'base_score': base_score,
            'risk_factor': risk_factor,
            'attention_weights': attention_weights,
            'feature_importance': explanation['feature_importance'],
            'explanation_vector': explanation['explanation_vector']
        }


class PrototypicalMemory:
    """Memory bank for storing prototypical merchants per category."""
    
    def __init__(self, k: int = 100, embedding_dim: int = 512):
        self.k = k
        self.embedding_dim = embedding_dim
        self.prototypes = {}  # category -> list of (merchant_id, embedding)
        
    def add_prototype(self, 
                     category: str, 
                     merchant_id: str, 
                     embedding: torch.Tensor):
        """Add a merchant as prototype for a category."""
        if category not in self.prototypes:
            self.prototypes[category] = []
            
        self.prototypes[category].append((merchant_id, embedding.cpu()))
        
        # Keep only top-k most recent
        if len(self.prototypes[category]) > self.k:
            self.prototypes[category] = self.prototypes[category][-self.k:]
    
    def find_similar(self, 
                    merchant_embedding: torch.Tensor,
                    category: str,
                    top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar prototypical merchants."""
        if category not in self.prototypes or not self.prototypes[category]:
            return []
            
        similarities = []
        merchant_emb = merchant_embedding.cpu()
        
        for merchant_id, prototype_emb in self.prototypes[category]:
            similarity = F.cosine_similarity(
                merchant_emb.unsqueeze(0),
                prototype_emb.unsqueeze(0)
            ).item()
            similarities.append((merchant_id, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, path: str):
        """Save prototypes to disk."""
        torch.save(self.prototypes, path)
        
    def load(self, path: str):
        """Load prototypes from disk."""
        try:
            self.prototypes = torch.load(path)
        except FileNotFoundError:
            self.prototypes = {}


class RuleBasedSurrogate:
    """Decision tree surrogate for model explainability."""
    
    def __init__(self, max_depth: int = 5):
        from sklearn.tree import DecisionTreeRegressor
        self.tree = DecisionTreeRegressor(max_depth=max_depth)
        self.is_fitted = False
        
    def fit(self, merchant_features: np.ndarray, scores: np.ndarray):
        """Train surrogate model."""
        self.tree.fit(merchant_features, scores)
        self.is_fitted = True
        
    def explain(self, 
                merchant_features: np.ndarray,
                feature_names: List[str]) -> Dict[str, any]:
        """Generate rule-based explanation."""
        if not self.is_fitted:
            return {"error": "Surrogate model not fitted"}
            
        # Get prediction path
        decision_path = self.tree.decision_path(merchant_features.reshape(1, -1))
        
        # Extract rules
        tree_rules = self._extract_rules(feature_names)
        
        return {
            "prediction": float(self.tree.predict(merchant_features.reshape(1, -1))[0]),
            "rules": tree_rules,
            "feature_importance": self.tree.feature_importances_.tolist()
        }
    
    def _extract_rules(self, feature_names: List[str]) -> List[Dict[str, str]]:
        """Extract human-readable rules from decision tree."""
        from sklearn.tree import _tree
        
        tree_ = self.tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        
        rules = []
        
        def recurse(node, depth, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # Left child (feature <= threshold)
                left_conditions = conditions + [f"{name} <= {threshold:.3f}"]
                recurse(tree_.children_left[node], depth + 1, left_conditions)
                
                # Right child (feature > threshold)
                right_conditions = conditions + [f"{name} > {threshold:.3f}"]
                recurse(tree_.children_right[node], depth + 1, right_conditions)
            else:
                # Leaf node
                value = tree_.value[node][0][0]
                rules.append({
                    "conditions": " AND ".join(conditions),
                    "score": float(value)
                })
        
        recurse(0, 0, [])
        return rules