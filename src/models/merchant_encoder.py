import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DealHistoryEncoder(nn.Module):
    """Transformer-based encoder for deal history sequences."""
    
    def __init__(self, input_dim=5, embed_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(50, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(embed_dim, 128)
        
    def forward(self, deal_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            deal_history: [batch_size, seq_len=50, input_dim=5]
        Returns:
            deal_embeddings: [batch_size, 128]
        """
        batch_size, seq_len, _ = deal_history.shape
        
        # Project to embedding space
        x = self.input_projection(deal_history)  # [B, 50, 128]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transpose for transformer: [seq_len, batch_size, embed_dim]
        x = x.transpose(0, 1)
        
        # Apply transformer
        x = self.transformer(x)  # [seq_len, batch_size, 128]
        
        # Global average pooling
        x = x.mean(dim=0)  # [batch_size, 128]
        
        return self.output_projection(x)


class OverridePatternAnalyzer(nn.Module):
    """LSTM with attention for analyzing override patterns."""
    
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        self.output_projection = nn.Linear(hidden_dim, 64)
        
    def forward(self, override_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            override_sequence: [batch_size, seq_len=30, input_dim=64]
        Returns:
            override_intent: [batch_size, 64]
        """
        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(override_sequence)
        
        # Self-attention
        attended, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # Take final state
        output = attended[-1]  # [batch_size, 128]
        
        return self.output_projection(output)


class SentimentExtractor:
    """BERT-based sentiment extraction from text feedback."""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
        # Freeze BERT parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.sentiment_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 32)
        )
        
    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            text_inputs: dict with 'input_ids', 'attention_mask'
        Returns:
            sentiment_vector: [batch_size, 32]
        """
        with torch.no_grad():
            outputs = self.model(**text_inputs)
            
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        return self.sentiment_head(cls_embedding)


class MerchantGraphEncoder(nn.Module):
    """Graph neural network for merchant-deal interactions."""
    
    def __init__(self, input_dim=10, hidden_dim=256, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Build GCN layers
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, 256))
        
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
        Returns:
            graph_embedding: [batch_size, 256]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        # Global mean pooling
        return global_mean_pool(x, batch)


class MerchantProfileEncoder(nn.Module):
    """Complete merchant encoder combining all components."""
    
    def __init__(self):
        super().__init__()
        
        # Sub-encoders
        self.deal_history_encoder = DealHistoryEncoder()
        self.override_analyzer = OverridePatternAnalyzer()
        self.graph_encoder = MerchantGraphEncoder()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 256 + 48, 512),  # All embeddings + geo
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512)
        )
        
    def forward(self, 
                deal_history: torch.Tensor,
                override_sequence: torch.Tensor,
                graph_data: Dict,
                geo_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            deal_history: [batch_size, 50, 5]
            override_sequence: [batch_size, 30, 64]
            graph_data: dict with 'x', 'edge_index', 'batch'
            geo_features: [batch_size, 48]
        Returns:
            merchant_profile: [batch_size, 512]
        """
        # Encode each modality
        deal_emb = self.deal_history_encoder(deal_history)
        override_emb = self.override_analyzer(override_sequence)
        graph_emb = self.graph_encoder(
            graph_data['x'],
            graph_data['edge_index'],
            graph_data['batch']
        )
        
        # Concatenate all embeddings
        combined = torch.cat([
            deal_emb,
            override_emb,
            graph_emb,
            geo_features
        ], dim=1)
        
        return self.fusion(combined)


class GeoFeatureExtractor:
    """Extracts geographic and contextual features."""
    
    def __init__(self, city_embeddings_path: Optional[str] = None):
        self.city_embeddings = {}
        if city_embeddings_path:
            self.load_city_embeddings(city_embeddings_path)
    
    def load_city_embeddings(self, path: str):
        """Load pre-trained city embeddings."""
        try:
            self.city_embeddings = torch.load(path)
            logger.info(f"Loaded city embeddings from {path}")
        except FileNotFoundError:
            logger.warning(f"City embeddings not found at {path}, using random")
            self.city_embeddings = {}
    
    def extract_features(self, 
                        merchant_id: str,
                        city_id: str,
                        category: str,
                        timestamp: int) -> torch.Tensor:
        """
        Extract geographic and contextual features.
        
        Args:
            merchant_id: Unique merchant identifier
            city_id: City identifier
            category: Deal category
            timestamp: Unix timestamp
            
        Returns:
            geo_features: [48] tensor
        """
        features = []
        
        # City embedding (24-dim)
        city_emb = self.city_embeddings.get(
            city_id, 
            torch.randn(24) * 0.1
        )
        features.append(city_emb)
        
        # Category-seasonality (16-dim)
        seasonality = self._compute_seasonality(category, timestamp)
        features.append(seasonality)
        
        # Local competition (8-dim)
        competition = self._compute_competition(merchant_id, city_id, category)
        features.append(competition)
        
        return torch.cat(features)
    
    def _compute_seasonality(self, category: str, timestamp: int) -> torch.Tensor:
        """Compute category-seasonality features."""
        # Placeholder - implement based on historical data
        # For now, return dummy features
        month = (timestamp // (30 * 24 * 3600)) % 12
        
        # Simple sinusoidal seasonality
        seasonality = torch.zeros(16)
        seasonality[0] = np.sin(2 * np.pi * month / 12)
        seasonality[1] = np.cos(2 * np.pi * month / 12)
        
        # Category-specific patterns
        category_hash = hash(category) % 1000
        torch.manual_seed(category_hash)
        seasonality[2:] = torch.randn(14) * 0.1
        
        return seasonality
    
    def _compute_competition(self, merchant_id: str, city_id: str, category: str) -> torch.Tensor:
        """Compute local competition density."""
        # Placeholder - implement based on merchant database
        # For now, return dummy features
        competition_key = f"{city_id}_{category}"
        torch.manual_seed(hash(competition_key) % 1000)
        return torch.randn(8) * 0.1