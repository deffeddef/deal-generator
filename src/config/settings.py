import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: str = os.getenv("REDIS_PASSWORD", "")
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    max_connections: int = 50


@dataclass
class ModelConfig:
    merchant_encoder_path: str = os.getenv("MERCHANT_ENCODER_PATH", "models/merchant_encoder.pt")
    contextual_scorer_path: str = os.getenv("CONTEXTUAL_SCORER_PATH", "models/contextual_scorer.pt")
    feature_store_url: str = os.getenv("FEATURE_STORE_URL", "http://localhost:8080")
    embedding_cache_ttl: int = 3600  # 1 hour
    max_batch_size: int = 32
    model_timeout: int = 30  # seconds


@dataclass
class MonitoringConfig:
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "8000"))
    metrics_endpoint: str = "/metrics"
    health_check_endpoint: str = "/health"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Alert thresholds
    latency_p95_threshold: float = 100.0  # ms
    error_rate_threshold: float = 0.05    # 5%
    gp30_drop_threshold: float = 0.2      # 20%
    override_rate_threshold: float = 0.3  # 30%


@dataclass
class ServiceConfig:
    host: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVICE_PORT", "8080"))
    workers: int = int(os.getenv("WORKERS", "4"))
    max_requests_per_worker: int = 1000
    timeout: int = 30


@dataclass
class ABTestConfig:
    enabled: bool = os.getenv("AB_TEST_ENABLED", "true").lower() == "true"
    experiment_name: str = os.getenv("AB_TEST_NAME", "aidg_v25_rollout")
    control_percentage: float = 0.95  # 95% control
    treatment_percentage: float = 0.05  # 5% treatment
    metrics_window: int = 3600  # 1 hour rolling window


@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_SERVERS", "localhost:9092")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "aidg_v25")
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    
    # Topics
    deal_events_topic: str = "deal_events"
    merchant_updates_topic: str = "merchant_updates"
    metrics_topic: str = "aidg_metrics"


from dataclasses import dataclass, field

@dataclass
class Config:
    redis: RedisConfig = field(default_factory=RedisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)


# Global config instance
config = Config()