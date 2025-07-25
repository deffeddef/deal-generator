import time
import logging
from typing import Dict, Any, List
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from datetime import datetime, timedelta
import threading
import queue

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and exposes metrics for the personalization service."""
    
    def __init__(self):
        # Request metrics
        self.request_counter = Counter(
            'aidg_requests_total',
            'Total number of requests',
            ['method', 'status']
        )
        
        self.request_duration = Histogram(
            'aidg_request_duration_seconds',
            'Request duration in seconds',
            ['method']
        )
        
        # Model metrics
        self.model_latency = Histogram(
            'aidg_model_inference_duration_seconds',
            'Model inference duration',
            ['model_type']
        )
        
        self.model_score = Histogram(
            'aidg_model_score_distribution',
            'Distribution of model scores',
            ['model_type'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Business metrics
        self.gp30_uplift = Gauge(
            'aidg_gp30_uplift',
            'Current GP30 uplift',
            ['merchant_id']
        )
        
        self.override_rate = Gauge(
            'aidg_override_rate',
            'Override rate by merchant',
            ['merchant_id']
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'aidg_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'aidg_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        self.cache_latency = Histogram(
            'aidg_cache_operation_duration_seconds',
            'Cache operation duration',
            ['operation', 'cache_type']
        )
        
        # Bandit metrics
        self.exploration_rate = Gauge(
            'aidg_exploration_rate',
            'Current exploration rate'
        )
        
        self.action_counts = Counter(
            'aidg_action_counts_total',
            'Total actions taken',
            ['action_type', 'action_id']
        )
        
        self.reward_distribution = Histogram(
            'aidg_reward_distribution',
            'Distribution of rewards received',
            buckets=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        
        # Error metrics
        self.errors_total = Counter(
            'aidg_errors_total',
            'Total errors',
            ['error_type']
        )
        
        # System metrics
        self.active_requests = Gauge(
            'aidg_active_requests',
            'Number of active requests'
        )
        
        self.memory_usage = Gauge(
            'aidg_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        # Rolling window metrics
        self.rolling_metrics = {
            'request_times': [],
            'gp30_values': [],
            'override_rates': [],
            'rewards': []
        }
        
        self.rolling_window = 3600  # 1 hour
        
        # Background thread for cleanup
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def record_request(self, method: str, duration: float, status: str = 'success'):
        """Record a request."""
        self.request_counter.labels(method=method, status=status).inc()
        self.request_duration.labels(method=method).observe(duration)
        
        # Add to rolling window
        self.rolling_metrics['request_times'].append({
            'timestamp': datetime.now(),
            'duration': duration,
            'method': method
        })
    
    def record_model_inference(self, model_type: str, duration: float, score: float):
        """Record model inference."""
        self.model_latency.labels(model_type=model_type).observe(duration)
        self.model_score.labels(model_type=model_type).observe(score)
    
    def record_score(self, score: float):
        """Record a model score."""
        self.model_score.labels(model_type='contextual').observe(score)
    
    def record_latency(self, operation: str, duration: float):
        """Record latency for an operation."""
        self.request_duration.labels(method=operation).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """Record a cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record a cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_cache_operation(self, operation: str, cache_type: str, duration: float):
        """Record cache operation duration."""
        self.cache_latency.labels(operation=operation, cache_type=cache_type).observe(duration)
    
    def record_error(self, error_type: str):
        """Record an error."""
        self.errors_total.labels(error_type=error_type).inc()
    
    def record_exploration_rate(self, rate: float):
        """Update exploration rate metric."""
        self.exploration_rate.set(rate)
    
    def record_action(self, action_type: str, action_id: str):
        """Record an action."""
        self.action_counts.labels(action_type=action_type, action_id=action_id).inc()
    
    def record_reward(self, reward: float):
        """Record a reward."""
        self.reward_distribution.observe(reward)
        
        self.rolling_metrics['rewards'].append({
            'timestamp': datetime.now(),
            'reward': reward
        })
    
    def record_gp30_uplift(self, merchant_id: str, uplift: float):
        """Record GP30 uplift for a merchant."""
        self.gp30_uplift.labels(merchant_id=merchant_id).set(uplift)
        
        self.rolling_metrics['gp30_values'].append({
            'timestamp': datetime.now(),
            'merchant_id': merchant_id,
            'uplift': uplift
        })
    
    def record_override_rate(self, merchant_id: str, rate: float):
        """Record override rate for a merchant."""
        self.override_rate.labels(merchant_id=merchant_id).set(rate)
        
        self.rolling_metrics['override_rates'].append({
            'timestamp': datetime.now(),
            'merchant_id': merchant_id,
            'rate': rate
        })
    
    def record_batch_size(self, size: int):
        """Record batch processing size."""
        # Could add a specific counter for batch sizes
        pass
    
    def increment_active_requests(self):
        """Increment active requests counter."""
        self.active_requests.inc()
    
    def decrement_active_requests(self):
        """Decrement active requests counter."""
        self.active_requests.dec()
    
    def _cleanup_old_metrics(self):
        """Background thread to clean up old rolling metrics."""
        while True:
            try:
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(seconds=self.rolling_window)
                
                for metric_list in self.rolling_metrics.values():
                    metric_list[:] = [
                        item for item in metric_list
                        if item['timestamp'] > cutoff_time
                    ]
                
                time.sleep(60)  # Clean up every minute
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                time.sleep(300)  # Retry in 5 minutes
    
    def get_rolling_averages(self, window_minutes: int = 60) -> Dict[str, float]:
        """Get rolling averages for key metrics."""
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # Average request duration
        recent_requests = [
            item['duration'] for item in self.rolling_metrics['request_times']
            if item['timestamp'] > cutoff_time
        ]
        avg_duration = np.mean(recent_requests) if recent_requests else 0.0
        
        # Average GP30 uplift
        recent_gp30 = [
            item['uplift'] for item in self.rolling_metrics['gp30_values']
            if item['timestamp'] > cutoff_time
        ]
        avg_gp30 = np.mean(recent_gp30) if recent_gp30 else 0.0
        
        # Average override rate
        recent_overrides = [
            item['rate'] for item in self.rolling_metrics['override_rates']
            if item['timestamp'] > cutoff_time
        ]
        avg_override = np.mean(recent_overrides) if recent_overrides else 0.0
        
        # Average reward
        recent_rewards = [
            item['reward'] for item in self.rolling_metrics['rewards']
            if item['timestamp'] > cutoff_time
        ]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        return {
            'avg_request_duration': avg_duration,
            'avg_gp30_uplift': avg_gp30,
            'avg_override_rate': avg_override,
            'avg_reward': avg_reward,
            'window_minutes': window_minutes
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        
        rolling = self.get_rolling_averages()
        
        return {
            'rolling_averages': rolling,
            'exploration_rate': self.exploration_rate._value.get(),
            'active_requests': self.active_requests._value.get(),
            'cache_stats': self.get_cache_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        try:
            hits = self.cache_hists._value.get()
            misses = self.cache_misses._value.get()
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0
            
            return {
                'cache_hits': hits,
                'cache_misses': misses,
                'hit_rate': hit_rate,
                'total_operations': total
            }
        except Exception:
            return {'cache_hits': 0, 'cache_misses': 0, 'hit_rate': 0.0, 'total_operations': 0}
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest().decode('utf-8')
    
    def create_dashboard(self) -> Dict[str, Any]:
        """Create dashboard configuration."""
        
        return {
            'dashboard': {
                'title': 'AIDG v2.5 Metrics Dashboard',
                'refresh_interval': 30,
                'panels': [
                    {
                        'title': 'Request Rate',
                        'type': 'graph',
                        'query': 'rate(aidg_requests_total[5m])',
                        'unit': 'req/s'
                    },
                    {
                        'title': 'Model Latency',
                        'type': 'graph',
                        'query': 'histogram_quantile(0.95, rate(aidg_model_inference_duration_seconds_bucket[5m]))',
                        'unit': 'seconds'
                    },
                    {
                        'title': 'GP30 Uplift',
                        'type': 'stat',
                        'query': 'avg(aidg_gp30_uplift)',
                        'unit': 'percentage'
                    },
                    {
                        'title': 'Exploration Rate',
                        'type': 'stat',
                        'query': 'aidg_exploration_rate',
                        'unit': 'percentage'
                    },
                    {
                        'title': 'Cache Hit Rate',
                        'type': 'stat',
                        'query': 'rate(aidg_cache_hits_total[5m]) / (rate(aidg_cache_hits_total[5m]) + rate(aidg_cache_misses_total[5m]))',
                        'unit': 'percentage'
                    },
                    {
                        'title': 'Error Rate',
                        'type': 'graph',
                        'query': 'rate(aidg_errors_total[5m])',
                        'unit': 'errors/s'
                    }
                ]
            }
        }


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alert_thresholds = {
            'latency_p95': 0.1,  # 100ms
            'error_rate': 0.05,   # 5%
            'gp30_drop': -0.2,    # -20%
            'override_rate': 0.3   # 30%
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check current metrics against thresholds."""
        
        alerts = []
        
        # Check latency
        rolling = self.metrics.get_rolling_averages(window_minutes=5)
        if rolling['avg_request_duration'] > self.alert_thresholds['latency_p95']:
            alerts.append({
                'type': 'latency',
                'severity': 'warning',
                'message': f"High latency: {rolling['avg_request_duration']:.2f}s",
                'threshold': self.alert_thresholds['latency_p95']
            })
        
        # Check GP30 uplift
        if rolling['avg_gp30_uplift'] < self.alert_thresholds['gp30_drop']:
            alerts.append({
                'type': 'performance',
                'severity': 'critical',
                'message': f"GP30 uplift dropped: {rolling['avg_gp30_uplift']:.2f}%",
                'threshold': self.alert_thresholds['gp30_drop']
            })
        
        # Check override rate
        if rolling['avg_override_rate'] > self.alert_thresholds['override_rate']:
            alerts.append({
                'type': 'quality',
                'severity': 'warning',
                'message': f"High override rate: {rolling['avg_override_rate']:.2f}%",
                'threshold': self.alert_thresholds['override_rate']
            })
        
        return alerts