import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Represents a deal variant action."""
    id: str
    type: str  # 'safe', 'explore', 'hybrid'
    deal_params: Dict[str, float]
    risk_score: float
    expected_gp30: float
    category: str
    baseline_gp30: float = 0.0


@dataclass
class State:
    """State representation for contextual bandit."""
    merchant_profile: torch.Tensor  # [512]
    market_context: Dict[str, float]
    historical_performance: Dict[str, float]
    override_rate: float
    deal_success_rate: float


@dataclass
class Reward:
    """Reward structure for deal outcomes."""
    gp30_uplift: float
    mm_approval_rate: float
    conversion_proxy: float
    override_frequency: float
    severity_score: float = 1.0


class RewardCalculator:
    """Computes composite rewards for deal outcomes."""
    
    def __init__(self, 
                 w1: float = 0.5,  # GP30 uplift weight
                 w2: float = 0.3,  # MM approval rate weight
                 w3: float = 0.15, # Conversion proxy weight
                 w4: float = 0.05): # Risk penalty weight
        self.weights = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4}
        
    def compute_reward(self, reward: Reward) -> float:
        """Compute composite reward."""
        risk_penalty = reward.override_frequency * reward.severity_score * 0.1
        
        composite_reward = (
            self.weights['w1'] * reward.gp30_uplift +
            self.weights['w2'] * reward.mm_approval_rate +
            self.weights['w3'] * reward.conversion_proxy -
            self.weights['w4'] * risk_penalty
        )
        
        return max(-1.0, min(1.0, composite_reward))  # Clip to [-1, 1]


class SafetyGuard:
    """Implements safety constraints for exploration."""
    
    def __init__(self, 
                 baseline_threshold: float = 0.8,
                 max_override_rate: float = 0.3,
                 high_risk_categories: Optional[List[str]] = None):
        self.baseline_threshold = baseline_threshold
        self.max_override_rate = max_override_rate
        self.high_risk_categories = high_risk_categories or {'gambling', 'tobacco', 'adult'}
        
    def validate_action(self, action: Action, state: State) -> Tuple[bool, str]:
        """Validate if action is safe to deploy."""
        
        # Baseline comparison
        if action.expected_gp30 < self.baseline_threshold * action.baseline_gp30:
            return False, f"GP30 {action.expected_gp30:.3f} below baseline threshold"
            
        # Override rate check
        if state.override_rate > self.max_override_rate:
            return False, f"Override rate {state.override_rate:.3f} exceeds maximum"
            
        # Category risk assessment
        if action.category in self.high_risk_categories:
            return False, f"High-risk category: {action.category}"
            
        # Risk score check
        if action.risk_score > 0.8:
            return False, f"Risk score {action.risk_score:.3f} too high"
            
        return True, "Valid"


class ExplorationManager:
    """Manages exploration rate based on performance."""
    
    def __init__(self,
                 initial_rate: float = 0.05,
                 min_rate: float = 0.01,
                 max_rate: float = 0.3,
                 success_threshold: float = 0.8,
                 ramp_up_factor: float = 1.5,
                 ramp_down_factor: float = 0.5):
        self.exploration_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.success_threshold = success_threshold
        self.ramp_up_factor = ramp_up_factor
        self.ramp_down_factor = ramp_down_factor
        self.performance_history = []
        
    def update_rate(self, success_rate: float) -> float:
        """Update exploration rate based on performance."""
        self.performance_history.append(success_rate)
        
        # Use rolling average of last 100 observations
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            
        avg_success = np.mean(self.performance_history[-10:])  # Last 10
        
        if avg_success >= self.success_threshold:
            # Increase exploration
            self.exploration_rate = min(
                self.max_rate,
                self.exploration_rate * self.ramp_up_factor
            )
        else:
            # Decrease exploration
            self.exploration_rate = max(
                self.min_rate,
                self.exploration_rate * self.ramp_down_factor
            )
            
        logger.info(f"Updated exploration rate to {self.exploration_rate:.3f}")
        return self.exploration_rate
    
    def should_explore(self) -> bool:
        """Randomly decide whether to explore based on current rate."""
        return np.random.random() < self.exploration_rate


class ThompsonSampling:
    """Thompson sampling for contextual bandit."""
    
    def __init__(self, state_dim: int = 512, action_dim: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Neural network for context-to-parameter mapping
        self.context_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and log-variance
        )
        
        # Prior parameters
        self.alpha = {}  # Success counts
        self.beta = {}   # Failure counts
        
    def get_action_parameters(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and variance for each action given state."""
        state_tensor = state.merchant_profile  # [512]
        
        # Get parameters from context network
        params = self.context_network(state_tensor)  # [action_dim * 2]
        means = params[:self.action_dim]
        log_vars = params[self.action_dim:]
        
        return means, torch.exp(log_vars)
    
    def select_action(self, 
                     state: State,
                     available_actions: List[Action],
                     risk_guard: SafetyGuard) -> Tuple[Action, Dict[str, float]]:
        """Select action using Thompson sampling with risk constraints."""
        
        # Filter safe actions
        safe_actions = []
        for action in available_actions:
            is_safe, reason = risk_guard.validate_action(action, state)
            if is_safe:
                safe_actions.append(action)
            else:
                logger.debug(f"Filtered unsafe action {action.id}: {reason}")
        
        if not safe_actions:
            # Fallback to safest action (lowest risk score)
            safe_actions = [min(available_actions, key=lambda a: a.risk_score)]
            logger.warning("No safe actions found, using lowest risk")
        
        # Get action parameters
        means, variances = self.get_action_parameters(state)
        
        # Sample from posterior
        samples = torch.normal(means, torch.sqrt(variances))
        
        # Select best safe action
        action_scores = []
        for action in safe_actions:
            action_idx = int(action.id.split('_')[-1])  # Extract index
            if action_idx < len(samples):
                score = samples[action_idx].item()
            else:
                score = 0.0
            action_scores.append((action, score))
        
        # Select action with highest score
        best_action, best_score = max(action_scores, key=lambda x: x[1])
        
        metadata = {
            'score': best_score,
            'safe_actions_count': len(safe_actions),
            'total_actions': len(available_actions)
        }
        
        return best_action, metadata
    
    def update(self, 
               state: State,
               action: Action,
               reward: float,
               reward_calculator: RewardCalculator):
        """Update posterior based on observed reward."""
        action_id = action.id
        
        # Initialize if new action
        if action_id not in self.alpha:
            self.alpha[action_id] = 1.0
            self.beta[action_id] = 1.0
            
        # Update counts
        # Map reward to success/failure (reward > 0 = success)
        if reward > 0:
            self.alpha[action_id] += reward
        else:
            self.beta[action_id] += abs(reward)


class RiskAwareBandit:
    """Main contextual bandit with risk awareness."""
    
    def __init__(self, 
                 state_dim: int = 512,
                 initial_exploration_rate: float = 0.05,
                 risk_threshold: float = 0.1):
        
        self.thompson_sampler = ThompsonSampling(state_dim)
        self.reward_calculator = RewardCalculator()
        self.safety_guard = SafetyGuard()
        self.exploration_manager = ExplorationManager(initial_exploration_rate)
        self.risk_threshold = risk_threshold
        
        # Performance tracking
        self.history = []
        self.action_counts = {}
        self.action_rewards = {}
        
    def select_action(self, 
                     state: State,
                     available_actions: List[Action]) -> Tuple[Action, Dict[str, Any]]:
        """Select best action considering exploration and risk."""
        
        # Check if we should explore
        if self.exploration_manager.should_explore():
            # Use Thompson sampling for exploration
            action, metadata = self.thompson_sampler.select_action(
                state, available_actions, self.safety_guard
            )
            metadata['selection_method'] = 'thompson_sampling'
        else:
            # Exploit: select best action from safe set
            safe_actions = []
            for action in available_actions:
                is_safe, _ = self.safety_guard.validate_action(action, state)
                if is_safe:
                    # Use expected GP30 as score
                    safe_actions.append((action, action.expected_gp30))
            
            if safe_actions:
                action, score = max(safe_actions, key=lambda x: x[1])
                metadata = {
                    'score': score,
                    'selection_method': 'exploitation',
                    'safe_actions_count': len(safe_actions)
                }
            else:
                # Fallback to lowest risk
                action = min(available_actions, key=lambda a: a.risk_score)
                metadata = {
                    'score': action.expected_gp30,
                    'selection_method': 'fallback',
                    'safe_actions_count': 0
                }
        
        return action, metadata
    
    def update(self, 
               state: State,
               action: Action,
               reward: Reward) -> Dict[str, float]:
        """Update bandit with observed reward."""
        
        # Compute composite reward
        composite_reward = self.reward_calculator.compute_reward(reward)
        
        # Update Thompson sampler
        self.thompson_sampler.update(
            state, action, composite_reward, self.reward_calculator
        )
        
        # Update performance tracking
        self.history.append({
            'action_id': action.id,
            'reward': composite_reward,
            'gp30_uplift': reward.gp30_uplift,
            'override_rate': reward.override_frequency
        })
        
        # Update action statistics
        if action.id not in self.action_counts:
            self.action_counts[action.id] = 0
            self.action_rewards[action.id] = []
            
        self.action_counts[action.id] += 1
        self.action_rewards[action.id].append(composite_reward)
        
        # Update exploration rate
        recent_rewards = [h['reward'] for h in self.history[-100:]]
        if recent_rewards:
            success_rate = np.mean([r > 0 for r in recent_rewards])
            new_rate = self.exploration_manager.update_rate(success_rate)
        
        return {
            'composite_reward': composite_reward,
            'total_actions': len(self.action_counts),
            'exploration_rate': self.exploration_manager.exploration_rate
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current bandit statistics."""
        if not self.history:
            return {"message": "No data yet"}
            
        rewards = [h['reward'] for h in self.history]
        return {
            'total_interactions': len(self.history),
            'average_reward': np.mean(rewards),
            'success_rate': np.mean([r > 0 for r in rewards]),
            'action_distribution': self.action_counts,
            'exploration_rate': self.exploration_manager.exploration_rate,
            'recent_performance': np.mean(rewards[-10:]) if len(rewards) >= 10 else None
        }


class BatchConstrainedQLearning:
    """Batch-constrained Q-learning for safe exploration."""
    
    def __init__(self, 
                 state_dim: int = 512,
                 action_dim: int = 10,
                 gamma: float = 0.99,
                 epsilon: float = 0.1):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.memory = []
        
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value for state-action pair."""
        state_action = torch.cat([state, action], dim=-1)
        return self.q_network(state_action)
    
    def select_action(self, 
                     state: State,
                     available_actions: List[Action],
                     safety_guard: SafetyGuard) -> Tuple[Action, Dict[str, Any]]:
        """Select action using batch-constrained Q-learning."""
        
        state_tensor = state.merchant_profile
        
        # Filter safe actions
        safe_actions = []
        for action in available_actions:
            is_safe, _ = safety_guard.validate_action(action, state)
            if is_safe:
                safe_actions.append(action)
        
        if not safe_actions:
            safe_actions = [min(available_actions, key=lambda a: a.risk_score)]
        
        # Convert actions to tensors
        action_tensors = []
        for action in safe_actions:
            action_vec = torch.tensor([
                action.expected_gp30,
                action.risk_score,
                hash(action.category) % 1000 / 1000.0
            ], dtype=torch.float32)
            action_tensors.append(action_vec)
        
        # Compute Q-values
        q_values = []
        for action_vec in action_tensors:
            q = self.get_q_value(state_tensor, action_vec)
            q_values.append(q.item())
        
        # Epsilon-greedy selection
        if np.random.random() < self.epsilon:
            selected_idx = np.random.randint(len(safe_actions))
            method = "epsilon_greedy_random"
        else:
            selected_idx = np.argmax(q_values)
            method = "epsilon_greedy_best"
        
        selected_action = safe_actions[selected_idx]
        metadata = {
            'q_value': q_values[selected_idx],
            'method': method,
            'safe_actions_count': len(safe_actions)
        }
        
        return selected_action, metadata
    
    def update(self, 
               state: State,
               action: Action,
               reward: float,
               next_state: Optional[State] = None):
        """Update Q-network with batch data."""
        
        state_tensor = state.merchant_profile
        action_vec = torch.tensor([
            action.expected_gp30,
            action.risk_score,
            hash(action.category) % 1000 / 1000.0
        ], dtype=torch.float32)
        
        # Store experience
        self.memory.append({
            'state': state_tensor,
            'action': action_vec,
            'reward': torch.tensor([reward], dtype=torch.float32),
            'next_state': next_state.merchant_profile if next_state else None
        })
        
        # Update network periodically
        if len(self.memory) >= 32:
            self._train_batch()
    
    def _train_batch(self):
        """Train Q-network on batch of experiences."""
        batch_size = min(32, len(self.memory))
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        
        for idx in batch_indices:
            exp = self.memory[idx]
            states.append(exp['state'])
            actions.append(exp['action'])
            rewards.append(exp['reward'])
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        
        # Compute Q-values
        q_values = self.get_q_value(states, actions)
        
        # Compute loss (simplified - no next state for now)
        loss = F.mse_loss(q_values, rewards)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._update_target_network()
    
    def _update_target_network(self):
        """Soft update target network."""
        tau = 0.001
        for target_param, param in zip(self.target_network.parameters(), 
                                      self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)