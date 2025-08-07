import mesa
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
from gymnasium import spaces

class PolicyNetwork(nn.Module):
    """Neural network for narrative spreading policy"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class ValueNetwork(nn.Module):
    """Value network for critic in Actor-Critic"""
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class RLNarrativeAgent(mesa.Agent):
    """Enhanced Narrative Agent with Reinforcement Learning capabilities"""
    
    AGENT_TYPES = {
        "Influencer": {"influence": 0.8, "spread_chance": 0.7, "learning_rate": 0.001},
        "Regular": {"influence": 0.5, "spread_chance": 0.5, "learning_rate": 0.002},
        "Skeptic": {"influence": 0.3, "spread_chance": 0.3, "learning_rate": 0.001}
    }

    def __init__(self, model, use_rl=True):
        super().__init__(model)
        self.type = random.choice(list(self.AGENT_TYPES.keys()))
        self.influence = self.AGENT_TYPES[self.type]["influence"]
        self.spread_chance = self.AGENT_TYPES[self.type]["spread_chance"]
        self.beliefs = {}  # {narrative_id: belief_score}
        self.sentiment = 0.0
        self.connections = []
        
        # RL Components
        self.use_rl = use_rl
        if use_rl:
            self.state_dim = 10  # [own_sentiment, avg_neighbor_sentiment, narrative_count, belief_strength, ...]
            self.action_dim = 4  # [spread_aggressive, spread_moderate, ignore, counter_narrative]
            
            self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
            self.value_net = ValueNetwork(self.state_dim)
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), 
                                             lr=self.AGENT_TYPES[self.type]["learning_rate"])
            self.value_optimizer = optim.Adam(self.value_net.parameters(), 
                                            lr=self.AGENT_TYPES[self.type]["learning_rate"])
            
            # Experience replay for training
            self.memory = deque(maxlen=1000)
            self.last_state = None
            self.last_action = None
            self.last_reward = 0
        
        # Performance metrics
        self.successful_spreads = 0
        self.total_attempts = 0
        self.narrative_creation_count = 0

    def get_state_vector(self):
        """Convert agent's current situation to RL state vector"""
        if not self.use_rl:
            return None
            
        state = np.zeros(self.state_dim)
        
        # Feature 0: Own sentiment
        state[0] = self.sentiment
        
        # Feature 1: Average neighbor sentiment
        if self.connections:
            neighbor_sentiments = [conn.sentiment for conn in self.connections]
            state[1] = np.mean(neighbor_sentiments) if neighbor_sentiments else 0
        
        # Feature 2: Number of narratives believed
        state[2] = len([b for b in self.beliefs.values() if b > 0.5])
        
        # Feature 3: Strongest belief
        state[3] = max(self.beliefs.values()) if self.beliefs else 0
        
        # Feature 4: Influence level (normalized)
        state[4] = self.influence
        
        # Feature 5: Network centrality (approximation)
        state[5] = len(self.connections) / max(1, len(self.model.agents) - 1)
        
        # Feature 6: Recent success rate
        if self.total_attempts > 0:
            state[6] = self.successful_spreads / self.total_attempts
        
        # Feature 7-9: Reserved for model-level features
        if hasattr(self.model, 'get_global_state_features'):
            global_features = self.model.get_global_state_features()
            state[7:10] = global_features[:3]
        
        return torch.FloatTensor(state)

    def select_action(self, state):
        """Select action using policy network"""
        if not self.use_rl or state is None:
            return random.randint(0, 3)  # Random action
        
        with torch.no_grad():
            action_probs = self.policy_net(state)
            action = torch.multinomial(action_probs, 1).item()
        
        return action

    def calculate_reward(self, previous_influence, current_influence, action):
        """Calculate reward based on action outcome"""
        if not self.use_rl:
            return 0
        
        base_reward = 0
        
        # Reward for increasing influence
        influence_change = current_influence - previous_influence
        base_reward += influence_change * 10
        
        # Penalty for failed spread attempts
        if action in [0, 1] and influence_change <= 0:  # Spread actions
            base_reward -= 1
        
        # Reward for successful narrative creation
        if action == 3 and hasattr(self, 'created_narrative_this_step'):
            base_reward += 5
            delattr(self, 'created_narrative_this_step')
        
        # Type-specific rewards
        if self.type == "Influencer":
            base_reward *= 1.2  # Influencers get bonus for spreading
        elif self.type == "Skeptic":
            base_reward += 2 if action == 2 else 0  # Skeptics rewarded for ignoring
        
        return base_reward

    def update_policy(self):
        """Update policy using Actor-Critic method"""
        if not self.use_rl or len(self.memory) < 10:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, min(32, len(self.memory)))
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch])
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float)
        next_states = torch.stack([exp[3] for exp in batch])
        
        # Compute values and advantages
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        
        # TD error (advantage)
        advantages = rewards + 0.99 * next_values - values
        
        # Policy loss
        action_probs = self.policy_net(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -torch.mean(torch.log(selected_action_probs) * advantages.detach())
        
        # Value loss
        value_loss = torch.mean(advantages ** 2)
        
        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def step(self):
        """Enhanced step function with RL-based decision making"""
        current_state = self.get_state_vector()
        current_influence = self.calculate_current_influence()
        
        # Select action using policy
        action = self.select_action(current_state)
        
        # Execute action
        previous_influence = current_influence
        reward = 0
        
        if action == 0:  # Aggressive spreading
            self.spread_narratives(aggressiveness=1.0)
        elif action == 1:  # Moderate spreading
            self.spread_narratives(aggressiveness=0.6)
        elif action == 2:  # Ignore/passive
            pass  # Do nothing this step
        elif action == 3:  # Create counter-narrative
            self.attempt_counter_narrative_creation()
        
        # Calculate reward
        new_influence = self.calculate_current_influence()
        reward = self.calculate_reward(previous_influence, new_influence, action)
        
        # Store experience for learning
        if self.use_rl and self.last_state is not None:
            self.memory.append((self.last_state, self.last_action, self.last_reward, current_state))
        
        self.last_state = current_state
        self.last_action = action
        self.last_reward = reward
        
        # Periodic learning
        if self.use_rl and self.model.schedule.steps % 10 == 0:
            self.update_policy()

    def spread_narratives(self, aggressiveness=0.5):
        """Spread narratives with varying aggressiveness"""
        for narrative_id, belief in self.beliefs.items():
            if belief > 0.5:
                spread_prob = self.spread_chance * aggressiveness
                if random.random() < spread_prob:
                    successful_spreads = 0
                    for neighbor in self.connections:
                        if neighbor.receive_narrative(narrative_id, belief, self.influence * aggressiveness):
                            successful_spreads += 1
                    
                    self.successful_spreads += successful_spreads
                    self.total_attempts += len(self.connections)

    def attempt_counter_narrative_creation(self):
        """Attempt to create a counter-narrative"""
        if not self.beliefs:
            return
        
        # Find strongest opposing narrative
        strongest_narrative = max(self.beliefs, key=self.beliefs.get)
        if self.beliefs[strongest_narrative] > 0.7 and random.random() < 0.3:
            # Signal to model to create counter-narrative
            if hasattr(self.model, 'agent_requested_counter_narrative'):
                self.model.agent_requested_counter_narrative(self, strongest_narrative)
                self.created_narrative_this_step = True
                self.narrative_creation_count += 1

    def calculate_current_influence(self):
        """Calculate agent's current influence in the network"""
        if not self.connections:
            return 0
        
        # Influence based on belief strength and network position
        belief_influence = sum(self.beliefs.values())
        network_influence = len(self.connections) * self.influence
        return belief_influence + network_influence * 0.1

    def receive_narrative(self, narrative_id, incoming_belief, sender_influence):
        """Enhanced narrative reception with learning"""
        if narrative_id not in self.beliefs:
            self.beliefs[narrative_id] = 0.0
        
        # Original belief update logic
        alpha = 0.3 * (sender_influence / self.influence if self.influence > 0 else 1.0)
        old_belief = self.beliefs[narrative_id]
        self.beliefs[narrative_id] = (1 - alpha) * old_belief + alpha * incoming_belief
        
        # Update sentiment
        if narrative_id in self.model.narratives:
            narrative_sentiment = self.model.narratives[narrative_id]['sentiment']
            self.sentiment = (self.sentiment + narrative_sentiment) / 2
        
        # Return success indicator for reward calculation
        belief_change = abs(self.beliefs[narrative_id] - old_belief)
        return belief_change > 0.1

    def get_performance_metrics(self):
        """Get agent's performance metrics"""
        return {
            'type': self.type,
            'successful_spreads': self.successful_spreads,
            'total_attempts': self.total_attempts,
            'success_rate': self.successful_spreads / max(1, self.total_attempts),
            'narrative_creations': self.narrative_creation_count,
            'current_influence': self.calculate_current_influence(),
            'beliefs_count': len([b for b in self.beliefs.values() if b > 0.5])
        }

# Legacy agent class for backward compatibility
class NarrativeAgent(RLNarrativeAgent):
    """Backward compatible agent class"""
    def __init__(self, model):
        super().__init__(model, use_rl=False)