import mesa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle
import os

class DQNNetwork(nn.Module):
    """Deep Q-Network for agent decision making"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class AdaptiveNarrativeAgent(mesa.Agent):
    """Enhanced agent with reinforcement learning capabilities"""
    
    AGENT_TYPES = {
        "Influencer": {"base_influence": 0.8, "learning_rate": 0.001},
        "Regular": {"base_influence": 0.5, "learning_rate": 0.0005},
        "Skeptic": {"base_influence": 0.3, "learning_rate": 0.0003},
        "Bot": {"base_influence": 0.6, "learning_rate": 0.002},  # New agent type
        "Counter_Agent": {"base_influence": 0.7, "learning_rate": 0.0015}  # Defensive agent
    }
    
    # Action space
    ACTIONS = {
        0: "ignore",
        1: "spread_belief",
        2: "spread_counter",
        3: "increase_skepticism",
        4: "form_new_connection",
        5: "break_connection"
    }
    
    def __init__(self, model, agent_type=None, enable_learning=True):
        super().__init__(model)
        self.type = agent_type or random.choice(list(self.AGENT_TYPES.keys()))
        self.base_influence = self.AGENT_TYPES[self.type]["base_influence"]
        self.learning_rate = self.AGENT_TYPES[self.type]["learning_rate"]
        
        # Core attributes
        self.beliefs = {}  # {narrative_id: belief_strength}
        self.sentiment = 0.0
        self.connections = []
        self.trust_scores = {}  # Trust in other agents
        self.memory = deque(maxlen=1000)  # Experience replay buffer
        
        # RL Components
        self.enable_learning = enable_learning
        if enable_learning:
            self.state_size = 15  # Engineered features
            self.action_size = len(self.ACTIONS)
            
            # Initialize networks
            self.q_network = DQNNetwork(self.state_size, self.action_size)
            self.target_network = DQNNetwork(self.state_size, self.action_size)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            # Training parameters
            self.epsilon = 1.0  # Exploration rate
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.1
            self.gamma = 0.95  # Discount factor
            self.batch_size = 32
            
            # Update target network
            self.update_target_network()
        
        # Performance tracking
        self.influence_history = []
        self.reward_history = []
        self.actions_taken = {action: 0 for action in self.ACTIONS.values()}
        
    def get_state(self):
        """Extract current state features for RL decision making"""
        if not hasattr(self.model, 'narratives') or not self.model.narratives:
            return np.zeros(self.state_size)
        
        features = []
        
        # Personal belief state
        total_beliefs = len(self.beliefs)
        strong_beliefs = sum(1 for b in self.beliefs.values() if b > 0.7)
        avg_belief_strength = np.mean(list(self.beliefs.values())) if self.beliefs else 0
        
        features.extend([total_beliefs / 10.0, strong_beliefs / 10.0, avg_belief_strength])
        
        # Network position features
        network_size = len(self.connections)
        avg_neighbor_beliefs = 0
        if self.connections:
            neighbor_belief_counts = [len(neighbor.beliefs) for neighbor in self.connections]
            avg_neighbor_beliefs = np.mean(neighbor_belief_counts) if neighbor_belief_counts else 0
        
        features.extend([network_size / 20.0, avg_neighbor_beliefs / 10.0])
        
        # Global narrative landscape
        narrative_count = len(self.model.narratives)
        dominant_narrative_strength = 0
        if hasattr(self.model, 'agents') and self.model.agents:
            # Find most believed narrative
            narrative_believers = {}
            for agent in self.model.agents:
                for nid, belief in agent.beliefs.items():
                    if belief > 0.5:
                        narrative_believers[nid] = narrative_believers.get(nid, 0) + 1
            
            if narrative_believers:
                max_believers = max(narrative_believers.values())
                dominant_narrative_strength = max_believers / len(self.model.agents)
        
        features.extend([narrative_count / 10.0, dominant_narrative_strength])
        
        # Time and social dynamics
        current_step = getattr(self.model, '_step_count', 0)
        time_feature = (current_step % 20) / 20.0  # Cyclic time feature
        sentiment_feature = (self.sentiment + 1) / 2  # Normalize to [0,1]
        
        features.extend([time_feature, sentiment_feature])
        
        # Agent type encoding (one-hot)
        type_encoding = [0] * len(self.AGENT_TYPES)
        if self.type in self.AGENT_TYPES:
            type_idx = list(self.AGENT_TYPES.keys()).index(self.type)
            type_encoding[type_idx] = 1
        
        features.extend(type_encoding)
        
        # Pad or truncate to exact state size
        while len(features) < self.state_size:
            features.append(0.0)
        features = features[:self.state_size]
        
        return np.array(features, dtype=np.float32)
    
    def calculate_reward(self, action, prev_state, new_state):
        """Calculate reward based on action outcomes"""
        reward = 0.0
        
        # Base reward for agent type goals
        if self.type == "Influencer":
            # Reward for spreading beliefs and gaining influence
            if action in [1, 2]:  # Spreading actions
                reward += 0.3
            if len(self.connections) > 5:
                reward += 0.2
        
        elif self.type == "Skeptic":
            # Reward for countering false narratives
            if action in [2, 3]:  # Counter actions
                reward += 0.4
            # Penalty for blind belief spreading
            if action == 1:
                reward -= 0.2
        
        elif self.type == "Counter_Agent":
            # Reward for defensive actions
            if action in [2, 3]:
                reward += 0.5
            # Bonus for breaking harmful connections
            if action == 5:
                reward += 0.1
        
        elif self.type == "Bot":
            # Bots get reward for rapid spreading
            if action == 1:
                reward += 0.3
            if action == 4:  # Form connections
                reward += 0.2
        
        # Global reward signals
        # Penalty for extreme polarization
        if abs(self.sentiment) > 0.8:
            reward -= 0.1
        
        # Reward for maintaining network stability
        if 3 <= len(self.connections) <= 8:
            reward += 0.1
        
        # Exploration bonus (small)
        reward += 0.01
        
        return reward
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if not self.enable_learning:
            # Fall back to rule-based behavior
            return self.rule_based_action()
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def rule_based_action(self):
        """Fallback rule-based decision making"""
        if self.type == "Skeptic":
            return random.choice([2, 3, 0])  # Counter, skepticism, or ignore
        elif self.type == "Influencer":
            return random.choice([1, 4, 1])  # Spread or form connections
        else:
            return random.choice([0, 1, 2])  # General behavior
    
    def execute_action(self, action_idx):
        """Execute the chosen action in the environment"""
        action_name = self.ACTIONS[action_idx]
        self.actions_taken[action_name] += 1
        
        if action_name == "spread_belief":
            self.spread_strongest_belief()
        
        elif action_name == "spread_counter":
            self.spread_counter_narrative()
        
        elif action_name == "increase_skepticism":
            self.increase_skepticism()
        
        elif action_name == "form_new_connection":
            self.form_new_connection()
        
        elif action_name == "break_connection":
            self.break_weakest_connection()
        
        # "ignore" action does nothing intentionally
    
    def spread_strongest_belief(self):
        """Spread the agent's strongest belief"""
        if not self.beliefs:
            return
        
        strongest_narrative = max(self.beliefs, key=self.beliefs.get)
        strongest_belief = self.beliefs[strongest_narrative]
        
        if strongest_belief > 0.3:  # Only spread if confident enough
            for neighbor in self.connections:
                influence_factor = self.base_influence * (strongest_belief ** 0.5)
                neighbor.receive_narrative(strongest_narrative, strongest_belief, influence_factor)
    
    def spread_counter_narrative(self):
        """Attempt to counter the most dominant harmful narrative"""
        if not hasattr(self.model, 'narratives'):
            return
        
        # Find most harmful narrative (most negative sentiment)
        harmful_narratives = [
            (nid, info) for nid, info in self.model.narratives.items() 
            if info['sentiment'] < -0.5
        ]
        
        if harmful_narratives:
            target_nid, _ = min(harmful_narratives, key=lambda x: x[1]['sentiment'])
            
            # Reduce belief in harmful narrative for neighbors
            for neighbor in self.connections:
                if target_nid in neighbor.beliefs:
                    neighbor.beliefs[target_nid] *= 0.8  # Reduce belief
    
    def increase_skepticism(self):
        """Increase skepticism towards all narratives"""
        for narrative_id in list(self.beliefs.keys()):
            self.beliefs[narrative_id] *= 0.9  # Reduce all beliefs slightly
        
        # Spread skepticism to neighbors
        for neighbor in self.connections:
            for nid in neighbor.beliefs:
                neighbor.beliefs[nid] *= 0.95
    
    def form_new_connection(self):
        """Form a new connection with a compatible agent"""
        if len(self.connections) >= 10:  # Connection limit
            return
        
        potential_connections = [
            agent for agent in self.model.agents 
            if agent != self and agent not in self.connections
        ]
        
        if potential_connections:
            # Choose agent with similar beliefs or complementary type
            if self.type == "Counter_Agent":
                # Counter agents prefer connecting to skeptics
                target = random.choice([a for a in potential_connections if a.type == "Skeptic"] or potential_connections)
            else:
                target = random.choice(potential_connections)
            
            self.connections.append(target)
            if self not in target.connections:
                target.connections.append(self)
    
    def break_weakest_connection(self):
        """Break connection with least trusted agent"""
        if not self.connections:
            return
        
        # Simple heuristic: remove random connection
        # In advanced version, could use trust scores
        to_remove = random.choice(self.connections)
        self.connections.remove(to_remove)
        
        if self in to_remove.connections:
            to_remove.connections.remove(self)
    
    def learn_from_experience(self):
        """Train the neural network from experience replay"""
        if not self.enable_learning or len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network for stable learning"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def step(self):
        """Enhanced step function with RL decision making"""
        if not self.enable_learning:
            # Fall back to original behavior
            self.original_step()
            return
        
        # Get current state
        current_state = self.get_state()
        
        # Choose and execute action
        action = self.choose_action(current_state)
        self.execute_action(action)
        
        # Get new state and calculate reward
        new_state = self.get_state()
        reward = self.calculate_reward(action, current_state, new_state)
        
        # Store experience
        self.memory.append((current_state, action, reward, new_state, False))
        self.reward_history.append(reward)
        
        # Learn from experience
        if len(self.memory) > 100:  # Start learning after some experience
            self.learn_from_experience()
        
        # Update influence tracking
        current_influence = len([n for n in self.connections if any(
            nid in self.beliefs and self.beliefs[nid] > 0.5 
            for nid in n.beliefs
        )])
        self.influence_history.append(current_influence)
        
    def original_step(self):
        """Original step behavior for non-learning agents"""
        for narrative_id, belief in self.beliefs.items():
            if belief > 0.5 and random.random() < 0.5:
                for neighbor in self.connections:
                    neighbor.receive_narrative(narrative_id, belief, self.base_influence)
    
    def receive_narrative(self, narrative_id, incoming_belief, sender_influence):
        """Enhanced narrative reception with trust mechanisms"""
        if narrative_id not in self.beliefs:
            self.beliefs[narrative_id] = 0.0
        
        # Trust-based influence adjustment
        trust_factor = 1.0  # Could be enhanced with actual trust scores
        
        # Agent type specific reception
        if self.type == "Skeptic":
            # Skeptics are naturally more resistant
            alpha = 0.1 * trust_factor
        elif self.type == "Counter_Agent":
            # Counter agents evaluate based on narrative sentiment
            narrative_info = self.model.narratives.get(narrative_id, {})
            if narrative_info.get('sentiment', 0) < -0.3:
                alpha = 0.05  # Very resistant to negative narratives
            else:
                alpha = 0.3
        else:
            alpha = 0.3 * trust_factor
        
        # Update belief
        self.beliefs[narrative_id] = (1 - alpha) * self.beliefs[narrative_id] + alpha * incoming_belief
        
        # Update sentiment
        if hasattr(self.model, 'narratives') and narrative_id in self.model.narratives:
            narrative_sentiment = self.model.narratives[narrative_id]['sentiment']
            self.sentiment = 0.7 * self.sentiment + 0.3 * narrative_sentiment
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.enable_learning:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'actions_taken': self.actions_taken,
                'reward_history': self.reward_history
            }, filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        if self.enable_learning and os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.actions_taken = checkpoint.get('actions_taken', self.actions_taken)
            self.reward_history = checkpoint.get('reward_history', [])
    
    def get_performance_metrics(self):
        """Get detailed performance metrics for analysis"""
        return {
            'agent_id': self.unique_id,
            'agent_type': self.type,
            'total_beliefs': len(self.beliefs),
            'strong_beliefs': len([b for b in self.beliefs.values() if b > 0.7]),
            'network_connections': len(self.connections),
            'current_sentiment': self.sentiment,
            'actions_taken': self.actions_taken.copy(),
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'influence_trend': np.mean(self.influence_history[-5:]) if len(self.influence_history) >= 5 else 0,
            'exploration_rate': self.epsilon if self.enable_learning else 0
        }