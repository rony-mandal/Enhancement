import mesa
from mesa import Model
import numpy as np
import pandas as pd
import torch
import random
from .agents import RLNarrativeAgent, NarrativeAgent
import os
import pickle

# Import our custom GAN components (assuming they're in the processing module)
try:
    from processing.gan_narrative_generator import GANNarrativeGenerator, TransformerNarrativeGAN, NarrativeEnvironment
except ImportError:
    print("Warning: GAN components not available. Some features will be disabled.")
    GANNarrativeGenerator = None
    TransformerNarrativeGAN = None

class EnhancedNarrativeModel(Model):
    """Enhanced Narrative Model with RL agents and GAN-based narrative generation"""
    
    def __init__(self, num_agents, initial_narratives, enable_counter_narratives=True, 
                 use_rl=True, use_gan=True, model_config=None):
        super().__init__()
        
        self.num_agents = num_agents
        self.narratives = initial_narratives.copy()
        self.counter_narratives = {}
        self.enable_counter_narratives = enable_counter_narratives
        self.use_rl = use_rl
        self.use_gan = use_gan
        self.model_config = model_config or {}
        
        # Initialize GAN components
        if self.use_gan and GANNarrativeGenerator is not None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.gan_generator = GANNarrativeGenerator(device=device)
            self.transformer_gan = TransformerNarrativeGAN(device=device)
            self.gan_trained = False
            
            # Initialize data for new counter-narrative
            self.data[f'narrative_{counter_nid}_believers'] = [0] * (self._step_count - 1)
            self.data[f'narrative_{counter_nid}_influence'] = [0] * (self._step_count - 1)
            
            # Seed counter-narrative
            if self.agents:
                self.agents[0].beliefs[counter_nid] = 1.0

    def _collect_step_data(self):
        """Collect comprehensive data for current step"""
        step_data = {'step': self._step_count}
        
        # Count believers and calculate influence for each narrative
        for nid in self.narratives:
            believers = sum(1 for agent in self.agents if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            step_data[f'narrative_{nid}_believers'] = believers
            
            # Calculate narrative influence (sum of all belief strengths)
            influence = sum(agent.beliefs.get(nid, 0) for agent in self.agents)
            step_data[f'narrative_{nid}_influence'] = influence
        
        # Agent metrics
        if self.agents:
            step_data['avg_sentiment'] = np.mean([agent.sentiment for agent in self.agents])
            
            if self.use_rl:
                rewards = [agent.last_reward for agent in self.agents if hasattr(agent, 'last_reward')]
                step_data['avg_agent_reward'] = np.mean(rewards) if rewards else 0
                
                success_rates = [agent.successful_spreads / max(1, agent.total_attempts) 
                               for agent in self.agents if hasattr(agent, 'successful_spreads')]
                step_data['successful_spread_rate'] = np.mean(success_rates) if success_rates else 0
        else:
            step_data['avg_sentiment'] = 0.0
            if self.use_rl:
                step_data['avg_agent_reward'] = 0.0
                step_data['successful_spread_rate'] = 0.0
        
        # System metrics
        step_data['total_narratives'] = len(self.narratives)
        step_data['counter_narratives_generated'] = len(self.counter_narratives)
        
        # Calculate narrative diversity (entropy)
        narrative_distribution = {}
        for agent in self.agents:
            for nid, belief in agent.beliefs.items():
                if belief > 0.5:
                    narrative_distribution[nid] = narrative_distribution.get(nid, 0) + 1
        
        if narrative_distribution:
            total = sum(narrative_distribution.values())
            entropy = -sum((count/total) * np.log(count/total + 1e-10) 
                          for count in narrative_distribution.values())
            step_data['narrative_diversity'] = entropy
        else:
            step_data['narrative_diversity'] = 0
        
        # Network metrics
        step_data['network_density'] = self._calculate_network_density()
        step_data['avg_clustering'] = self._calculate_average_clustering()
        
        # Append data ensuring consistency
        for key, value in step_data.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                # New metric - pad with zeros and append
                self.data[key] = [0] * (self._step_count - 1) + [value]

    def _calculate_network_density(self):
        """Calculate network density"""
        if len(self.agents) <= 1:
            return 0
        
        total_connections = sum(len(agent.connections) for agent in self.agents)
        max_connections = len(self.agents) * (len(self.agents) - 1)
        return total_connections / max_connections if max_connections > 0 else 0

    def _calculate_average_clustering(self):
        """Calculate average clustering coefficient"""
        if len(self.agents) <= 2:
            return 0
        
        clustering_coeffs = []
        for agent in self.agents:
            if len(agent.connections) < 2:
                clustering_coeffs.append(0)
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = len(agent.connections) * (len(agent.connections) - 1) // 2
            
            for i, conn1 in enumerate(agent.connections):
                for j, conn2 in enumerate(agent.connections[i+1:], i+1):
                    if conn2 in conn1.connections:
                        triangles += 1
            
            clustering = triangles / possible_triangles if possible_triangles > 0 else 0
            clustering_coeffs.append(clustering)
        
        return np.mean(clustering_coeffs) if clustering_coeffs else 0

    def _update_rl_environment(self):
        """Update RL environment state"""
        if not self.rl_environment:
            return
        
        # Update environment with current model state
        self.rl_environment.current_narratives = self.narratives.copy()
        self.rl_environment.step_count = self._step_count
        
        # Calculate total influence for reward calculation
        total_influence = sum(sum(agent.beliefs.values()) for agent in self.agents)
        self.rl_environment.total_influence = total_influence

    def _collect_network_data(self):
        """Collect network structure data"""
        self.network_data.append({
            'step': self._step_count,
            'nodes': [(a.unique_id, a.type) for a in self.agents],
            'edges': [(a.unique_id, n.unique_id) for a in self.agents for n in a.connections]
        })

    def _update_global_trends(self):
        """Update global influence trends"""
        positive_influence = 0
        negative_influence = 0
        neutral_influence = 0
        
        for agent in self.agents:
            if agent.sentiment > 0.3:
                positive_influence += sum(agent.beliefs.values())
            elif agent.sentiment < -0.3:
                negative_influence += sum(agent.beliefs.values())
            else:
                neutral_influence += sum(agent.beliefs.values())
        
        self.global_influence_trends['positive'].append(positive_influence)
        self.global_influence_trends['negative'].append(negative_influence)
        self.global_influence_trends['neutral'].append(neutral_influence)

    def get_enhanced_data_frame(self):
        """Get enhanced DataFrame with all collected metrics"""
        if not self.data['step']:
            return pd.DataFrame()
        
        # Create DataFrame with consistent lengths
        df_data = {}
        max_length = len(self.data['step'])
        
        for key in self.data:
            if isinstance(self.data[key], list) and key != 'event_log':
                current_length = len(self.data[key])
                if current_length < max_length:
                    padded_data = self.data[key] + [0] * (max_length - current_length)
                    df_data[key] = padded_data
                else:
                    df_data[key] = self.data[key][:max_length]
        
        return pd.DataFrame(df_data)

    def get_agent_performance_report(self):
        """Get detailed agent performance report"""
        if not self.use_rl:
            return {}
        
        report = {}
        for agent in self.agents:
            if hasattr(agent, 'get_performance_metrics'):
                report[agent.unique_id] = agent.get_performance_metrics()
        
        return report

    def get_narrative_evolution_report(self):
        """Get narrative evolution and generation report"""
        report = {
            'total_narratives': len(self.narratives),
            'counter_narratives': len(self.counter_narratives),
            'generated_narratives': len([n for n in self.narratives.values() if n.get('generated', False)]),
            'evolution_history': self.narrative_evolution_history,
            'external_events': self.external_events
        }
        
        if self.gan_generator:
            report['gan_training_history'] = getattr(self.gan_generator, 'training_history', {})
            report['gan_model_quality'] = self._evaluate_gan_quality()
        
        return report

    def _evaluate_gan_quality(self):
        """Evaluate GAN model quality"""
        if not self.gan_generator or not self.gan_trained:
            return {'status': 'not_trained'}
        
        # Generate sample narratives and evaluate
        samples = []
        for _ in range(5):
            try:
                sample = self.gan_generator.generate_narrative()
                quality_metrics = self.gan_generator.evaluate_narrative_quality(sample)
                samples.append(quality_metrics)
            except Exception as e:
                continue
        
        if samples:
            avg_quality = np.mean([s['quality'] for s in samples])
            avg_diversity = np.mean([s['diversity'] for s in samples])
            return {
                'status': 'trained',
                'average_quality': avg_quality,
                'average_diversity': avg_diversity,
                'sample_count': len(samples)
            }
        else:
            return {'status': 'generation_failed'}

    def save_enhanced_model(self, filepath):
        """Save enhanced model state including RL and GAN components"""
        model_state = {
            'narratives': self.narratives,
            'counter_narratives': self.counter_narratives,
            'step_count': self._step_count,
            'data': self.data,
            'network_data': self.network_data,
            'config': {
                'num_agents': self.num_agents,
                'use_rl': self.use_rl,
                'use_gan': self.use_gan,
                'enable_counter_narratives': self.enable_counter_narratives,
                'model_config': self.model_config
            },
            'evolution_history': self.narrative_evolution_history,
            'external_events': self.external_events,
            'global_trends': self.global_influence_trends
        }
        
        # Save agent states
        agent_states = []
        for agent in self.agents:
            agent_state = {
                'unique_id': agent.unique_id,
                'type': agent.type,
                'beliefs': agent.beliefs,
                'sentiment': agent.sentiment,
                'influence': agent.influence
            }
            
            if hasattr(agent, 'get_performance_metrics'):
                agent_state['performance'] = agent.get_performance_metrics()
            
            agent_states.append(agent_state)
        
        model_state['agents'] = agent_states
        
        # Save main model state
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        # Save GAN models separately if available
        if self.gan_generator and self.gan_trained:
            gan_path = filepath.replace('.pkl', '_gan.pth')
            try:
                self.gan_generator.save_model(gan_path)
            except Exception as e:
                print(f"Failed to save GAN model: {e}")
        
        print(f"Enhanced model saved to {filepath}")

    def load_enhanced_model(self, filepath):
        """Load enhanced model state"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            # Restore model state
            self.narratives = model_state['narratives']
            self.counter_narratives = model_state['counter_narratives']
            self._step_count = model_state['step_count']
            self.data = model_state['data']
            self.network_data = model_state['network_data']
            self.narrative_evolution_history = model_state['evolution_history']
            self.external_events = model_state['external_events']
            self.global_influence_trends = model_state['global_trends']
            
            # Restore agent states
            for agent_state in model_state['agents']:
                agent_id = agent_state['unique_id']
                if agent_id < len(self.agents):
                    agent = self.agents[agent_id]
                    agent.beliefs = agent_state['beliefs']
                    agent.sentiment = agent_state['sentiment']
                    agent.influence = agent_state['influence']
            
            # Load GAN model if available
            if self.gan_generator:
                gan_path = filepath.replace('.pkl', '_gan.pth')
                if os.path.exists(gan_path):
                    try:
                        self.gan_generator.load_model(gan_path)
                        self.gan_trained = True
                    except Exception as e:
                        print(f"Failed to load GAN model: {e}")
            
            print(f"Enhanced model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

# Legacy compatibility class
class NarrativeModel(EnhancedNarrativeModel):
    """Legacy compatibility wrapper"""
    def __init__(self, num_agents, initial_narratives, enable_counter_narratives=True):
        super().__init__(num_agents, initial_narratives, enable_counter_narratives, 
                         use_rl=False, use_gan=False)

# Factory function for creating models with different configurations
def create_narrative_model(num_agents, initial_narratives, model_type='enhanced', **kwargs):
    """Factory function for creating different types of narrative models
    
    Args:
        model_type: 'basic', 'rl_only', 'gan_only', 'enhanced'
    """
    configs = {
        'basic': {'use_rl': False, 'use_gan': False},
        'rl_only': {'use_rl': True, 'use_gan': False},
        'gan_only': {'use_rl': False, 'use_gan': True},
        'enhanced': {'use_rl': True, 'use_gan': True}
    }
    
    config = configs.get(model_type, configs['enhanced'])
    config.update(kwargs)
    
    return EnhancedNarrativeModel(num_agents, initial_narratives, **config) Pre-train GAN if narratives available
            if initial_narratives:
                self._pretrain_gan()
        else:
            self.gan_generator = None
            self.transformer_gan = None
        
        # Initialize RL Environment
        if self.use_rl and self.gan_generator:
            self.rl_environment = NarrativeEnvironment(self, self.gan_generator)
            self.rl_agent_performance = {}
        else:
            self.rl_environment = None
        
        # Create agents with RL capabilities
        self.agents = []
        for i in range(num_agents):
            if self.use_rl:
                agent = RLNarrativeAgent(self, use_rl=True)
            else:
                agent = NarrativeAgent(self)
            self.agents.append(agent)
        
        # Setup agent connections
        self._setup_agent_network()
        
        # Initialize first narrative
        if initial_narratives and self.agents:
            first_narrative_id = list(initial_narratives.keys())[0]
            self.agents[0].beliefs[first_narrative_id] = 1.0
        
        # Data collection
        self._initialize_data_collection()
        
        # Advanced features
        self.external_events = []
        self.narrative_evolution_history = []
        self.agent_learning_metrics = {}
        self._step_count = 0
        self.global_influence_trends = {'positive': [], 'negative': [], 'neutral': []}

    def _pretrain_gan(self):
        """Pre-train GAN on initial narratives"""
        if not self.gan_generator or self.gan_trained:
            return
        
        print("Pre-training GAN on initial narratives...")
        narrative_texts = [n['text'] for n in self.narratives.values()]
        
        if len(narrative_texts) >= 10:  # Need sufficient data
            try:
                self.gan_generator.train_gan(narrative_texts, epochs=50, batch_size=min(8, len(narrative_texts)))
                self.gan_trained = True
                print("GAN pre-training completed successfully!")
            except Exception as e:
                print(f"GAN pre-training failed: {e}")
                self.gan_trained = False
        else:
            print("Insufficient data for GAN pre-training. Using basic generation.")

    def _setup_agent_network(self):
        """Setup agent network connections with improved topology"""
        if len(self.agents) < 2:
            return
        
        network_type = self.model_config.get('network_type', 'small_world')
        
        if network_type == 'small_world':
            self._create_small_world_network()
        elif network_type == 'scale_free':
            self._create_scale_free_network()
        else:
            self._create_random_network()

    def _create_small_world_network(self):
        """Create small-world network topology"""
        k = min(6, len(self.agents) // 2)  # Each agent connects to k neighbors
        p = 0.1  # Rewiring probability
        
        # Ring lattice
        for i, agent in enumerate(self.agents):
            connections = []
            for j in range(-k//2, k//2 + 1):
                if j != 0:
                    neighbor_idx = (i + j) % len(self.agents)
                    connections.append(self.agents[neighbor_idx])
            agent.connections = connections
        
        # Rewiring
        for agent in self.agents:
            for i, connection in enumerate(agent.connections):
                if random.random() < p:
                    new_connection = random.choice([a for a in self.agents if a != agent and a not in agent.connections])
                    agent.connections[i] = new_connection

    def _create_scale_free_network(self):
        """Create scale-free network using preferential attachment"""
        if len(self.agents) < 3:
            self._create_random_network()
            return
        
        # Start with a complete graph of 3 nodes
        for i in range(3):
            self.agents[i].connections = [self.agents[j] for j in range(3) if j != i]
        
        # Add remaining nodes with preferential attachment
        for i in range(3, len(self.agents)):
            new_agent = self.agents[i]
            degrees = [len(agent.connections) for agent in self.agents[:i]]
            total_degree = sum(degrees)
            
            if total_degree == 0:
                # Connect to random existing agent
                new_agent.connections = [random.choice(self.agents[:i])]
            else:
                # Preferential attachment - connect to 2 existing nodes
                probabilities = [d / total_degree for d in degrees]
                connections = np.random.choice(self.agents[:i], size=min(2, i), 
                                             replace=False, p=probabilities).tolist()
                new_agent.connections = connections
                
                # Update connections of selected nodes
                for connection in connections:
                    if new_agent not in connection.connections:
                        connection.connections.append(new_agent)

    def _create_random_network(self):
        """Create random network connections"""
        connection_count = min(5, len(self.agents) - 1)
        for agent in self.agents:
            possible_connections = [a for a in self.agents if a != agent]
            agent.connections = random.sample(possible_connections, 
                                            min(connection_count, len(possible_connections)))

    def _initialize_data_collection(self):
        """Initialize comprehensive data collection"""
        self.data = {'step': []}
        
        # Narrative tracking
        for nid in self.narratives:
            self.data[f'narrative_{nid}_believers'] = []
            self.data[f'narrative_{nid}_influence'] = []
        
        # Agent performance tracking
        self.data['avg_sentiment'] = []
        self.data['total_narratives'] = []
        self.data['counter_narratives_generated'] = []
        
        # RL specific metrics
        if self.use_rl:
            self.data['avg_agent_reward'] = []
            self.data['successful_spread_rate'] = []
            self.data['narrative_diversity'] = []
        
        # Network metrics
        self.data['network_density'] = []
        self.data['avg_clustering'] = []
        
        self.network_data = []

    def get_global_state_features(self):
        """Get global state features for RL agents"""
        features = np.zeros(3)
        
        if self.agents:
            # Global sentiment trend
            sentiments = [agent.sentiment for agent in self.agents]
            features[0] = np.mean(sentiments) if sentiments else 0
            
            # Narrative diversity (entropy)
            all_beliefs = {}
            for agent in self.agents:
                for nid, belief in agent.beliefs.items():
                    if belief > 0.5:  # Only count strong beliefs
                        all_beliefs[nid] = all_beliefs.get(nid, 0) + 1
            
            if all_beliefs:
                total_believers = sum(all_beliefs.values())
                entropy = -sum((count/total_believers) * np.log(count/total_believers + 1e-10) 
                              for count in all_beliefs.values())
                features[1] = entropy / np.log(len(all_beliefs) + 1)  # Normalized entropy
            
            # Network activity level
            recent_spreads = sum(agent.successful_spreads for agent in self.agents 
                               if hasattr(agent, 'successful_spreads'))
            features[2] = min(1.0, recent_spreads / (len(self.agents) * 5))  # Normalized activity
        
        return features

    def agent_requested_counter_narrative(self, requesting_agent, target_narrative_id):
        """Handle agent request for counter-narrative generation"""
        if not self.enable_counter_narratives:
            return
        
        if target_narrative_id not in self.narratives:
            return
        
        target_narrative = self.narratives[target_narrative_id]
        
        # Use GAN if available, otherwise use simple negation
        if self.gan_generator and self.gan_trained:
            try:
                counter_text = self.gan_generator.generate_counter_narrative(target_narrative['text'])
                counter_sentiment = -target_narrative['sentiment']
            except Exception as e:
                print(f"GAN counter-narrative generation failed: {e}")
                counter_text = f"Contrary to reports, {target_narrative['text'].lower()}"
                counter_sentiment = -target_narrative['sentiment']
        else:
            counter_text = f"No, {target_narrative['text'].lower().replace('is', 'is not')}"
            counter_sentiment = -target_narrative['sentiment']
        
        # Create counter-narrative
        counter_nid = max(self.narratives.keys()) + 1
        self.counter_narratives[counter_nid] = {
            'text': counter_text,
            'embedding': target_narrative['embedding'],  # Reuse embedding for simplicity
            'sentiment': counter_sentiment,
            'created_by': requesting_agent.unique_id,
            'targets': target_narrative_id,
            'creation_step': self._step_count
        }
        
        self.narratives[counter_nid] = self.counter_narratives[counter_nid]
        
        # Initialize tracking for new narrative
        self.data[f'narrative_{counter_nid}_believers'] = [0] * self._step_count
        self.data[f'narrative_{counter_nid}_influence'] = [0] * self._step_count
        
        # Seed in requesting agent
        requesting_agent.beliefs[counter_nid] = 1.0
        
        # Track narrative evolution
        self.narrative_evolution_history.append({
            'step': self._step_count,
            'action': 'counter_generated',
            'new_narrative_id': counter_nid,
            'target_narrative_id': target_narrative_id,
            'agent_id': requesting_agent.unique_id
        })

    def generate_adaptive_narrative(self, context="general"):
        """Generate contextually appropriate narrative using GAN"""
        if not self.gan_generator or not self.gan_trained:
            return None
        
        # Determine appropriate theme and sentiment based on current state
        if self.agents:
            avg_sentiment = np.mean([agent.sentiment for agent in self.agents])
            
            # Generate opposing narrative if population is too polarized
            if abs(avg_sentiment) > 0.6:
                target_sentiment = -np.sign(avg_sentiment) * 0.5
            else:
                target_sentiment = random.uniform(-0.8, 0.8)
        else:
            target_sentiment = 0.0
        
        try:
            if self.transformer_gan:
                narrative = self.transformer_gan.generate_advanced_narrative()
            else:
                narrative = self.gan_generator.generate_narrative(
                    theme=context, 
                    sentiment_target=target_sentiment
                )
            return narrative
        except Exception as e:
            print(f"Adaptive narrative generation failed: {e}")
            return None

    def advanced_events(self):
        """Advanced event system with RL and GAN integration"""
        if self._step_count % 10 == 0 and self.narratives:
            event_type = np.random.choice(["media_boost", "fact_check", "viral_spread", "generate_new"], 
                                        p=[0.25, 0.25, 0.25, 0.25])
            
            if event_type == "generate_new" and self.gan_generator:
                # Generate new narrative using GAN
                new_narrative = self.generate_adaptive_narrative()
                if new_narrative:
                    self._add_generated_narrative(new_narrative)
            
            elif event_type in ["media_boost", "fact_check", "viral_spread"]:
                self._execute_traditional_event(event_type)
            
            # Log event
            self.external_events.append({
                'step': self._step_count,
                'type': event_type,
                'impact': self._calculate_event_impact()
            })

    def _add_generated_narrative(self, text):
        """Add GAN-generated narrative to the model"""
        narrative_id = max(self.narratives.keys()) + 1 if self.narratives else 0
        
        # Process new narrative
        if hasattr(self, 'gan_generator') and self.gan_generator:
            sentiment = self.gan_generator.sentiment_analyzer.polarity_scores(text)['compound']
            embedding = self.gan_generator.sentence_model.encode(text)
        else:
            sentiment = 0.0
            embedding = np.random.randn(384)  # Default embedding size
        
        self.narratives[narrative_id] = {
            'text': text,
            'sentiment': sentiment,
            'embedding': embedding,
            'generated': True,
            'creation_step': self._step_count
        }
        
        # Initialize tracking
        self.data[f'narrative_{narrative_id}_believers'] = [0] * self._step_count
        self.data[f'narrative_{narrative_id}_influence'] = [0] * self._step_count
        
        # Seed in random agent
        if self.agents:
            seed_agent = random.choice(self.agents)
            seed_agent.beliefs[narrative_id] = 1.0
            
        print(f"Generated new narrative {narrative_id}: {text[:50]}...")

    def _execute_traditional_event(self, event_type):
        """Execute traditional event types"""
        target_nid = random.choice(list(self.narratives.keys()))
        affected_agents = random.sample(self.agents, min(20, len(self.agents)))
        
        if event_type == "media_boost":
            for agent in affected_agents:
                if target_nid not in agent.beliefs:
                    agent.beliefs[target_nid] = 0.0
                agent.beliefs[target_nid] = min(1.0, agent.beliefs[target_nid] + 0.4)
        
        elif event_type == "fact_check":
            for agent in affected_agents:
                if target_nid in agent.beliefs:
                    agent.beliefs[target_nid] = max(0.0, agent.beliefs[target_nid] - 0.5)
        
        elif event_type == "viral_spread":
            # Boost spread chances for this step
            for agent in affected_agents:
                if hasattr(agent, 'spread_chance'):
                    agent.spread_chance = min(1.0, agent.spread_chance * 1.5)

    def _calculate_event_impact(self):
        """Calculate the impact of the last event"""
        if len(self.data['avg_sentiment']) < 2:
            return 0
        
        current_sentiment = self.data['avg_sentiment'][-1]
        previous_sentiment = self.data['avg_sentiment'][-2]
        return abs(current_sentiment - previous_sentiment)

    def step(self):
        """Enhanced step function with RL and GAN integration"""
        # Execute agent steps
        for agent in self.agents:
            agent.step()
        
        self._step_count += 1
        
        # Generate counter-narratives (traditional system)
        if (self.enable_counter_narratives and 
            self._step_count % 5 == 0 and 
            self.narratives and 
            not self.use_rl):  # Only if not using RL (RL agents handle this)
            self._generate_traditional_counter_narrative()
        
        # Execute advanced events
        self.advanced_events()
        
        # Collect comprehensive data
        self._collect_step_data()
        
        # Update RL environment
        if self.rl_environment and self._step_count % 5 == 0:
            self._update_rl_environment()
        
        # Collect network data
        self._collect_network_data()
        
        # Update global trends
        self._update_global_trends()

    def _generate_traditional_counter_narrative(self):
        """Generate counter-narrative using traditional method"""
        dominant_nid = max(self.narratives, 
                          key=lambda x: sum(1 for a in self.agents 
                                          if x in a.beliefs and a.beliefs[x] > 0.5))
        
        counter_text = f"No, {self.narratives[dominant_nid]['text'].lower().replace('is', 'is not')}"
        
        if counter_text not in [n['text'] for n in self.narratives.values()]:
            counter_nid = max(self.narratives.keys()) + 1
            self.counter_narratives[counter_nid] = {
                'text': counter_text,
                'embedding': self.narratives[dominant_nid]['embedding'],
                'sentiment': -self.narratives[dominant_nid]['sentiment']
            }
            self.narratives[counter_nid] = self.counter_narratives[counter_nid]
            
            #