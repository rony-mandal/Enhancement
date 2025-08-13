import mesa
import numpy as np
import pandas as pd
import torch
import random
import json
import os
from datetime import datetime
from collections import defaultdict
import networkx as nx

# Import our custom modules
from .rl_agent import AdaptiveNarrativeAgent
from .narrative_gan import NarrativeGAN, AdaptiveCounterNarrativeSystem

class EnhancedNarrativeModel(mesa.Model):
    """
    Enhanced narrative spread simulation with RL agents and GAN-generated content
    
    Features:
    - Adaptive RL agents that learn optimal strategies
    - GAN-generated narratives and counter-narratives
    - Advanced network dynamics
    - Real-time strategy adaptation
    - Comprehensive analytics
    """
    
    def __init__(self, num_agents=100, initial_narratives=None, enable_rl=True, 
                 enable_gan=True, enable_counter_narratives=True, scenario_type="War/Conflict"):
        super().__init__()
        
        # Core parameters
        self.num_agents = num_agents
        self.enable_rl = enable_rl
        self.enable_gan = enable_gan
        self.enable_counter_narratives = enable_counter_narratives
        self.scenario_type = scenario_type
        
        # Initialize narratives
        self.narratives = initial_narratives.copy() if initial_narratives else {}
        self.counter_narratives = {}
        self.narrative_id_counter = max(self.narratives.keys()) if self.narratives else 0
        
        # Initialize AI systems
        if self.enable_gan:
            self.narrative_gan = NarrativeGAN()
            self.counter_system = AdaptiveCounterNarrativeSystem(self.narrative_gan)
            self._prepare_gan_training()
        
        # Create agents with diverse types
        self._create_agent_population()
        
        # Initialize network structure
        self._initialize_network()
        
        # Data collection systems
        self._initialize_data_collection()
        
        # Simulation state
        self._step_count = 0
        self.phase = "initialization"  # phases: initialization, spreading, peak, resolution
        self.crisis_events = []
        
        # Advanced analytics
        self.influence_networks = []
        self.narrative_lifecycles = {}
        self.agent_performance_metrics = {}
        
        print(f"🚀 Enhanced simulation initialized:")
        print(f"   • Agents: {num_agents} ({len([a for a in self.agents if a.enable_learning])} with RL)")
        print(f"   • Narratives: {len(self.narratives)}")
        print(f"   • GAN enabled: {enable_gan}")
        print(f"   • Scenario: {scenario_type}")
    
    def _create_agent_population(self):
        """Create diverse agent population with realistic distributions"""
        
        # Agent type distribution (realistic for information warfare scenarios)
        type_distribution = {
            "Regular": 0.6,      # Most people are regular users
            "Skeptic": 0.15,     # Natural skeptics
            "Influencer": 0.1,   # Social media influencers
            "Bot": 0.1,          # Automated accounts
            "Counter_Agent": 0.05 # Defensive/counter-narrative agents
        }
        
        # Create agents based on distribution
        agent_types = []
        for agent_type, proportion in type_distribution.items():
            count = int(self.num_agents * proportion)
            agent_types.extend([agent_type] * count)
        
        # Add remaining agents as Regular
        while len(agent_types) < self.num_agents:
            agent_types.append("Regular")
        
        # Shuffle for randomness
        random.shuffle(agent_types)
        
        # Create agents
        for i, agent_type in enumerate(agent_types):
            agent = AdaptiveNarrativeAgent(
                model=self,
                agent_type=agent_type,
                enable_learning=self.enable_rl
            )
            
            # Assign initial beliefs for some agents
            if i < min(3, len(self.narratives)) and self.narratives:
                narrative_id = list(self.narratives.keys())[i % len(self.narratives)]
                agent.beliefs[narrative_id] = random.uniform(0.6, 1.0)
    
    def _initialize_network(self):
        """Create realistic social network structure"""
        agents_list = list(self.agents)
        
        # Create scale-free network (realistic for social media)
        G = nx.barabasi_albert_graph(len(agents_list), m=3)
        
        # Apply network to agents
        for agent_idx, agent in enumerate(agents_list):
            neighbors = list(G.neighbors(agent_idx))
            agent.connections = [agents_list[neighbor_idx] for neighbor_idx in neighbors]
        
        # Add some homophily (similar agents connect more)
        self._add_homophily_connections()
        
        # Add some random long-range connections
        self._add_random_connections()
    
    def _add_homophily_connections(self):
        """Add connections between similar agent types"""
        agents_by_type = defaultdict(list)
        for agent in self.agents:
            agents_by_type[agent.type].append(agent)
        
        # Add intra-type connections
        for agent_type, type_agents in agents_by_type.items():
            for i, agent in enumerate(type_agents):
                if len(agent.connections) < 8:  # Don't overcrowd
                    potential_connections = [a for a in type_agents[i+1:] 
                                           if a not in agent.connections and len(a.connections) < 8]
                    if potential_connections and random.random() < 0.3:
                        new_connection = random.choice(potential_connections)
                        agent.connections.append(new_connection)
                        new_connection.connections.append(agent)
    
    def _add_random_connections(self):
        """Add random long-range connections"""
        agents_list = list(self.agents)
        for agent in agents_list:
            if len(agent.connections) < 10 and random.random() < 0.1:
                potential = [a for a in agents_list if a != agent and a not in agent.connections]
                if potential:
                    new_connection = random.choice(potential)
                    agent.connections.append(new_connection)
                    new_connection.connections.append(agent)
    
    def _prepare_gan_training(self):
        """Prepare GAN with training data from narrative scenarios"""
        if not self.enable_gan:
            return
        
        # Get training narratives based on scenario
        training_narratives = self._get_training_narratives()
        
        if training_narratives:
            print("🤖 Training GAN on narrative data...")
            self.narrative_gan.train(training_narratives, epochs=50)
        
        # Try to initialize GPT-2 for advanced generation
        self.narrative_gan.initialize_pretrained()
    
    def _get_training_narratives(self):
        """Get training narratives for GAN based on scenario type"""
        training_data = {
            "War/Conflict": [
                "war is happening in the region",
                "peace negotiations have failed",
                "military forces are mobilizing",
                "civilians are being evacuated",
                "enemy attack is imminent",
                "our defenses are strong",
                "the situation is under control",
                "international aid is arriving"
            ],
            "Health Emergency": [
                "new virus outbreak spreading rapidly",
                "hospitals are overwhelmed with patients",
                "vaccine trials show promising results",
                "health officials confirm containment",
                "medical supplies are running low",
                "recovery rate is improving",
                "symptoms are being misreported",
                "treatment protocols are effective"
            ],
            "Economic Crisis": [
                "market crash threatens economy",
                "unemployment rates are rising",
                "government announces stimulus package",
                "banks are limiting withdrawals",
                "economic recovery is beginning",
                "inflation is spiraling out of control",
                "job market shows signs of improvement",
                "financial institutions remain stable"
            ]
        }
        
        # Get narratives for current scenario + some from current narratives
        scenario_narratives = training_data.get(self.scenario_type, [])
        current_narratives = [info['text'] for info in self.narratives.values()]
        
        return scenario_narratives + current_narratives
    
    def _initialize_data_collection(self):
        """Initialize comprehensive data collection systems"""
        self.data = {
            'step': [],
            'phase': [],
            'total_narratives': [],
            'avg_sentiment': [],
            'network_density': [],
            'information_entropy': [],
            'polarization_index': []
        }
        
        # Initialize narrative-specific tracking
        for nid in self.narratives:
            self.data[f'narrative_{nid}_believers'] = []
            self.data[f'narrative_{nid}_strength'] = []
        
        # Advanced metrics
        self.network_evolution = []
        self.agent_learning_curves = defaultdict(list)
        self.narrative_competition_data = []
        self.crisis_response_metrics = []
    
    def generate_crisis_event(self):
        """Generate realistic crisis events that affect narrative dynamics"""
        crisis_types = {
            "media_report": {
                "probability": 0.3,
                "impact": "moderate",
                "description": "Major media outlet publishes contradictory report"
            },
            "official_statement": {
                "probability": 0.2,
                "impact": "high",
                "description": "Government official makes public statement"
            },
            "social_media_viral": {
                "probability": 0.4,
                "impact": "high",
                "description": "Content goes viral on social media"
            },
            "expert_debunk": {
                "probability": 0.1,
                "impact": "moderate",
                "description": "Expert analysis debunks circulating narrative"
            }
        }
        
        # Select crisis type
        crisis_type = random.choices(
            list(crisis_types.keys()),
            weights=[info["probability"] for info in crisis_types.values()]
        )[0]
        
        crisis_info = crisis_types[crisis_type]
        
        # Generate or select target narrative
        if self.narratives:
            target_narrative_id = random.choice(list(self.narratives.keys()))
            
            # Apply crisis effects
            affected_agents = random.sample(
                list(self.agents), 
                min(30, len(self.agents))
            )
            
            effect_strength = 0.4 if crisis_info["impact"] == "moderate" else 0.6
            
            for agent in affected_agents:
                if target_narrative_id in agent.beliefs:
                    # Crisis can either boost or debunk
                    if random.random() < 0.6:  # Boost probability
                        agent.beliefs[target_narrative_id] = min(
                            1.0, agent.beliefs[target_narrative_id] + effect_strength
                        )
                    else:  # Debunk
                        agent.beliefs[target_narrative_id] = max(
                            0.0, agent.beliefs[target_narrative_id] - effect_strength
                        )
            
            # Record crisis event
            crisis_event = {
                'step': self._step_count,
                'type': crisis_type,
                'description': crisis_info["description"],
                'target_narrative': target_narrative_id,
                'affected_agents': len(affected_agents),
                'impact_level': crisis_info["impact"]
            }
            
            self.crisis_events.append(crisis_event)
            return crisis_event
        
        return None
    
    def adaptive_counter_narrative_generation(self):
        """Generate counter-narratives using AI system"""
        if not (self.enable_gan and self.enable_counter_narratives):
            return
        
        # Find dominant harmful narratives
        narrative_influence = {}
        for nid, narrative_info in self.narratives.items():
            if narrative_info.get('sentiment', 0) < -0.4:  # Negative narratives
                believers = sum(1 for agent in self.agents 
                              if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
                narrative_influence[nid] = believers / len(self.agents)
        
        # Generate counter for most influential harmful narrative
        if narrative_influence:
            dominant_nid = max(narrative_influence, key=narrative_influence.get)
            influence_score = narrative_influence[dominant_nid]
            
            if influence_score > 0.3:  # Threshold for intervention
                dominant_narrative = self.narratives[dominant_nid]
                
                # Prepare context for counter-generation
                simulation_context = {
                    'believer_count': sum(1 for a in self.agents if dominant_nid in a.beliefs and a.beliefs[dominant_nid] > 0.5),
                    'total_agents': len(self.agents),
                    'spread_rate': influence_score,
                    'step': self._step_count,
                    'scenario': self.scenario_type
                }
                
                # Generate adaptive counter-narrative
                counter_narrative = self.counter_system.generate_adaptive_counter(
                    dominant_narrative, simulation_context
                )
                
                # Add to simulation
                self.narrative_id_counter += 1
                counter_id = self.narrative_id_counter
                
                self.narratives[counter_id] = {
                    'text': counter_narrative['text'],
                    'sentiment': counter_narrative.get('sentiment', 0),
                    'embedding': np.random.randn(384),  # Placeholder
                    'is_counter': True,
                    'targets': [dominant_nid],
                    'generation_method': counter_narrative.get('method', 'GAN'),
                    'confidence': counter_narrative.get('confidence', 0.8)
                }
                
                # Initialize tracking for new counter-narrative
                self.data[f'narrative_{counter_id}_believers'] = [0] * self._step_count
                self.data[f'narrative_{counter_id}_strength'] = [0] * self._step_count
                
                # Seed counter-narrative in counter-agents and skeptics
                counter_agents = [a for a in self.agents if a.type in ["Counter_Agent", "Skeptic"]]
                if counter_agents:
                    seed_agent = random.choice(counter_agents)
                    seed_agent.beliefs[counter_id] = 0.9
                
                print(f"🔄 Generated counter-narrative: '{counter_narrative['text']}'")
                return counter_id
        
        return None
    
    def calculate_simulation_phase(self):
        """Determine current simulation phase based on narrative dynamics"""
        if self._step_count < 5:
            return "initialization"
        
        # Calculate metrics for phase determination
        total_beliefs = sum(len(agent.beliefs) for agent in self.agents)
        avg_beliefs_per_agent = total_beliefs / len(self.agents) if self.agents else 0
        
        # Calculate narrative competition
        narrative_strengths = []
        for nid in self.narratives:
            believers = sum(1 for agent in self.agents 
                          if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            narrative_strengths.append(believers / len(self.agents))
        
        max_penetration = max(narrative_strengths) if narrative_strengths else 0
        
        # Determine phase
        if max_penetration < 0.2:
            return "initialization"
        elif max_penetration < 0.6:
            return "spreading"
        elif max_penetration < 0.8:
            return "peak"
        else:
            return "resolution"
    
    def calculate_advanced_metrics(self):
        """Calculate advanced simulation metrics"""
        if not self.agents:
            return {}
        
        # Information entropy - measures narrative diversity
        narrative_distributions = defaultdict(int)
        for agent in self.agents:
            for nid, belief in agent.beliefs.items():
                if belief > 0.5:
                    narrative_distributions[nid] += 1
        
        total_believers = sum(narrative_distributions.values())
        if total_believers > 0:
            entropy = -sum((count/total_believers) * np.log2(count/total_believers + 1e-10) 
                          for count in narrative_distributions.values())
        else:
            entropy = 0
        
        # Polarization index - measures how divided the population is
        sentiment_scores = [agent.sentiment for agent in self.agents]
        polarization = np.std(sentiment_scores) if sentiment_scores else 0
        
        # Network density
        total_possible_connections = len(self.agents) * (len(self.agents) - 1) / 2
        actual_connections = sum(len(agent.connections) for agent in self.agents) / 2
        network_density = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        return {
            'information_entropy': entropy,
            'polarization_index': polarization,
            'network_density': network_density,
            'narrative_competition': len([n for n in narrative_distributions.values() if n > 5]),
            'avg_belief_strength': np.mean([np.mean(list(a.beliefs.values())) for a in self.agents if a.beliefs])
        }
    
    def update_agent_learning_curves(self):
        """Track learning progress of RL agents"""
        for agent in self.agents:
            if agent.enable_learning and hasattr(agent, 'reward_history'):
                recent_performance = np.mean(agent.reward_history[-10:]) if len(agent.reward_history) >= 10 else 0
                self.agent_learning_curves[f"{agent.type}_{agent.unique_id}"].append({
                    'step': self._step_count,
                    'avg_reward': recent_performance,
                    'exploration_rate': agent.epsilon,
                    'total_actions': sum(agent.actions_taken.values())
                })
    
    def generate_narrative_with_gan(self, context_hint=None, sentiment_target=None):
        """Generate new narrative using GAN based on current context"""
        if not self.enable_gan:
            return None
        
        # Determine generation parameters based on context
        if sentiment_target is None:
            # Analyze current sentiment landscape
            current_sentiments = [info.get('sentiment', 0) for info in self.narratives.values()]
            avg_sentiment = np.mean(current_sentiments) if current_sentiments else 0
            
            # Generate opposing sentiment with some probability
            if random.random() < 0.4:
                sentiment_target = -avg_sentiment * 0.8
            else:
                sentiment_target = random.uniform(-0.9, 0.9)
        
        # Extract keywords from existing narratives for context
        topic_keywords = []
        if self.narratives:
            for narrative_info in list(self.narratives.values())[:3]:
                words = narrative_info['text'].lower().split()
                topic_keywords.extend([w for w in words if len(w) > 3])
        
        # Generate new narrative
        generated = self.narrative_gan.generate_narrative(
            sentiment_target=sentiment_target,
            topic_keywords=topic_keywords[:5],
            context=context_hint or self.scenario_type
        )
        
        return generated
    
    def dynamic_narrative_injection(self):
        """Dynamically inject new narratives based on simulation state"""
        if not self.enable_gan or self._step_count % 15 != 0:  # Every 15 steps
            return
        
        # Analyze need for new narratives
        current_phase = self.calculate_simulation_phase()
        advanced_metrics = self.calculate_advanced_metrics()
        
        should_inject = False
        injection_reason = ""
        
        # Low entropy - need more narrative diversity
        if advanced_metrics.get('information_entropy', 0) < 1.0:
            should_inject = True
            injection_reason = "low_diversity"
        
        # High polarization - inject moderate narrative
        elif advanced_metrics.get('polarization_index', 0) > 0.8:
            should_inject = True
            injection_reason = "high_polarization"
        
        # Stagnant phase - inject disruptive narrative
        elif current_phase == "peak" and self._step_count % 30 == 0:
            should_inject = True
            injection_reason = "phase_disruption"
        
        if should_inject:
            # Generate context-appropriate narrative
            if injection_reason == "high_polarization":
                sentiment_target = 0.1  # Neutral/slightly positive
                context_hint = "stabilization"
            elif injection_reason == "low_diversity":
                sentiment_target = random.uniform(-0.8, 0.8)
                context_hint = "alternative_perspective"
            else:
                sentiment_target = random.uniform(-0.9, -0.5)  # Disruptive
                context_hint = "crisis_development"
            
            generated = self.generate_narrative_with_gan(context_hint, sentiment_target)
            
            if generated and generated['text']:
                # Add to simulation
                self.narrative_id_counter += 1
                new_id = self.narrative_id_counter
                
                self.narratives[new_id] = {
                    'text': generated['text'],
                    'sentiment': generated.get('sentiment', sentiment_target),
                    'embedding': np.random.randn(384),
                    'is_generated': True,
                    'generation_reason': injection_reason,
                    'generation_step': self._step_count,
                    'confidence': generated.get('confidence', 0.7)
                }
                
                # Initialize tracking
                self.data[f'narrative_{new_id}_believers'] = [0] * self._step_count
                self.data[f'narrative_{new_id}_strength'] = [0] * self._step_count
                
                # Seed in appropriate agents
                if injection_reason == "high_polarization":
                    # Seed in regular agents
                    target_agents = [a for a in self.agents if a.type == "Regular"]
                else:
                    # Seed in influencers or bots
                    target_agents = [a for a in self.agents if a.type in ["Influencer", "Bot"]]
                
                if target_agents:
                    seed_agent = random.choice(target_agents)
                    seed_agent.beliefs[new_id] = random.uniform(0.7, 0.9)
                
                print(f"💡 Injected narrative ({injection_reason}): '{generated['text']}'")
    
    def step(self):
        """Enhanced step function with all AI systems active"""
        # Update simulation phase
        self.phase = self.calculate_simulation_phase()
        
        # Agent actions (RL-driven or rule-based)
        self.agents.do("step")
        
        # Update step counter
        self._step_count += 1
        
        # AI-driven narrative generation
        if self._step_count % 8 == 0:  # Every 8 steps
            self.adaptive_counter_narrative_generation()
        
        # Dynamic narrative injection
        self.dynamic_narrative_injection()
        
        # Generate crisis events
        if self._step_count % 12 == 0 and random.random() < 0.3:
            crisis = self.generate_crisis_event()
            if crisis:
                print(f"⚠️ Crisis: {crisis['description']}")
        
        # Update target networks for RL agents (every 10 steps)
        if self._step_count % 10 == 0:
            for agent in self.agents:
                if agent.enable_learning:
                    agent.update_target_network()
        
        # Collect comprehensive data
        self.collect_step_data()
        
        # Update learning curves
        self.update_agent_learning_curves()
        
        # Network evolution tracking
        self.track_network_evolution()
    
    def collect_step_data(self):
        """Collect comprehensive data for this simulation step"""
        # Basic metrics
        step_data = {
            'step': self._step_count,
            'phase': self.phase,
            'total_narratives': len(self.narratives)
        }
        
        # Calculate advanced metrics
        advanced_metrics = self.calculate_advanced_metrics()
        step_data.update(advanced_metrics)
        
        # Average sentiment
        if self.agents:
            step_data['avg_sentiment'] = np.mean([agent.sentiment for agent in self.agents])
        else:
            step_data['avg_sentiment'] = 0.0
        
        # Narrative-specific metrics
        for nid in self.narratives:
            believers = sum(1 for agent in self.agents 
                          if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            avg_strength = np.mean([agent.beliefs[nid] for agent in self.agents 
                                   if nid in agent.beliefs]) if any(nid in a.beliefs for a in self.agents) else 0
            
            step_data[f'narrative_{nid}_believers'] = believers
            step_data[f'narrative_{nid}_strength'] = avg_strength
        
        # Append to data, ensuring consistency
        for key, value in step_data.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                # New metric - pad with zeros for previous steps
                self.data[key] = [0] * (self._step_count - 1) + [value]
        
        # Narrative competition tracking
        narrative_competition = {
            'step': self._step_count,
            'competing_narratives': []
        }
        
        for nid, narrative_info in self.narratives.items():
            believers = sum(1 for agent in self.agents 
                          if nid in agent.beliefs and agent.beliefs[nid] > 0.5)
            narrative_competition['competing_narratives'].append({
                'id': nid,
                'text': narrative_info['text'][:50] + "...",
                'believers': believers,
                'penetration': believers / len(self.agents),
                'sentiment': narrative_info.get('sentiment', 0),
                'is_counter': narrative_info.get('is_counter', False),
                'is_generated': narrative_info.get('is_generated', False)
            })
        
        self.narrative_competition_data.append(narrative_competition)
    
    def track_network_evolution(self):
        """Track how the network structure evolves over time"""
        # Create network snapshot
        G = nx.Graph()
        for agent in self.agents:
            G.add_node(agent.unique_id, type=agent.type, sentiment=agent.sentiment)
        
        for agent in self.agents:
            for neighbor in agent.connections:
                G.add_edge(agent.unique_id, neighbor.unique_id)
        
        # Calculate network metrics
        if len(G.nodes()) > 0:
            network_metrics = {
                'step': self._step_count,
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': nx.density(G),
                'clustering': nx.average_clustering(G) if len(G.nodes()) > 2 else 0,
                'components': nx.number_connected_components(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else -1
            }
            
            # Agent type connectivity
            type_connectivity = defaultdict(list)
            for node in G.nodes():
                agent = next(a for a in self.agents if a.unique_id == node)
                type_connectivity[agent.type].append(G.degree(node))
            
            for agent_type, degrees in type_connectivity.items():
                network_metrics[f'avg_degree_{agent_type}'] = np.mean(degrees)
            
            self.network_evolution.append(network_metrics)
    
    def get_comprehensive_data(self):
        """Get all collected data as structured format"""
        main_df = self.get_data_frame()
        
        # Additional data structures
        network_df = pd.DataFrame(self.network_evolution) if self.network_evolution else pd.DataFrame()
        competition_df = pd.DataFrame([
            {
                'step': comp['step'],
                'narrative_id': narr['id'],
                'narrative_text': narr['text'],
                'believers': narr['believers'],
                'penetration': narr['penetration'],
                'sentiment': narr['sentiment'],
                'is_counter': narr['is_counter'],
                'is_generated': narr['is_generated']
            }
            for comp in self.narrative_competition_data
            for narr in comp['competing_narratives']
        ]) if self.narrative_competition_data else pd.DataFrame()
        
        crisis_df = pd.DataFrame(self.crisis_events) if self.crisis_events else pd.DataFrame()
        
        # Agent performance data
        agent_performance_data = []
        for agent in self.agents:
            if hasattr(agent, 'get_performance_metrics'):
                metrics = agent.get_performance_metrics()
                agent_performance_data.append(metrics)
        
        agent_performance_df = pd.DataFrame(agent_performance_data) if agent_performance_data else pd.DataFrame()
        
        return {
            'main_simulation': main_df,
            'network_evolution': network_df,
            'narrative_competition': competition_df,
            'crisis_events': crisis_df,
            'agent_performance': agent_performance_df,
            'learning_curves': dict(self.agent_learning_curves),
            'model_summary': self.get_model_summary()
        }
    
    def get_model_summary(self):
        """Get summary statistics of the simulation"""
        if not self.agents:
            return {}
        
        # Count agent types
        agent_type_counts = defaultdict(int)
        rl_agent_count = 0
        for agent in self.agents:
            agent_type_counts[agent.type] += 1
            if agent.enable_learning:
                rl_agent_count += 1
        
        # Narrative statistics
        generated_narratives = sum(1 for n in self.narratives.values() if n.get('is_generated', False))
        counter_narratives = sum(1 for n in self.narratives.values() if n.get('is_counter', False))
        
        # Performance statistics
        avg_agent_beliefs = np.mean([len(agent.beliefs) for agent in self.agents])
        avg_connections = np.mean([len(agent.connections) for agent in self.agents])
        
        return {
            'simulation_steps': self._step_count,
            'total_agents': len(self.agents),
            'rl_agents': rl_agent_count,
            'agent_types': dict(agent_type_counts),
            'total_narratives': len(self.narratives),
            'generated_narratives': generated_narratives,
            'counter_narratives': counter_narratives,
            'crisis_events': len(self.crisis_events),
            'avg_agent_beliefs': avg_agent_beliefs,
            'avg_connections': avg_connections,
            'final_phase': self.phase,
            'ai_features': {
                'reinforcement_learning': self.enable_rl,
                'gan_generation': self.enable_gan,
                'counter_narratives': self.enable_counter_narratives
            }
        }
    
    def save_simulation_state(self, filepath):
        """Save complete simulation state for later analysis"""
        simulation_state = {
            'model_summary': self.get_model_summary(),
            'comprehensive_data': self.get_comprehensive_data(),
            'narratives': self.narratives,
            'agent_states': [agent.get_performance_metrics() for agent in self.agents if hasattr(agent, 'get_performance_metrics')],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(simulation_state, f, indent=2, default=str)
        
        print(f"💾 Simulation state saved to {filepath}")
        
        # Save RL models if enabled
        if self.enable_rl:
            model_dir = os.path.dirname(filepath)
            for i, agent in enumerate(self.agents):
                if agent.enable_learning:
                    model_path = os.path.join(model_dir, f"agent_{agent.unique_id}_model.pth")
                    agent.save_model(model_path)
        
        # Save GAN model if enabled
        if self.enable_gan:
            gan_path = filepath.replace('.json', '_gan.pth')
            self.narrative_gan.save_model(gan_path)
    
    def get_data_frame(self):
        """Enhanced version of original method with additional error handling"""
        if not self.data['step']:
            return pd.DataFrame()
        
        # Ensure all arrays have consistent length
        df_data = {}
        max_length = len(self.data['step'])
        
        for key, values in self.data.items():
            if isinstance(values, list) and key != 'event_log':
                current_length = len(values)
                if current_length < max_length:
                    # Pad with appropriate default values
                    if 'believers' in key or 'strength' in key:
                        padded_values = values + [0] * (max_length - current_length)
                    else:
                        padded_values = values + [values[-1] if values else 0] * (max_length - current_length)
                    df_data[key] = padded_values
                else:
                    df_data[key] = values[:max_length]
        
        return pd.DataFrame(df_data)