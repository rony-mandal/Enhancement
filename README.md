# Enhanced Narrative Spread Simulation with RL and GANs

## 🚀 Project Overview

This enhanced version of your DRDO ISSA Lab project integrates **Reinforcement Learning (RL)** and **Generative Adversarial Networks (GANs)** to create a sophisticated narrative spread simulation system. The project now features:

### 🧠 Reinforcement Learning Features
- **Intelligent Agents**: Agents learn optimal narrative spreading strategies using Actor-Critic methods
- **Adaptive Behavior**: Agents adapt their spreading patterns based on success/failure feedback
- **Policy Networks**: Neural networks guide agent decision-making
- **Performance Metrics**: Comprehensive tracking of agent learning and performance

### 🎭 GAN Features  
- **Narrative Generation**: Automatic generation of contextually appropriate narratives
- **Counter-Narrative Creation**: Smart generation of opposing viewpoints
- **Quality Assessment**: Evaluation metrics for generated content
- **Adaptive Content**: Context-aware narrative generation based on simulation state

### 🌐 Enhanced Simulation
- **Multiple Network Topologies**: Small-world, scale-free, and random networks
- **Advanced Event System**: Dynamic events with GAN-generated content
- **Comprehensive Analytics**: Enhanced visualization and reporting
- **Model Persistence**: Save/load complete simulation states

## 📁 Updated Project Structure

```
your_project/
├── data/
│   ├── climate_narratives.csv
│   ├── economic_narratives.csv
│   ├── election_narratives.csv
│   ├── health_narratives.csv
│   ├── psyops_narratives.csv
│   └── tech_narratives.csv
├── processing/
│   ├── __init__.py
│   ├── narrative_processor.py
│   └── gan_narrative_generator.py          # NEW
├── simulation/
│   ├── __init__.py
│   ├── agents.py                           # ENHANCED with RL
│   └── model.py                           # ENHANCED with GAN integration
├── app.py                                 # ENHANCED UI
├── requirements.txt                       # UPDATED dependencies
└── README.md
```

## 🔧 Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**New Dependencies Added:**
- `gymnasium` - RL environment framework
- `stable-baselines3[extra]` - RL algorithms
- `torch`, `torchvision`, `torchaudio` - Deep learning
- `transformers` - Transformer models for advanced text generation
- `wandb` - Experiment tracking (optional)

### 2. File Integration

Replace your existing files with the enhanced versions:

1. **Replace `requirements.txt`** with the enhanced requirements
2. **Add `processing/gan_narrative_generator.py`** - New GAN components
3. **Replace `simulation/agents.py`** with RL-enhanced agents
4. **Replace `simulation/model.py`** with enhanced model
5. **Replace `app.py`** with enhanced Streamlit interface

### 3. Verify Installation

```bash
python -c "import torch; import transformers; print('All dependencies installed successfully!')"
```

## 🎮 Usage Guide

### Running the Enhanced Simulation

```bash
streamlit run app.py
```

### Model Types Available

1. **Basic Model** (`basic`)
   - Traditional agent-based simulation
   - No RL or GAN components
   - Fastest execution

2. **RL-Only Model** (`rl_only`)
   - Agents use reinforcement learning
   - No narrative generation
   - Medium complexity

3. **GAN-Only Model** (`gan_only`)
   - Automatic narrative generation
   - No agent learning
   - Medium complexity

4. **Enhanced Model** (`enhanced`)
   - Full RL + GAN integration
   - Most sophisticated features
   - Highest complexity

### Key Features in Enhanced Interface

#### 🔧 Model Configuration Sidebar
- Select model type (basic/rl_only/gan_only/enhanced)
- Choose network topology (small-world/scale-free/random)
- Configure RL and GAN parameters

#### 📊 Enhanced Results Tabs
- **Narrative Dynamics**: Traditional believer tracking plus influence metrics
- **RL Performance**: Agent learning curves and performance metrics
- **GAN Analytics**: Generation quality, training progress, live demo
- **Network Analysis**: Advanced network metrics and evolution
- **System Metrics**: Comprehensive system performance data

#### 🎭 Live Generation Demo
- Generate narratives with specific themes and sentiments
- Real-time quality assessment
- Interactive parameter tuning

## 🧠 RL Components Explanation

### Agent Learning Process
1. **State Observation**: Agents observe their environment (sentiment, connections, beliefs)
2. **Action Selection**: Policy network chooses actions (spread aggressively, moderately, ignore, create counter)
3. **Reward Calculation**: Agents receive feedback based on influence changes
4. **Policy Update**: Networks are updated using Actor-Critic algorithm

### Key RL Metrics
- **Success Rate**: Percentage of successful spreading attempts
- **Average Reward**: Mean reward per step across all agents
- **Learning Curves**: Performance improvement over time

### Customizing RL Behavior
```python
# In agents.py, modify reward calculation
def calculate_reward(self, previous_influence, current_influence, action):
    base_reward = (current_influence - previous_influence) * 10
    # Add custom reward logic here
    return base_reward
```

## 🎭 GAN Components Explanation

### Narrative Generation Process
1. **Training Phase**: GAN learns from existing narrative datasets
2. **Generation Phase**: Creates new narratives based on context and sentiment targets
3. **Quality Assessment**: Evaluates generated content for coherence and relevance

### GAN Architecture
- **Generator**: LSTM-based sequence generator
- **Discriminator**: Bidirectional LSTM classifier
- **Advanced Option**: Transformer-based generation using GPT-2

### Customizing GAN Behavior
```python
# Generate context-specific narratives
narrative = model.gan_generator.generate_narrative(
    theme="climate",
    sentiment_target=0.7,
    max_length=50
)
```

## 📈 Performance Optimization

### For Large Simulations (>200 agents)
```python
# Use basic model for speed
model = create_narrative_model(
    num_agents=500, 
    narratives=narratives, 
    model_type='basic'
)
```

### For Advanced Analysis
```python
# Use enhanced model with custom configuration
model_config = {
    'network_type': 'scale_free',
    'learning_enabled': True,
    'gan_training_steps': 100
}

model = create_narrative_model(
    num_agents=100,
    narratives=narratives,
    model_type='enhanced',
    model_config=model_config
)
```

## 🔬 Research Applications

### Information Warfare Analysis
- Study how different narrative strategies compete
- Analyze the effectiveness of counter-narratives
- Examine network effects on information spread

### Social Media Simulation
- Model viral content spread patterns
- Test intervention strategies
- Analyze echo chamber formation

### Crisis Communication
- Simulate emergency information dissemination
- Test official vs. unofficial narrative competition
- Optimize communication strategies

## 📊 Data Export and Analysis

### Export Simulation Data
```python
# Get comprehensive data
df = model.get_enhanced_data_frame()
df.to_csv('simulation_results.csv')

# Get agent performance
performance = model.get_agent_performance_report()

# Get narrative evolution
evolution = model.get_narrative_evolution_report()
```

### Save/Load Model States
```python
# Save complete model state
model.save_enhanced_model('my_simulation.pkl')

# Load for continued analysis
model.load_enhanced_model('my_simulation.pkl')
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use CPU-only mode
   device = 'cpu'
   model = create_narrative_model(..., model_config={'device': 'cpu'})
   ```

2. **GAN Training Fails**
   - Reduce `gan_training_steps` to 20-30
   - Ensure sufficient narrative data (minimum 10 samples)
   - Check for duplicate narratives in dataset

3. **RL Agents Not Learning**
   - Increase simulation steps (minimum 50 for visible learning)
   - Adjust learning rates in agent configuration
   - Verify reward calculation is working

4. **Memory Usage High**
   - Reduce number of agents
   - Use 'basic' model type for large simulations
   - Clear model history periodically

### Performance Tips
- Start with small simulations (50 agents, 20 steps) to test setup
- Use 'rl_only' or 'gan_only' models to isolate issues
- Monitor system resources during long simulations

## 🎯 Next Steps for Further Enhancement

1. **Advanced RL Algorithms**: Implement PPO, A3C for better learning
2. **Transformer GANs**: Use GPT-3/4 for higher quality generation  
3. **Multi-Agent RL**: Cooperative and competitive agent strategies
4. **Real-time Data Integration**: Connect to social media APIs
5. **Visualization Enhancements**: 3D network views, animation

## 📚 Additional Resources

- [Mesa Documentation](https://mesa.readthedocs.io/)
- [Stable Baselines3 Guide](https://stable-baselines3.readthedocs.io/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 🤝 Contributing

This enhanced framework provides a solid foundation for advanced narrative simulation research. The modular design allows for easy extension and customization based on specific research needs.

---
*Enhanced for DRDO ISSA Lab - Combining traditional agent-based modeling with cutting-edge AI techniques*