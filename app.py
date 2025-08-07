import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
import numpy as np
from simulation.model import create_narrative_model, EnhancedNarrativeModel
from processing.narrative_processor import process_narratives, load_narrative_data, get_available_scenarios

def run_enhanced_dashboard():
    st.title("🧠 Enhanced Narrative Spread Simulation")
    st.markdown("*Advanced simulation with Reinforcement Learning and GAN-based narrative generation*")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        model_type = st.selectbox(
            "Model Type",
            options=['basic', 'rl_only', 'gan_only', 'enhanced'],
            index=3,
            help="Choose the type of model to use"
        )
        
        model_descriptions = {
            'basic': "Traditional agent-based model",
            'rl_only': "Agents use reinforcement learning",
            'gan_only': "GAN-based narrative generation",
            'enhanced': "Full RL + GAN integration"
        }
        
        st.info(f"**{model_type.title()}**: {model_descriptions[model_type]}")
        
        # Advanced options
        st.subheader("⚙️ Advanced Options")
        
        if model_type in ['rl_only', 'enhanced']:
            st.markdown("**RL Options:**")
            network_type = st.selectbox(
                "Network Topology",
                options=['small_world', 'scale_free', 'random'],
                index=0
            )
            
            learning_enabled = st.checkbox("Enable Agent Learning", value=True)
        else:
            network_type = 'random'
            learning_enabled = False
        
        if model_type in ['gan_only', 'enhanced']:
            st.markdown("**GAN Options:**")
            enable_narrative_generation = st.checkbox("Auto-Generate Narratives", value=True)
            gan_training_steps = st.slider("GAN Training Steps", 10, 200, 50)
        else:
            enable_narrative_generation = False
            gan_training_steps = 50
    
    # Main content
    # Data source selection
    data_source = st.radio("Data Source", ["Manual Input", "Preloaded Data"])
    
    if data_source == "Manual Input":
        narrative_input = st.text_area("Enter narratives (one per line):")
        narrative_texts = [text.strip() for text in narrative_input.split('\n') if text.strip()]
        narratives = process_narratives(narrative_texts) if narrative_texts else {}
    else:
        # Scenario selection dropdown
        available_scenarios = get_available_scenarios()
        if not available_scenarios:
            st.error("No narrative scenario files found in data/ directory!")
            return
        
        selected_scenario = st.selectbox(
            "Select Narrative Scenario",
            options=list(available_scenarios.keys()),
            index=0,
            help="Choose a predefined narrative scenario to simulate"
        )
        
        narratives = load_narrative_data(selected_scenario)
        
        if narratives:
            st.info(f"Loaded **{selected_scenario}** scenario with {len(narratives)} narratives")
            
            # Show loaded narratives in an expander
            with st.expander("📋 View Loaded Narratives"):
                for nid, narrative in narratives.items():
                    sentiment_emoji = "😟" if narrative['sentiment'] < -0.3 else ("😐" if narrative['sentiment'] < 0.3 else "😊")
                    st.write(f"{sentiment_emoji} **Narrative {nid}:** {narrative['text']} *(sentiment: {narrative['sentiment']:.2f})*")
    
    if not narratives:
        st.warning("Please enter at least one narrative or ensure preloaded data exists.")
        return
    
    # Simulation parameters
    st.subheader("⚙️ Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_agents = st.slider("Number of agents", 10, 500, 100)
    with col2:
        steps = st.slider("Simulation steps", 10, 200, 50)
    with col3:
        enable_counter_narratives = st.checkbox("Enable Counter-Narratives", value=True)
    
    # Model configuration
    model_config = {
        'network_type': network_type,
        'learning_enabled': learning_enabled,
        'gan_training_steps': gan_training_steps,
        'enable_narrative_generation': enable_narrative_generation
    }
    
    # Run simulation button
    if st.button("🚀 Run Enhanced Simulation", type="primary"):
        run_enhanced_simulation(
            narratives, num_agents, steps, model_type, 
            enable_counter_narratives, model_config
        )

def run_enhanced_simulation(narratives, num_agents, steps, model_type, 
                          enable_counter_narratives, model_config):
    """Run the enhanced simulation with chosen configuration"""
    
    st.subheader("🎯 Enhanced Simulation Results")
    
    # Initialize progress tracking
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.info(f"Initializing {model_type} model with {num_agents} agents...")
        init_progress = st.progress(0)
    
    try:
        # Create model
        model = create_narrative_model(
            num_agents, narratives, model_type=model_type,
            enable_counter_narratives=enable_counter_narratives,
            model_config=model_config
        )
        
        init_progress.progress(0.3)
        
        # Pre-training phase for GAN models
        if model_type in ['gan_only', 'enhanced'] and hasattr(model, 'gan_generator'):
            st.info("Pre-training GAN model... This may take a moment.")
            init_progress.progress(0.7)
        
        init_progress.progress(1.0)
        st.success("Model initialized successfully!")
        init_progress.empty()
        
        # Run simulation with progress bar
        simulation_progress = st.progress(0)
        status_text = st.empty()
        
        for step in range(steps):
            model.step()
            
            # Update progress
            progress = (step + 1) / steps
            simulation_progress.progress(progress)
            status_text.text(f"Step {step + 1}/{steps} - {progress*100:.1f}% complete")
            
            # Show intermediate results every 10 steps
            if (step + 1) % 10 == 0 and model_type == 'enhanced':
                with status_text:
                    df = model.get_enhanced_data_frame()
                    if len(df) > 0:
                        current_narratives = len([col for col in df.columns if 'narrative_' in col and '_believers' in col])
                        avg_sentiment = df['avg_sentiment'].iloc[-1] if 'avg_sentiment' in df.columns else 0
                        st.text(f"Step {step + 1}: {current_narratives} narratives, avg sentiment: {avg_sentiment:.2f}")
        
        simulation_progress.empty()
        status_text.empty()
        st.success("Simulation completed successfully!")
        
        # Display enhanced results
        display_enhanced_results(model, model_type)
        
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        st.exception(e)

def display_enhanced_results(model, model_type):
    """Display comprehensive results with enhanced visualizations"""
    
    # Get enhanced data
    df = model.get_enhanced_data_frame()
    if df.empty:
        st.error("No simulation data available")
        return
    
    # Create tabs for different result categories
    if model_type == 'enhanced':
        tabs = st.tabs([
            "📈 Narrative Dynamics", 
            "🧠 RL Performance", 
            "🎭 GAN Analytics",
            "🌐 Network Analysis", 
            "📊 System Metrics"
        ])
    elif model_type == 'rl_only':
        tabs = st.tabs([
            "📈 Narrative Dynamics", 
            "🧠 RL Performance", 
            "🌐 Network Analysis", 
            "📊 System Metrics"
        ])
    elif model_type == 'gan_only':
        tabs = st.tabs([
            "📈 Narrative Dynamics", 
            "🎭 GAN Analytics",
            "🌐 Network Analysis", 
            "📊 System Metrics"
        ])
    else:
        tabs = st.tabs([
            "📈 Narrative Dynamics", 
            "🌐 Network Analysis", 
            "📊 System Metrics"
        ])
    
    # Narrative Dynamics Tab
    with tabs[0]:
        display_narrative_dynamics(df, model)
    
    # RL Performance Tab (if applicable)
    if model_type in ['rl_only', 'enhanced'] and len(tabs) > 1:
        with tabs[1]:
            display_rl_performance(df, model)
    
    # GAN Analytics Tab (if applicable)
    gan_tab_idx = 2 if model_type == 'enhanced' else 1 if model_type == 'gan_only' else None
    if gan_tab_idx and gan_tab_idx < len(tabs):
        with tabs[gan_tab_idx]:
            display_gan_analytics(model)
    
    # Network Analysis Tab
    network_tab_idx = -2
    with tabs[network_tab_idx]:
        display_network_analysis(model)
    
    # System Metrics Tab
    with tabs[-1]:
        display_system_metrics(df, model)

def display_narrative_dynamics(df, model):
    """Display narrative spread dynamics"""
    st.subheader("📈 Narrative Spread Over Time")
    
    # Get believer columns
    believer_columns = [col for col in df.columns if 'narrative_' in col and '_believers' in col]
    influence_columns = [col for col in df.columns if 'narrative_' in col and '_influence' in col]
    
    if not believer_columns:
        st.warning("No narrative data available")
        return
    
    # Create subplot with believers and influence
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Number of Believers', 'Narrative Influence'),
        vertical_spacing=0.1
    )
    
    # Believers plot
    for col in believer_columns:
        narrative_id = col.split('_')[1]
        narrative_text = ""
        
        # Get narrative text
        if hasattr(model, 'narratives') and int(narrative_id) in model.narratives:
            narrative_text = model.narratives[int(narrative_id)]['text'][:30] + "..."
        
        fig.add_trace(
            go.Scatter(
                x=df['step'], y=df[col],
                mode='lines+markers',
                name=f"N{narrative_id}: {narrative_text}",
                line=dict(width=2)
            ),
            row=1, col=1
        )
    
    # Influence plot
    for col in influence_columns:
        if col in df.columns:
            narrative_id = col.split('_')[1]
            fig.add_trace(
                go.Scatter(
                    x=df['step'], y=df[col],
                    mode='lines',
                    name=f"Influence N{narrative_id}",
                    line=dict(dash='dot'),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    fig.update_layout(height=600, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Final results summary
    st.subheader("📊 Final Results Summary")
    
    if len(df) > 0:
        final_row = df.iloc[-1]
        narrative_results = []
        
        for col in believer_columns:
            narrative_id = col.split('_')[1]
            narrative_id_int = int(narrative_id)
            
            if hasattr(model, 'narratives') and narrative_id_int in model.narratives:
                narrative_text = model.narratives[narrative_id_int]['text']
                is_counter = narrative_id_int in getattr(model, 'counter_narratives', {})
                sentiment = model.narratives[narrative_id_int].get('sentiment', 0)
            else:
                narrative_text = f"Generated narrative {narrative_id}"
                is_counter = True
                sentiment = 0
            
            final_believers = final_row[col]
            final_influence = final_row.get(f'narrative_{narrative_id}_influence', 0)
            
            narrative_results.append({
                'id': narrative_id_int,
                'text': narrative_text,
                'believers': final_believers,
                'influence': final_influence,
                'sentiment': sentiment,
                'is_counter': is_counter
            })
        
        # Sort by influence
        narrative_results.sort(key=lambda x: x['influence'], reverse=True)
        
        # Display in columns
        cols = st.columns(2)
        for i, result in enumerate(narrative_results[:6]):  # Show top 6
            col = cols[i % 2]
            
            with col:
                sentiment_emoji = "😟" if result['sentiment'] < -0.3 else ("😐" if abs(result['sentiment']) <= 0.3 else "😊")
                type_emoji = "🔄" if result['is_counter'] else "📢"
                
                st.metric(
                    label=f"{type_emoji} {sentiment_emoji} Narrative {result['id']}",
                    value=f"{result['believers']} believers",
                    delta=f"Influence: {result['influence']:.1f}"
                )
                st.caption(f'"{result["text"][:60]}..."')

def display_rl_performance(df, model):
    """Display RL agent performance metrics"""
    st.subheader("🧠 Reinforcement Learning Performance")
    
    # RL metrics over time
    rl_columns = [col for col in df.columns if col in ['avg_agent_reward', 'successful_spread_rate']]
    
    if rl_columns:
        fig = make_subplots(
            rows=len(rl_columns), cols=1,
            subplot_titles=tuple(col.replace('_', ' ').title() for col in rl_columns),
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(rl_columns):
            fig.add_trace(
                go.Scatter(
                    x=df['step'], y=df[col],
                    mode='lines+markers',
                    name=col.replace('_', ' ').title(),
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(height=400 * len(rl_columns), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent performance report
    if hasattr(model, 'get_agent_performance_report'):
        performance_report = model.get_agent_performance_report()
        
        if performance_report:
            st.subheader("👥 Individual Agent Performance")
            
            # Convert to DataFrame for easier display
            perf_df = pd.DataFrame(performance_report).T
            
            # Display top performers
            if 'success_rate' in perf_df.columns:
                top_performers = perf_df.nlargest(5, 'success_rate')
                
                st.markdown("**Top Performing Agents:**")
                for idx, (agent_id, row) in enumerate(top_performers.iterrows()):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Agent ID", agent_id)
                    with col2:
                        st.metric("Type", row.get('type', 'Unknown'))
                    with col3:
                        st.metric("Success Rate", f"{row.get('success_rate', 0):.2%}")