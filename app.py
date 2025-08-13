import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import our enhanced modules
from simulation.model import EnhancedNarrativeModel
from simulation.rl_agent import AdaptiveNarrativeAgent
from processing.narrative_processor import process_narratives, load_narrative_data, get_available_scenarios

def main():
    # Configure page
    st.set_page_config(
        page_title="🤖 Advanced AI Narrative Warfare Simulation",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .crisis-alert {
        background-color: #ffe8e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Advanced AI-Driven Information Warfare Simulation")
    st.markdown("""
    **Cutting-edge simulation combining Reinforcement Learning, GANs, and Agent-Based Modeling**
    
    🧠 **RL Agents**: Adaptive decision-making with Deep Q-Networks  
    🎭 **GAN System**: Dynamic narrative generation and counter-narratives  
    🕸️ **Network Dynamics**: Realistic social network evolution  
    📊 **Advanced Analytics**: Multi-dimensional performance tracking
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Simulation Configuration")
        
        # Data source selection
        data_source = st.radio("📊 Data Source", ["Manual Input", "Preloaded Scenarios"])
        
        if data_source == "Manual Input":
            narrative_input = st.text_area(
                "✍️ Enter narratives (one per line):",
                placeholder="e.g., 'War is escalating in the region'\n'Peace talks are progressing'"
            )
            narrative_texts = [text.strip() for text in narrative_input.split('\n') if text.strip()]
            narratives = process_narratives(narrative_texts) if narrative_texts else {}
            scenario_type = "Custom"
        else:
            available_scenarios = get_available_scenarios()
            if not available_scenarios:
                st.error("❌ No scenario files found in data/ directory!")
                st.stop()
            
            scenario_type = st.selectbox(
                "🎯 Select Scenario",
                options=list(available_scenarios.keys()),
                help="Choose a predefined information warfare scenario"
            )
            
            narratives = load_narrative_data(scenario_type)
            
            if narratives:
                with st.expander("📋 View Loaded Narratives"):
                    for nid, narrative in narratives.items():
                        sentiment_color = "🔴" if narrative['sentiment'] < -0.3 else ("🟡" if narrative['sentiment'] < 0.3 else "🟢")
                        st.write(f"{sentiment_color} **{narrative['text']}** *(sentiment: {narrative['sentiment']:.2f})*")
        
        st.divider()
        
        # AI Configuration
        st.subheader("🤖 AI Configuration")
        
        enable_rl = st.checkbox(
            "🧠 Enable Reinforcement Learning",
            value=True,
            help="Agents learn optimal strategies using Deep Q-Networks"
        )
        
        enable_gan = st.checkbox(
            "🎭 Enable GAN Generation",
            value=True,
            help="Generate realistic narratives and counter-narratives using GANs"
        )
        
        enable_counter = st.checkbox(
            "🔄 Enable Adaptive Counter-Narratives",
            value=True,
            help="AI system generates targeted counter-narratives against harmful content"
        )
        
        st.divider()
        
        # Simulation Parameters
        st.subheader("⚙️ Simulation Parameters")
        
        num_agents = st.slider("👥 Number of Agents", 50, 500, 150, step=25)
        max_steps = st.slider("📈 Simulation Steps", 10, 100, 40, step=5)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_real_time = st.checkbox("📊 Real-time Updates", value=False, help="Update visualization every few steps")
            save_state = st.checkbox("💾 Save Simulation State", value=True, help="Save complete simulation for later analysis")
            detailed_logging = st.checkbox("📝 Detailed Logging", value=True, help="Show detailed step-by-step information")
    
    # Validation
    if not narratives:
        st.warning("⚠️ Please provide at least one narrative to start the simulation.")
        st.stop()
    
    # Information panel
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Narratives", len(narratives))
    with col2:
        st.metric("👥 Agents", num_agents)
    with col3:
        st.metric("🎯 Scenario", scenario_type)
    with col4:
        ai_features = sum([enable_rl, enable_gan, enable_counter])
        st.metric("🤖 AI Features", f"{ai_features}/3")
    
    # Run simulation button
    if st.button("🚀 Launch Advanced Simulation", type="primary", use_container_width=True):
        run_advanced_simulation(
            narratives, num_agents, max_steps, scenario_type,
            enable_rl, enable_gan, enable_counter,
            show_real_time, save_state, detailed_logging
        )

def run_advanced_simulation(narratives, num_agents, max_steps, scenario_type,
                           enable_rl, enable_gan, enable_counter,
                           show_real_time, save_state, detailed_logging):
    """Run the advanced simulation with all AI features"""
    
    # Initialize simulation
    with st.spinner("🔄 Initializing Advanced AI Systems..."):
        model = EnhancedNarrativeModel(
            num_agents=num_agents,
            initial_narratives=narratives,
            enable_rl=enable_rl,
            enable_gan=enable_gan,
            enable_counter_narratives=enable_counter,
            scenario_type=scenario_type
        )
    
    st.success("✅ Advanced simulation initialized successfully!")
    
    # Create real-time containers
    if show_real_time:
        progress_container = st.container()
        metrics_container = st.container()
        chart_container = st.container()
    else:
        progress_container = st.container()
    
    # Progress tracking
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    # Simulation execution
    simulation_log = []
    
    for step in range(max_steps):
        # Update progress
        progress = (step + 1) / max_steps
        progress_bar.progress(progress)
        status_text.text(f"🔄 Step {step + 1}/{max_steps} - Phase: {model.phase}")
        
        # Execute simulation step
        model.step()
        
        # Log important events
        if detailed_logging and hasattr(model, 'crisis_events') and model.crisis_events:
            last_crisis = model.crisis_events[-1]
            if last_crisis['step'] == step + 1:
                simulation_log.append(f"⚠️ Step {step + 1}: {last_crisis['description']}")
        
        # Real-time updates
        if show_real_time and (step + 1) % 5 == 0:
            update_real_time_display(model, metrics_container, chart_container)
        
        # Brief pause for visual effect
        if show_real_time:
            import time
            time.sleep(0.1)
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Get comprehensive results
    comprehensive_data = model.get_comprehensive_data()
    model_summary = comprehensive_data['model_summary']
    
    # Display results
    display_simulation_results(model, comprehensive_data, simulation_log)
    
    # Save simulation state
    if save_state:
        save_simulation_results(model, comprehensive_data, scenario_type)

def update_real_time_display(model, metrics_container, chart_container):
    """Update real-time visualization during simulation"""
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate current metrics
        total_beliefs = sum(len(agent.beliefs) for agent in model.agents)
        avg_sentiment = np.mean([agent.sentiment for agent in model.agents])
        active_narratives = len([n for n in model.narratives.values() if any(
            nid in agent.beliefs and agent.beliefs[nid] > 0.5 
            for agent in model.agents for nid in [list(model.narratives.keys())[list(model.narratives.values()).index(n)]]
        )])
        
        with col1:
            st.metric("Total Beliefs", total_beliefs)
        with col2:
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        with col3:
            st.metric("Active Narratives", active_narratives)
        with col4:
            st.metric("Current Phase", model.phase)
    
    # Quick chart update
    with chart_container:
        if model.data['step']:
            df = model.get_data_frame()
            fig = px.line(df, x='step', y='avg_sentiment', 
                         title='Sentiment Evolution (Real-time)')
            st.plotly_chart(fig, use_container_width=True, key=f"realtime_{model._step_count}")

def display_simulation_results(model, comprehensive_data, simulation_log):
    """Display comprehensive simulation results"""
    
    st.header("📊 Simulation Results")
    
    # Summary statistics
    summary = comprehensive_data['model_summary']
    
    st.markdown("### 🎯 Executive Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="highlight-box">
        <h4>🔍 Simulation Overview</h4>
        <ul>
        <li><strong>Total Steps:</strong> {summary['simulation_steps']}</li>
        <li><strong>Final Phase:</strong> {summary['final_phase']}</li>
        <li><strong>Crisis Events:</strong> {summary['crisis_events']}</li>
        <li><strong>Generated Narratives:</strong> {summary['generated_narratives']}</li>
        <li><strong>Counter-Narratives:</strong> {summary['counter_narratives']}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="highlight-box">
        <h4>👥 Agent Performance</h4>
        <ul>
        <li><strong>Total Agents:</strong> {summary['total_agents']}</li>
        <li><strong>RL-Enabled:</strong> {summary['rl_agents']}</li>
        <li><strong>Avg Beliefs/Agent:</strong> {summary['avg_agent_beliefs']:.1f}</li>
        <li><strong>Avg Connections:</strong> {summary['avg_connections']:.1f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabbed interface for detailed results
    tabs = st.tabs([
        "📈 Narrative Evolution", 
        "🧠 Agent Intelligence", 
        "🕸️ Network Dynamics", 
        "⚠️ Crisis Response", 
        "🎭 AI Generation",
        "📊 Advanced Analytics"
    ])
    
    # Tab 1: Narrative Evolution
    with tabs[0]:
        display_narrative_evolution(comprehensive_data)
    
    # Tab 2: Agent Intelligence
    with tabs[1]:
        display_agent_intelligence(comprehensive_data)
    
    # Tab 3: Network Dynamics
    with tabs[2]:
        display_network_dynamics(comprehensive_data)
    
    # Tab 4: Crisis Response
    with tabs[3]:
        display_crisis_analysis(comprehensive_data, simulation_log)
    
    # Tab 5: AI Generation
    with tabs[4]:
        display_ai_generation_analysis(model, comprehensive_data)
    
    # Tab 6: Advanced Analytics
    with tabs[5]:
        display_advanced_analytics(comprehensive_data)

def display_narrative_evolution(data):
    """Display narrative spread and competition analysis"""
    main_df = data['main_simulation']
    competition_df = data['narrative_competition']
    
    if main_df.empty:
        st.warning("No data available for narrative evolution.")
        return
    
    # Narrative believers over time
    st.subheader("📈 Narrative Spread Over Time")
    
    believer_columns = [col for col in main_df.columns if 'believers' in col and 'narrative_' in col]
    
    if believer_columns:
        fig = go.Figure()
        
        for col in believer_columns:
            narrative_id = col.split('_')[1]
            fig.add_trace(go.Scatter(
                x=main_df['step'],
                y=main_df[col],
                mode='lines+markers',
                name=f'Narrative {narrative_id}',
                line=dict(width=2),
                hovertemplate='<b>Narrative %{fullData.name}</b><br>Step: %{x}<br>Believers: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Number of Believers for Each Narrative",
            xaxis_title="Simulation Step",
            yaxis_title="Number of Believers",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Narrative competition heatmap
    if not competition_df.empty:
        st.subheader("🔥 Narrative Competition Matrix")
        
        # Create penetration heatmap
        pivot_data = competition_df.pivot_table(
            index='narrative_id', 
            columns='step', 
            values='penetration',
            fill_value=0
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=[f"Narrative {idx}" for idx in pivot_data.index],
            colorscale='Viridis',
            hoveronpoints=False,
            hovertemplate='Step: %{x}<br>Narrative: %{y}<br>Penetration: %{z:.2%}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title="Narrative Penetration Heatmap",
            xaxis_title="Simulation Step",
            yaxis_title="Narratives",
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Phase evolution
    if 'phase' in main_df.columns:
        st.subheader("🔄 Simulation Phases")
        
        phase_changes = []
        current_phase = None
        
        for idx, row in main_df.iterrows():
            if row['phase'] != current_phase:
                phase_changes.append({
                    'step': row['step'],
                    'phase': row['phase']
                })
                current_phase = row['phase']
        
        if len(phase_changes) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Phase timeline
                fig_phase = go.Figure()
                
                for i, phase_change in enumerate(phase_changes):
                    next_step = phase_changes[i+1]['step'] if i+1 < len(phase_changes) else main_df['step'].max()
                    
                    fig_phase.add_shape(
                        type="rect",
                        x0=phase_change['step'], x1=next_step,
                        y0=0, y1=1,
                        fillcolor=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                        opacity=0.3,
                        line_width=0
                    )
                    
                    fig_phase.add_annotation(
                        x=(phase_change['step'] + next_step) / 2,
                        y=0.5,
                        text=phase_change['phase'],
                        showarrow=False,
                        font=dict(size=12, color="black")
                    )
                
                fig_phase.update_layout(
                    title="Simulation Phase Timeline",
                    xaxis_title="Simulation Step",
                    yaxis=dict(showticklabels=False, range=[0, 1]),
                    height=200,
                    showlegend=False
                )
                
                st.plotly_chart(fig_phase, use_container_width=True)
            
            with col2:
                # Phase summary
                st.markdown("**Phase Breakdown:**")
                for phase_change in phase_changes:
                    st.write(f"• **{phase_change['phase'].title()}**: Started at step {phase_change['step']}")

def display_agent_intelligence(data):
    """Display RL agent learning and performance metrics"""
    agent_df = data['agent_performance']
    learning_curves = data['learning_curves']
    
    if agent_df.empty:
        st.warning("No agent performance data available.")
        return
    
    st.subheader("🧠 Reinforcement Learning Performance")
    
    # Agent type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'agent_type' in agent_df.columns:
            type_counts = agent_df['agent_type'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Agent Type Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Performance metrics by type
        if 'average_reward' in agent_df.columns and 'agent_type' in agent_df.columns:
            avg_rewards = agent_df.groupby('agent_type')['average_reward'].mean().sort_values(ascending=True)
            
            fig_bar = px.bar(
                x=avg_rewards.values,
                y=avg_rewards.index,
                orientation='h',
                title="Average Reward by Agent Type",
                labels={'x': 'Average Reward', 'y': 'Agent Type'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Learning curves
    if learning_curves:
        st.subheader("📚 Learning Curves")
        
        # Select representative agents for display
        sample_agents = list(learning_curves.keys())[:5]  # Show first 5 for clarity
        
        if sample_agents:
            fig_learning = go.Figure()
            
            for agent_id in sample_agents:
                agent_data = learning_curves[agent_id]
                if agent_data:
                    steps = [point['step'] for point in agent_data]
                    rewards = [point['avg_reward'] for point in agent_data]
                    
                    fig_learning.add_trace(go.Scatter(
                        x=steps,
                        y=rewards,
                        mode='lines',
                        name=f'Agent {agent_id.split("_")[-1]}',
                        hovertemplate='<b>%{fullData.name}</b><br>Step: %{x}<br>Avg Reward: %{y:.3f}<extra></extra>'
                    ))
            
            fig_learning.update_layout(
                title="Agent Learning Progress (Sample)",
                xaxis_title="Simulation Step",
                yaxis_title="Average Reward",
                height=400
            )
            
            st.plotly_chart(fig_learning, use_container_width=True)
    
    # Action analysis
    if 'actions_taken' in agent_df.columns:
        st.subheader("⚡ Action Analysis")
        
        # Parse actions taken (assuming it's stored as string representation of dict)
        all_actions = {}
        for idx, row in agent_df.iterrows():
            try:
                if isinstance(row['actions_taken'], str):
                    actions = eval(row['actions_taken'])  # Note: In production, use json.loads
                elif isinstance(row['actions_taken'], dict):
                    actions = row['actions_taken']
                else:
                    continue
                
                for action, count in actions.items():
                    all_actions[action] = all_actions.get(action, 0) + count
            except:
                continue
        
        if all_actions:
            action_df = pd.DataFrame(list(all_actions.items()), columns=['Action', 'Count'])
            
            fig_actions = px.bar(
                action_df,
                x='Action',
                y='Count',
                title="Total Actions Taken by All RL Agents",
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_actions.update_xaxes(tickangle=45)
            st.plotly_chart(fig_actions, use_container_width=True)

def display_network_dynamics(data):
    """Display network evolution and structure analysis"""
    network_df = data['network_evolution']
    
    if network_df.empty:
        st.warning("No network evolution data available.")
        return
    
    st.subheader("🕸️ Network Structure Evolution")
    
    # Network metrics over time
    col1, col2 = st.columns(2)
    
    with col1:
        if 'density' in network_df.columns:
            fig_density = px.line(
                network_df,
                x='step',
                y='density',
                title="Network Density Over Time",
                labels={'density': 'Network Density', 'step': 'Simulation Step'}
            )
            st.plotly_chart(fig_density, use_container_width=True)
    
    with col2:
        if 'clustering' in network_df.columns:
            fig_clustering = px.line(
                network_df,
                x='step',
                y='clustering',
                title="Average Clustering Coefficient",
                labels={'clustering': 'Clustering Coefficient', 'step': 'Simulation Step'}
            )
            st.plotly_chart(fig_clustering, use_container_width=True)
    
    # Agent type connectivity
    st.subheader("🔗 Connectivity by Agent Type")
    
    agent_type_columns = [col for col in network_df.columns if 'avg_degree_' in col]
    
    if agent_type_columns:
        connectivity_data = []
        for col in agent_type_columns:
            agent_type = col.replace('avg_degree_', '')
            for idx, row in network_df.iterrows():
                connectivity_data.append({
                    'step': row['step'],
                    'agent_type': agent_type,
                    'avg_degree': row[col]
                })
        
        connectivity_df = pd.DataFrame(connectivity_data)
        
        fig_connectivity = px.line(
            connectivity_df,
            x='step',
            y='avg_degree',
            color='agent_type',
            title="Average Connections by Agent Type",
            labels={'avg_degree': 'Average Connections', 'step': 'Simulation Step'}
        )
        st.plotly_chart(fig_connectivity, use_container_width=True)
    
    # Network health indicators
    if all(col in network_df.columns for col in ['components', 'diameter']):
        st.subheader("💪 Network Health Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Connected components
            fig_components = px.line(
                network_df,
                x='step',
                y='components',
                title="Number of Connected Components",
                labels={'components': 'Connected Components', 'step': 'Simulation Step'}
            )
            fig_components.add_hline(y=1, line_dash="dash", line_color="red", 
                                   annotation_text="Fully Connected")
            st.plotly_chart(fig_components, use_container_width=True)
        
        with col2:
            # Network diameter (when connected)
            connected_data = network_df[network_df['diameter'] > 0]
            if not connected_data.empty:
                fig_diameter = px.line(
                    connected_data,
                    x='step',
                    y='diameter',
                    title="Network Diameter (When Connected)",
                    labels={'diameter': 'Network Diameter', 'step': 'Simulation Step'}
                )
                st.plotly_chart(fig_diameter, use_container_width=True)

def display_crisis_analysis(data, simulation_log):
    """Display crisis events and their impact analysis"""
    crisis_df = data['crisis_events']
    
    st.subheader("⚠️ Crisis Events Analysis")
    
    if crisis_df.empty:
        st.info("No crisis events occurred during this simulation.")
        return
    
    # Crisis timeline
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Crisis events over time
        fig_crisis = go.Figure()
        
        crisis_types = crisis_df['type'].unique()
        colors = px.colors.qualitative.Set1[:len(crisis_types)]
        
        for i, crisis_type in enumerate(crisis_types):
            type_data = crisis_df[crisis_df['type'] == crisis_type]
            
            fig_crisis.add_trace(go.Scatter(
                x=type_data['step'],
                y=[crisis_type] * len(type_data),
                mode='markers',
                marker=dict(size=type_data['affected_agents'] * 0.5, color=colors[i]),
                name=crisis_type,
                hovertemplate='<b>%{y}</b><br>Step: %{x}<br>Affected: %{text} agents<extra></extra>',
                text=type_data['affected_agents']
            ))
        
        fig_crisis.update_layout(
            title="Crisis Events Timeline",
            xaxis_title="Simulation Step",
            yaxis_title="Crisis Type",
            height=400
        )
        
        st.plotly_chart(fig_crisis, use_container_width=True)
    
    with col2:
        # Crisis summary statistics
        st.markdown("**Crisis Summary:**")
        st.metric("Total Events", len(crisis_df))
        
        if 'impact_level' in crisis_df.columns:
            impact_counts = crisis_df['impact_level'].value_counts()
            st.markdown("**Impact Distribution:**")
            for impact, count in impact_counts.items():
                st.write(f"• {impact.title()}: {count}")
        
        if 'affected_agents' in crisis_df.columns:
            avg_affected = crisis_df['affected_agents'].mean()
            st.metric("Avg Agents Affected", f"{avg_affected:.1f}")
    
    # Detailed crisis log
    if simulation_log:
        st.subheader("📝 Detailed Event Log")
        
        with st.expander("View Complete Event Log"):
            for log_entry in simulation_log:
                st.markdown(f"<div class='crisis-alert'>{log_entry}</div>", unsafe_allow_html=True)

def display_ai_generation_analysis(model, data):
    """Display AI-generated content analysis"""
    st.subheader("🎭 AI Content Generation Analysis")
    
    # Analyze generated vs original narratives
    generated_narratives = []
    counter_narratives = []
    original_narratives = []
    
    for nid, narrative in model.narratives.items():
        narrative_info = {
            'id': nid,
            'text': narrative['text'],
            'sentiment': narrative.get('sentiment', 0),
            'confidence': narrative.get('confidence', 'N/A')
        }
        
        if narrative.get('is_generated', False):
            generated_narratives.append(narrative_info)
        elif narrative.get('is_counter', False):
            counter_narratives.append(narrative_info)
        else:
            original_narratives.append(narrative_info)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎭 Generated Narratives", len(generated_narratives))
    with col2:
        st.metric("🔄 Counter-Narratives", len(counter_narratives))
    with col3:
        st.metric("📝 Original Narratives", len(original_narratives))
    
    # Generated content showcase
    if generated_narratives or counter_narratives:
        tabs = st.tabs(["🤖 AI Generated", "🛡️ Counter-Narratives"])
        
        with tabs[0]:
            if generated_narratives:
                st.markdown("**AI-Generated Narratives:**")
                for narrative in generated_narratives:
                    confidence_color = "🟢" if narrative['confidence'] == 'N/A' or narrative['confidence'] > 0.7 else "🟡"
                    st.markdown(f"""
                    <div class="highlight-box">
                    <strong>Narrative {narrative['id']}</strong> {confidence_color}<br>
                    <em>"{narrative['text']}"</em><br>
                    <small>Sentiment: {narrative['sentiment']:.2f} | Confidence: {narrative['confidence']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No AI-generated narratives were created during this simulation.")
        
        with tabs[1]:
            if counter_narratives:
                st.markdown("**AI-Generated Counter-Narratives:**")
                for narrative in counter_narratives:
                    st.markdown(f"""
                    <div class="highlight-box">
                    <strong>Counter-Narrative {narrative['id']}</strong> 🔄<br>
                    <em>"{narrative['text']}"</em><br>
                    <small>Sentiment: {narrative['sentiment']:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No counter-narratives were generated during this simulation.")
    
    # GAN training insights
    if hasattr(model, 'narrative_gan') and model.narrative_gan.training_history:
        st.subheader("🧠 GAN Training Insights")
        
        history = model.narrative_gan.training_history
        
        if history['g_losses'] and history['d_losses']:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_losses = go.Figure()
                fig_losses.add_trace(go.Scatter(
                    y=history['g_losses'],
                    name='Generator Loss',
                    line=dict(color='blue')
                ))
                fig_losses.add_trace(go.Scatter(
                    y=history['d_losses'],
                    name='Discriminator Loss',
                    line=dict(color='red')
                ))
                fig_losses.update_layout(
                    title="GAN Training Losses",
                    xaxis_title="Training Epoch",
                    yaxis_title="Loss Value"
                )
                st.plotly_chart(fig_losses, use_container_width=True)
            
            with col2:
                if history['generated_samples']:
                    st.markdown("**Sample Generated Text During Training:**")
                    for sample in history['generated_samples'][-3:]:  # Show last 3 samples
                        st.text(f"Epoch {sample['epoch']}: {sample['sample']}")

def display_advanced_analytics(data):
    """Display advanced simulation analytics"""
    main_df = data['main_simulation']
    
    if main_df.empty:
        st.warning("No data available for advanced analytics.")
        return
    
    st.subheader("📊 Advanced Analytics Dashboard")
    
    # Information theory metrics
    col1, col2 = st.columns(2)
    
    with col1:
        if 'information_entropy' in main_df.columns:
            fig_entropy = px.line(
                main_df,
                x='step',
                y='information_entropy',
                title="Information Entropy Over Time",
                labels={'information_entropy': 'Information Entropy (bits)', 'step': 'Simulation Step'}
            )
            fig_entropy.add_annotation(
                text="Higher entropy = More diverse narratives",
                xref="paper", yref="paper",
                x=0.02, y=0.98, showarrow=False,
                font=dict(size=10, color="gray")
            )
            st.plotly_chart(fig_entropy, use_container_width=True)
    
    with col2:
        if 'polarization_index' in main_df.columns:
            fig_polarization = px.line(
                main_df,
                x='step',
                y='polarization_index',
                title="Population Polarization Index",
                labels={'polarization_index': 'Polarization Index', 'step': 'Simulation Step'}
            )
            fig_polarization.add_annotation(
                text="Higher values = More polarized population",
                xref="paper", yref="paper",
                x=0.02, y=0.98, showarrow=False,
                font=dict(size=10, color="gray")
            )
            st.plotly_chart(fig_polarization, use_container_width=True)
    
    # Multi-dimensional analysis
    if all(col in main_df.columns for col in ['information_entropy', 'polarization_index', 'avg_sentiment']):
        st.subheader("🎯 Multi-Dimensional Analysis")
        
        # 3D scatter plot
        fig_3d = go.Figure(data=go.Scatter3d(
            x=main_df['information_entropy'],
            y=main_df['polarization_index'],
            z=main_df['avg_sentiment'],
            mode='markers+lines',
            marker=dict(
                size=5,
                color=main_df['step'],
                colorscale='Viridis',
                colorbar=dict(title="Simulation Step"),
                opacity=0.8
            ),
            line=dict(
                color='darkblue',
                width=2
            ),
            hovertemplate='<b>Step %{text}</b><br>' +
                         'Entropy: %{x:.2f}<br>' +
                         'Polarization: %{y:.2f}<br>' +
                         'Sentiment: %{z:.2f}<extra></extra>',
            text=main_df['step']
        ))
        
        fig_3d.update_layout(
            title="Information Landscape Evolution",
            scene=dict(
                xaxis_title="Information Entropy",
                yaxis_title="Polarization Index",
                zaxis_title="Average Sentiment",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Correlation analysis
    numeric_columns = main_df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 3:
        st.subheader("🔍 Correlation Analysis")
        
        correlation_matrix = main_df[numeric_columns].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig_corr.update_layout(
            title="Variable Correlation Matrix",
            height=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

def save_simulation_results(model, comprehensive_data, scenario_type):
    """Save simulation results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_{scenario_type}_{timestamp}.json"
    
    try:
        # Create results directory if it doesn't exist
        os.makedirs("simulation_results", exist_ok=True)
        filepath = os.path.join("simulation_results", filename)
        
        # Save comprehensive results
        model.save_simulation_state(filepath)
        
        st.success(f"💾 Simulation results saved to `{filepath}`")
        
        # Offer download
        with open(filepath, 'r') as f:
            st.download_button(
                label="📥 Download Simulation Results",
                data=f.read(),
                file_name=filename,
                mime="application/json"
            )
    
    except Exception as e:
        st.error(f"❌ Error saving simulation results: {str(e)}")

if __name__ == "__main__":
    main()