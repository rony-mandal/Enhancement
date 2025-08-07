from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import os

nltk.download('vader_lexicon')

model = SentenceTransformer('all-MiniLM-L6-v2')
sid = SentimentIntensityAnalyzer()

def process_narratives(narrative_texts):
    narratives = {}
    for i, text in enumerate(narrative_texts):
        embedding = model.encode(text)
        sentiment = sid.polarity_scores(text)['compound']
        narratives[i] = {'text': text, 'embedding': embedding, 'sentiment': sentiment}
    return narratives

def get_available_scenarios():
    """Get list of available narrative scenarios from CSV files"""
    scenario_mapping = {
        'War/Conflict': 'data/psyops_narratives.csv',
        'Economic Crisis': 'data/economic_narratives.csv',
        'Health Emergency': 'data/health_narratives.csv',
        'Political Election': 'data/election_narratives.csv',
        'Climate Change': 'data/climate_narratives.csv',
        'Technology/AI': 'data/tech_narratives.csv'
    }
    
    # Filter out scenarios where CSV files don't exist
    available_scenarios = {}
    for scenario_name, file_path in scenario_mapping.items():
        if os.path.exists(file_path):
            available_scenarios[scenario_name] = file_path
    
    return available_scenarios

def load_narrative_data(scenario='War/Conflict'):
    """Load narrative data based on selected scenario"""
    scenario_mapping = get_available_scenarios()
    
    if scenario not in scenario_mapping:
        # Default to war scenario if selected scenario doesn't exist
        file_path = 'data/psyops_narratives.csv'
    else:
        file_path = scenario_mapping[scenario]
    
    try:
        df = pd.read_csv(file_path)
        narratives = {}
        for i, row in df.iterrows():
            text = row['text']
            sentiment = row.get('sentiment', sid.polarity_scores(text)['compound'])
            embedding = model.encode(text)
            narratives[i] = {'text': text, 'embedding': embedding, 'sentiment': sentiment}
        return narratives
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Using empty narratives.")
        return {}
    except Exception as e:
        print(f"Error loading narrative data: {e}")
        return {}