# Narrative Spread Simulation

This project simulates the spread of narratives in a population using agent-based modeling with Mesa 3.2.0. It includes diverse agent types (Influencers, Regulars, Skeptics), influence dynamics, adaptive counter-narratives, and advanced visualizations.

## Setup

1. Install Python 3.8 or higher.
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```
   streamlit run app.py
   ```

## Usage

- Enter one or more narratives in the text area, each on a new line.
- Adjust the number of agents and simulation steps using the sliders.
- Click "Run Simulation" to start the simulation and view the results, including believer counts, sentiment trends, and an agent network graph.

## Notes

- The simulation uses a pre-trained sentence transformer model. Ensure you have internet access the first time you run the script to download the model.
- The VADER sentiment analyzer will also download its lexicon on first run.
- After the initial run, the system can be used offline.
- This project is optimized for Mesa 3.2.0.