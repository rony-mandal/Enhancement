import mesa
import random

class NarrativeAgent(mesa.Agent):
    AGENT_TYPES = {
        "Influencer": {"influence": 0.8, "spread_chance": 0.7},
        "Regular": {"influence": 0.5, "spread_chance": 0.5},
        "Skeptic": {"influence": 0.3, "spread_chance": 0.3}
    }

    def __init__(self, model):
        super().__init__(model)
        self.type = random.choice(list(self.AGENT_TYPES.keys()))
        self.influence = self.AGENT_TYPES[self.type]["influence"]
        self.spread_chance = self.AGENT_TYPES[self.type]["spread_chance"]
        self.beliefs = {}  # {narrative_id: belief_score}
        self.sentiment = 0.0
        self.connections = []

    def step(self):
        for narrative_id, belief in self.beliefs.items():
            if belief > 0.5 and random.random() < self.spread_chance:
                for neighbor in self.connections:
                    neighbor.receive_narrative(narrative_id, belief, self.influence)

    def receive_narrative(self, narrative_id, incoming_belief, sender_influence):
        if narrative_id not in self.beliefs:
            self.beliefs[narrative_id] = 0.0
        alpha = 0.3 * (sender_influence / self.influence if self.influence > 0 else 1.0)
        self.beliefs[narrative_id] = (1 - alpha) * self.beliefs[narrative_id] + alpha * incoming_belief
        narrative_sentiment = self.model.narratives[narrative_id]['sentiment']
        self.sentiment = (self.sentiment + narrative_sentiment) / 2