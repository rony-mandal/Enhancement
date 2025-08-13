import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

class NarrativeGAN:
    """Advanced GAN system for generating contextually appropriate narratives"""
    
    def __init__(self, vocab_size=1000, embedding_dim=128, hidden_dim=256, max_seq_length=50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Initialize networks
        self.generator = NarrativeGenerator(vocab_size, embedding_dim, hidden_dim, max_seq_length)
        self.discriminator = NarrativeDiscriminator(vocab_size, embedding_dim, hidden_dim)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Training parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Vocabulary management
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.build_vocabulary()
        
        # Pre-trained model for advanced generation (optional)
        self.use_pretrained = False
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        
        # Training history
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'generated_samples': []
        }
    
    def build_vocabulary(self):
        """Build vocabulary from common disinformation/narrative terms"""
        # Core vocabulary for information warfare scenarios
        base_vocab = [
            '<PAD>', '<START>', '<END>', '<UNK>',
            # Common narrative terms
            'war', 'peace', 'attack', 'defense', 'enemy', 'ally', 'threat', 'safe',
            'government', 'military', 'soldiers', 'civilians', 'crisis', 'emergency',
            'fake', 'real', 'truth', 'lie', 'propaganda', 'information', 'news',
            'happening', 'occurred', 'imminent', 'prevented', 'stopped', 'continuing',
            # Sentiment words
            'good', 'bad', 'dangerous', 'secure', 'worried', 'confident', 'afraid',
            'hopeful', 'desperate', 'strong', 'weak', 'winning', 'losing',
            # Time and urgency
            'now', 'soon', 'never', 'always', 'today', 'tomorrow', 'yesterday',
            'urgent', 'immediate', 'delayed', 'postponed',
            # Location terms
            'here', 'there', 'everywhere', 'nowhere', 'border', 'city', 'country',
            # Action words
            'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
            'happening', 'occurred', 'planned', 'executed', 'failed', 'succeeded'
        ]
        
        # Add numbers and common words
        numbers = [str(i) for i in range(100)]
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'from', 'with', 'by']
        
        full_vocab = base_vocab + numbers + common_words
        
        # Pad vocabulary to desired size
        while len(full_vocab) < self.vocab_size:
            full_vocab.append(f'<WORD_{len(full_vocab)}>')
        
        # Create mappings
        for idx, word in enumerate(full_vocab[:self.vocab_size]):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def initialize_pretrained(self):
        """Initialize GPT-2 for advanced text generation"""
        try:
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_model.to(self.device)
            self.use_pretrained = True
            print("✅ GPT-2 initialized successfully")
        except Exception as e:
            print(f"⚠️ Could not initialize GPT-2: {e}")
            self.use_pretrained = False
    
    def text_to_sequence(self, text):
        """Convert text to sequence of token indices"""
        words = text.lower().split()
        sequence = [self.word_to_idx.get('<START>', 1)]
        
        for word in words[:self.max_seq_length-2]:
            sequence.append(self.word_to_idx.get(word, self.word_to_idx.get('<UNK>', 3)))
        
        sequence.append(self.word_to_idx.get('<END>', 2))
        
        # Pad sequence
        while len(sequence) < self.max_seq_length:
            sequence.append(self.word_to_idx.get('<PAD>', 0))
        
        return torch.LongTensor(sequence[:self.max_seq_length])
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        words = []
        for idx in sequence:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            
            word = self.idx_to_word.get(idx, '<UNK>')
            if word in ['<PAD>', '<END>']:
                break
            if word not in ['<START>', '<UNK>']:
                words.append(word)
        
        return ' '.join(words)
    
    def generate_narrative(self, sentiment_target=0.0, topic_keywords=None, context=None):
        """Generate a new narrative with specified parameters"""
        
        if self.use_pretrained and self.gpt2_model:
            return self.generate_with_gpt2(sentiment_target, topic_keywords, context)
        else:
            return self.generate_with_gan(sentiment_target, topic_keywords, context)
    
    def generate_with_gan(self, sentiment_target=0.0, topic_keywords=None, context=None):
        """Generate narrative using the custom GAN"""
        self.generator.eval()
        
        with torch.no_grad():
            # Create condition vector
            condition = self.create_condition_vector(sentiment_target, topic_keywords, context)
            
            # Generate noise
            noise = torch.randn(1, 100).to(self.device)
            
            # Generate sequence
            generated_sequence = self.generator(noise, condition)
            
            # Convert to text
            generated_text = self.sequence_to_text(generated_sequence[0])
            
            return {
                'text': generated_text,
                'sentiment': sentiment_target,
                'confidence': 0.8,  # Placeholder
                'method': 'GAN'
            }
    
    def generate_with_gpt2(self, sentiment_target=0.0, topic_keywords=None, context=None):
        """Generate narrative using GPT-2"""
        # Create prompt based on parameters
        prompt = self.create_narrative_prompt(sentiment_target, topic_keywords, context)
        
        try:
            inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.gpt2_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.gpt2_tokenizer.eos_token_id
                )
            
            generated_text = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            return {
                'text': generated_text,
                'sentiment': sentiment_target,
                'confidence': 0.9,
                'method': 'GPT-2'
            }
        
        except Exception as e:
            print(f"GPT-2 generation failed: {e}")
            return self.generate_with_gan(sentiment_target, topic_keywords, context)
    
    def create_narrative_prompt(self, sentiment_target, topic_keywords, context):
        """Create prompt for narrative generation"""
        prompt_parts = []
        
        if sentiment_target < -0.5:
            prompt_parts.append("Breaking: Alarming situation as")
        elif sentiment_target > 0.5:
            prompt_parts.append("Good news: Positive developments show")
        else:
            prompt_parts.append("Reports indicate that")
        
        if topic_keywords:
            prompt_parts.append(' '.join(topic_keywords[:3]))
        
        if context:
            prompt_parts.append(f"in {context}")
        
        return ' '.join(prompt_parts) + ' '
    
    def create_condition_vector(self, sentiment_target, topic_keywords, context):
        """Create conditioning vector for controlled generation"""
        condition_dim = 20
        condition = torch.zeros(1, condition_dim).to(self.device)
        
        # Sentiment encoding
        condition[0, 0] = sentiment_target
        
        # Topic encoding (simplified)
        if topic_keywords:
            for i, keyword in enumerate(topic_keywords[:5]):
                if keyword in self.word_to_idx:
                    idx = self.word_to_idx[keyword]
                    condition[0, i + 1] = idx / self.vocab_size
        
        return condition
    
    def generate_counter_narrative(self, original_narrative, context=None):
        """Generate counter-narrative for a given narrative"""
        # Analyze original narrative
        sentiment = self.estimate_sentiment(original_narrative)
        topic_keywords = self.extract_keywords(original_narrative)
        
        # Generate opposite sentiment narrative
        counter_sentiment = -sentiment * 0.8  # Slightly less extreme
        
        # Modify keywords for counter-narrative
        counter_keywords = self.get_counter_keywords(topic_keywords)
        
        counter_narrative = self.generate_narrative(
            sentiment_target=counter_sentiment,
            topic_keywords=counter_keywords,
            context=context
        )
        
        counter_narrative['is_counter'] = True
        counter_narrative['original_narrative'] = original_narrative
        
        return counter_narrative
    
    def estimate_sentiment(self, text):
        """Simple sentiment estimation"""
        positive_words = ['good', 'safe', 'peace', 'secure', 'hopeful', 'confident', 'winning', 'succeeded']
        negative_words = ['bad', 'dangerous', 'war', 'threat', 'afraid', 'worried', 'losing', 'failed']
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def extract_keywords(self, text):
        """Extract key terms from narrative"""
        words = text.lower().split()
        important_words = [word for word in words if word in self.word_to_idx and 
                          word not in ['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are']]
        return important_words[:5]
    
    def get_counter_keywords(self, original_keywords):
        """Generate counter-keywords for opposing narrative"""
        counter_mapping = {
            'war': 'peace',
            'attack': 'defense',
            'enemy': 'ally',
            'threat': 'safe',
            'dangerous': 'secure',
            'bad': 'good',
            'losing': 'winning',
            'failed': 'succeeded',
            'worried': 'confident',
            'afraid': 'hopeful'
        }
        
        counter_keywords = []
        for keyword in original_keywords:
            counter = counter_mapping.get(keyword, keyword)
            counter_keywords.append(counter)
        
        return counter_keywords
    
    def train_step(self, real_narratives):
        """Single training step for the GAN"""
        batch_size = len(real_narratives)
        
        # Prepare real data
        real_sequences = []
        for narrative in real_narratives:
            seq = self.text_to_sequence(narrative)
            real_sequences.append(seq)
        
        real_data = torch.stack(real_sequences).to(self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_output = self.discriminator(real_data)
        real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))
        
        # Fake data
        noise = torch.randn(batch_size, 100).to(self.device)
        condition = torch.randn(batch_size, 20).to(self.device)  # Random conditions
        fake_data = self.generator(noise, condition)
        fake_output = self.discriminator(fake_data.detach())
        fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_output = self.discriminator(fake_data)
        g_loss = nn.BCELoss()(fake_output, torch.ones_like(fake_output))
        g_loss.backward()
        self.g_optimizer.step()
        
        # Record losses
        self.training_history['g_losses'].append(g_loss.item())
        self.training_history['d_losses'].append(d_loss.item())
        
        return g_loss.item(), d_loss.item()
    
    def train(self, training_narratives, epochs=100):
        """Train the GAN on narrative data"""
        print(f"🚀 Training GAN for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Sample random batch
            batch_narratives = random.sample(training_narratives, min(8, len(training_narratives)))
            
            g_loss, d_loss = self.train_step(batch_narratives)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: G_Loss={g_loss:.4f}, D_Loss={d_loss:.4f}")
                
                # Generate sample
                sample = self.generate_narrative(sentiment_target=random.uniform(-1, 1))
                self.training_history['generated_samples'].append({
                    'epoch': epoch,
                    'sample': sample['text']
                })
                print(f"Sample: {sample['text']}")
        
        print("✅ GAN training completed!")
    
    def save_model(self, filepath):
        """Save the trained GAN model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'training_history': self.training_history,
            'vocab_mappings': {
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word
            }
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained GAN model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.training_history = checkpoint.get('training_history', self.training_history)
            
            vocab_mappings = checkpoint.get('vocab_mappings', {})
            if vocab_mappings:
                self.word_to_idx = vocab_mappings['word_to_idx']
                self.idx_to_word = vocab_mappings['idx_to_word']
            
            print(f"📂 Model loaded from {filepath}")


class NarrativeGenerator(nn.Module):
    """Generator network for creating narrative sequences"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_seq_length):
        super(NarrativeGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Input processing
        self.noise_projection = nn.Linear(100, hidden_dim)
        self.condition_projection = nn.Linear(20, hidden_dim)
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Output layers
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, noise, condition):
        batch_size = noise.size(0)
        
        # Process inputs
        noise_features = torch.relu(self.noise_projection(noise))  # [batch, hidden_dim]
        condition_features = torch.relu(self.condition_projection(condition))  # [batch, hidden_dim]
        
        # Combine features
        combined_features = noise_features + condition_features  # [batch, hidden_dim]
        
        # Expand for sequence generation
        lstm_input = combined_features.unsqueeze(1).repeat(1, self.max_seq_length, 1)  # [batch, seq_len, hidden_dim]
        
        # Generate sequence
        lstm_output, _ = self.lstm(lstm_input)  # [batch, seq_len, hidden_dim]
        lstm_output = self.dropout(lstm_output)
        
        # Project to vocabulary
        logits = self.output_projection(lstm_output)  # [batch, seq_len, vocab_size]
        
        # Apply softmax and sample
        probs = torch.softmax(logits, dim=-1)
        
        # Gumbel softmax for differentiable sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-20) + 1e-20)
        samples = torch.softmax((torch.log(probs + 1e-20) + gumbel_noise) / 0.5, dim=-1)
        
        # Convert to hard samples during inference
        if not self.training:
            _, max_indices = torch.max(samples, dim=-1)
            return max_indices
        
        return samples


class NarrativeDiscriminator(nn.Module):
    """Discriminator network for evaluating narrative authenticity"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NarrativeDiscriminator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Handle both hard samples (indices) and soft samples (probabilities)
        if x.dtype == torch.long:
            # Hard samples - use embedding
            embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        else:
            # Soft samples - matrix multiply with embedding weights
            embedded = torch.matmul(x, self.embedding.weight)  # [batch, seq_len, embedding_dim]
        
        # Transpose for conv1d [batch, embedding_dim, seq_len]
        conv_input = embedded.transpose(1, 2)
        
        # Convolutional feature extraction
        conv_out = torch.relu(self.conv1(conv_input))
        conv_out = torch.relu(self.conv2(conv_out))
        conv_out = torch.relu(self.conv3(conv_out))
        
        # Transpose back for LSTM [batch, seq_len, hidden_dim]
        lstm_input = conv_out.transpose(1, 2)
        
        # LSTM processing
        lstm_output, (hidden, _) = self.lstm(lstm_input)
        
        # Use final hidden state for classification
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # Concatenate bidirectional
        
        # Classification
        output = self.classifier(final_hidden)
        
        return output.squeeze(-1)


class AdaptiveCounterNarrativeSystem:
    """System for generating adaptive counter-narratives during simulation"""
    
    def __init__(self, narrative_gan):
        self.narrative_gan = narrative_gan
        self.counter_narrative_history = []
        self.effectiveness_scores = {}
        
    def generate_adaptive_counter(self, dominant_narrative, simulation_context):
        """Generate counter-narrative adapted to current simulation state"""
        
        # Analyze dominant narrative
        narrative_analysis = self.analyze_narrative_impact(dominant_narrative, simulation_context)
        
        # Generate multiple counter-narrative candidates
        candidates = []
        for i in range(3):
            counter = self.narrative_gan.generate_counter_narrative(
                dominant_narrative['text'],
                context=simulation_context
            )
            
            # Score candidate effectiveness
            effectiveness = self.score_counter_effectiveness(
                counter, dominant_narrative, narrative_analysis
            )
            
            candidates.append({
                'narrative': counter,
                'effectiveness_score': effectiveness,
                'strategy': self.determine_counter_strategy(narrative_analysis)
            })
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda x: x['effectiveness_score'])
        
        # Record for learning
        self.counter_narrative_history.append({
            'original': dominant_narrative,
            'counter': best_candidate,
            'context': simulation_context
        })
        
        return best_candidate['narrative']
    
    def analyze_narrative_impact(self, narrative, context):
        """Analyze the impact and characteristics of a narrative"""
        return {
            'sentiment_strength': abs(narrative.get('sentiment', 0)),
            'belief_penetration': context.get('believer_count', 0) / context.get('total_agents', 1),
            'spread_velocity': context.get('spread_rate', 0),
            'agent_types_affected': context.get('affected_types', []),
            'network_coverage': context.get('network_coverage', 0)
        }
    
    def score_counter_effectiveness(self, counter, original, analysis):
        """Score the potential effectiveness of a counter-narrative"""
        base_score = 0.5
        
        # Sentiment opposition bonus
        if counter.get('sentiment', 0) * original.get('sentiment', 0) < 0:
            base_score += 0.3
        
        # Confidence bonus
        base_score += counter.get('confidence', 0) * 0.2
        
        # Context relevance (simplified)
        if analysis['sentiment_strength'] > 0.7:
            base_score += 0.2  # Strong narratives need strong counters
        
        return min(1.0, base_score)
    
    def determine_counter_strategy(self, analysis):
        """Determine the best counter-narrative strategy"""
        if analysis['belief_penetration'] > 0.7:
            return "aggressive_debunk"
        elif analysis['spread_velocity'] > 0.5:
            return "rapid_response"
        elif analysis['sentiment_strength'] > 0.8:
            return "emotional_counter"
        else:
            return "factual_correction"
        