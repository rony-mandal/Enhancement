import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
import os

class NarrativeDataset(Dataset):
    """Dataset class for narrative texts"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class NarrativeGenerator(nn.Module):
    """Generator network for creating narrative texts"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, seq_len=64):
        super(NarrativeGenerator, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, noise, seq_len=None):
        if seq_len is None:
            seq_len = self.seq_len
        
        batch_size = noise.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(2, batch_size, self.hidden_dim)
        c0 = torch.zeros(2, batch_size, self.hidden_dim)
        
        outputs = []
        input_token = torch.randint(0, self.vocab_size, (batch_size, 1))
        
        for _ in range(seq_len):
            embedded = self.embedding(input_token)
            lstm_out, (h0, c0) = self.lstm(embedded, (h0, c0))
            output = self.output(self.dropout(lstm_out))
            outputs.append(output)
            input_token = torch.argmax(output, dim=-1)
        
        return torch.cat(outputs, dim=1)

class NarrativeDiscriminator(nn.Module):
    """Discriminator network for evaluating narrative authenticity"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super(NarrativeDiscriminator, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequences):
        embedded = self.embedding(sequences)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last hidden state from both directions
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.classifier(final_hidden)
        
        return output

class GANNarrativeGenerator:
    """Main GAN-based narrative generation system"""
    
    def __init__(self, vocab_size=10000, device='cpu'):
        self.device = device
        self.vocab_size = vocab_size
        
        # Initialize networks
        self.generator = NarrativeGenerator(vocab_size).to(device)
        self.discriminator = NarrativeDiscriminator(vocab_size).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # For text processing
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Simple tokenizer (in real scenario, use GPT2Tokenizer)
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.build_vocab()
        
        # Training history
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'generated_samples': []
        }

    def build_vocab(self):
        """Build vocabulary from common words and narrative-specific terms"""
        common_words = [
            'the', 'is', 'are', 'was', 'were', 'will', 'be', 'not', 'no', 'yes',
            'climate', 'change', 'warming', 'crisis', 'economy', 'economic', 'growth',
            'election', 'vote', 'democracy', 'health', 'virus', 'vaccine', 'war',
            'peace', 'technology', 'ai', 'artificial', 'intelligence', 'data',
            'people', 'government', 'media', 'news', 'false', 'true', 'fact',
            'dangerous', 'safe', 'risk', 'threat', 'secure', 'attack', 'defense',
            'rising', 'falling', 'increasing', 'decreasing', 'spreading', 'stopping'
        ]
        
        # Add padding, unknown, start, end tokens
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        vocab = special_tokens + common_words
        
        # Fill remaining vocabulary with dummy tokens
        while len(vocab) < self.vocab_size:
            vocab.append(f'<TOKEN_{len(vocab)}>')
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def text_to_indices(self, text):
        """Convert text to indices"""
        words = text.lower().split()
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        return indices

    def indices_to_text(self, indices):
        """Convert indices back to text"""
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx < len(self.idx_to_word):
                word = self.idx_to_word[idx]
                if word not in ['<PAD>', '<START>', '<END>']:
                    words.append(word)
        return ' '.join(words)

    def train_gan(self, narrative_texts, epochs=100, batch_size=32):
        """Train the GAN on narrative texts"""
        print(f"Training GAN on {len(narrative_texts)} narrative samples...")
        
        # Convert texts to indices
        text_indices = []
        for text in narrative_texts:
            indices = self.text_to_indices(text)
            # Pad or truncate to fixed length
            if len(indices) < 64:
                indices.extend([self.word_to_idx['<PAD>']] * (64 - len(indices)))
            else:
                indices = indices[:64]
            text_indices.append(indices)
        
        real_data = torch.tensor(text_indices).to(self.device)
        dataset = torch.utils.data.TensorDataset(real_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch_idx, (real_batch,) in enumerate(dataloader):
                batch_size = real_batch.size(0)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_output = self.discriminator(real_batch)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake data
                noise = torch.randn(batch_size, 100).to(self.device)
                fake_batch = self.generator(noise)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Convert continuous fake_batch to discrete indices for discriminator
                fake_indices = torch.argmax(fake_batch, dim=-1)
                fake_output = self.discriminator(fake_indices.detach())
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                fake_output = self.discriminator(fake_indices)
                g_loss = self.criterion(fake_output, real_labels)  # Want discriminator to think fake is real
                g_loss.backward()
                self.g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            
            self.training_history['g_losses'].append(avg_g_loss)
            self.training_history['d_losses'].append(avg_d_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
                
                # Generate sample
                sample = self.generate_narrative()
                self.training_history['generated_samples'].append((epoch, sample))
                print(f"Sample: {sample}")

    def generate_narrative(self, theme="", sentiment_target=None, max_length=64):
        """Generate a new narrative text"""
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(1, 100).to(self.device)
            generated_logits = self.generator(noise, max_length)
            generated_indices = torch.argmax(generated_logits, dim=-1)
            
            narrative_text = self.indices_to_text(generated_indices[0])
        
        self.generator.train()
        
        # Post-process the generated text
        narrative_text = self.post_process_narrative(narrative_text, theme, sentiment_target)
        
        return narrative_text

    def post_process_narrative(self, text, theme="", sentiment_target=None):
        """Post-process generated narrative for coherence and theme alignment"""
        words = text.split()
        
        # Remove duplicates and clean up
        cleaned_words = []
        for word in words:
            if word not in ['<UNK>', '<PAD>'] and (not cleaned_words or word != cleaned_words[-1]):
                cleaned_words.append(word)
        
        # Ensure minimum length
        if len(cleaned_words) < 5:
            theme_words = theme.split() if theme else ['situation', 'developing']
            cleaned_words.extend(theme_words)
        
        processed_text = ' '.join(cleaned_words[:15])  # Limit length
        
        # Add theme-specific words if provided
        if theme:
            theme_keywords = {
                'climate': ['climate change', 'warming', 'emissions'],
                'economic': ['economy', 'inflation', 'recession'],
                'health': ['health crisis', 'virus', 'pandemic'],
                'election': ['election', 'voting', 'democracy'],
                'war': ['conflict', 'military', 'security'],
                'tech': ['technology', 'ai', 'innovation']
            }
            
            for key, keywords in theme_keywords.items():
                if key in theme.lower():
                    processed_text = f"{random.choice(keywords)} {processed_text}"
                    break
        
        return processed_text.capitalize()

    def generate_counter_narrative(self, original_narrative):
        """Generate counter-narrative to oppose an existing narrative"""
        # Analyze original narrative
        sentiment = self.sentiment_analyzer.polarity_scores(original_narrative)['compound']
        
        # Generate opposing narrative with opposite sentiment
        counter_sentiment = -sentiment
        counter_narrative = self.generate_narrative(sentiment_target=counter_sentiment)
        
        # Add opposing language patterns
        if sentiment > 0:
            opposing_prefixes = ["No, ", "False: ", "Contrary to reports, "]
        else:
            opposing_prefixes = ["Actually, ", "In reality, ", "The truth is "]
        
        counter_narrative = random.choice(opposing_prefixes) + counter_narrative.lower()
        
        return counter_narrative

    def augment_narrative_dataset(self, original_narratives, augmentation_factor=2):
        """Augment existing narrative dataset with GAN-generated variants"""
        augmented_narratives = []
        
        for narrative in original_narratives:
            # Add original
            augmented_narratives.append(narrative)
            
            # Generate variants
            for _ in range(augmentation_factor):
                # Extract theme from original
                words = narrative.lower().split()
                theme = ' '.join(words[:3])  # Use first few words as theme hint
                
                variant = self.generate_narrative(theme=theme)
                augmented_narratives.append(variant)
        
        return augmented_narratives

    def evaluate_narrative_quality(self, narrative):
        """Evaluate the quality of a generated narrative"""
        metrics = {}
        
        # Sentiment consistency
        sentiment = self.sentiment_analyzer.polarity_scores(narrative)
        metrics['sentiment'] = sentiment['compound']
        
        # Length appropriateness
        word_count = len(narrative.split())
        metrics['length_score'] = min(1.0, word_count / 10.0)  # Ideal ~10 words
        
        # Coherence (simple heuristic - real implementation would use more sophisticated methods)
        unique_words = len(set(narrative.lower().split()))
        total_words = len(narrative.split())
        metrics['diversity'] = unique_words / max(total_words, 1)
        
        # Overall quality score
        metrics['quality'] = (metrics['length_score'] * 0.3 + 
                            abs(metrics['sentiment']) * 0.4 + 
                            metrics['diversity'] * 0.3)
        
        return metrics

    def save_model(self, path):
        """Save trained GAN models"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'vocab': {'word_to_idx': self.word_to_idx, 'idx_to_word': self.idx_to_word},
            'training_history': self.training_history
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load trained GAN models"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.word_to_idx = checkpoint['vocab']['word_to_idx']
            self.idx_to_word = checkpoint['vocab']['idx_to_word']
            self.training_history = checkpoint['training_history']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

# Advanced GAN with Transformer architecture for better narrative quality
class TransformerNarrativeGAN:
    """Advanced GAN using transformer architecture for narrative generation"""
    
    def __init__(self, model_name='gpt2', device='cpu'):
        self.device = device
        self.model_name = model_name
        
        # Load pre-trained GPT-2 for fine-tuning
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generator (fine-tuned GPT-2)
        self.generator = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        
        # Discriminator (BERT-style classifier)
        config = GPT2Config.from_pretrained(model_name)
        config.num_labels = 1
        self.discriminator = GPT2ForSequenceClassification(config).to(device)
        
        # Optimizers
        self.g_optimizer = optim.AdamW(self.generator.parameters(), lr=1e-5)
        self.d_optimizer = optim.AdamW(self.discriminator.parameters(), lr=1e-5)
        
        # Utilities
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def prepare_dataset(self, narrative_texts, max_length=128):
        """Prepare dataset for training"""
        return NarrativeDataset(narrative_texts, self.tokenizer, max_length)

    def train_generator(self, train_dataloader, epochs=3):
        """Fine-tune the generator on narrative data"""
        print("Fine-tuning generator...")
        
        self.generator.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.g_optimizer.zero_grad()
                
                outputs = self.generator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                self.g_optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Generator Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def generate_advanced_narrative(self, prompt="", max_length=50, temperature=0.8):
        """Generate narrative using fine-tuned transformer"""
        self.generator.eval()
        
        with torch.no_grad():
            if prompt:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            else:
                # Start with a random narrative seed
                seeds = ["The situation is", "Reports indicate", "Sources confirm", "Analysis shows"]
                prompt = random.choice(seeds)
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            output = self.generator.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                top_p=0.9
            )
            
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean up the generated text
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text

class GPT2ForSequenceClassification(GPT2LMHeadModel):
    """GPT-2 model modified for sequence classification (discriminator)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.n_embd, config.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use the last token's hidden state for classification
        logits = self.classifier(hidden_states[:, -1, :])
        prediction = self.sigmoid(logits)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(prediction.view(-1), labels.view(-1))
        
        return type('Output', (), {
            'loss': loss,
            'logits': prediction,
            'hidden_states': hidden_states
        })()

# Reinforcement Learning Environment for Narrative Optimization
class NarrativeEnvironment:
    """RL Environment for optimizing narrative generation and spread"""
    
    def __init__(self, narrative_model, gan_generator):
        self.narrative_model = narrative_model
        self.gan_generator = gan_generator
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_narratives = {}
        self.step_count = 0
        self.total_influence = 0
        return self.get_state()

    def get_state(self):
        """Get current state vector for RL agent"""
        state = np.zeros(20)  # 20-dimensional state space
        
        # Narrative statistics
        state[0] = len(self.current_narratives)
        if self.current_narratives:
            sentiments = [n['sentiment'] for n in self.current_narratives.values()]
            state[1] = np.mean(sentiments)
            state[2] = np.std(sentiments) if len(sentiments) > 1 else 0
        
        # Model statistics
        if hasattr(self.narrative_model, 'agents') and self.narrative_model.agents:
            agent_beliefs = []
            agent_sentiments = []
            for agent in self.narrative_model.agents:
                agent_beliefs.append(len(agent.beliefs))
                agent_sentiments.append(agent.sentiment)
            
            state[3] = np.mean(agent_beliefs)
            state[4] = np.mean(agent_sentiments)
            state[5] = np.std(agent_sentiments) if len(agent_sentiments) > 1 else 0
        
        # Environmental factors
        state[6] = self.step_count / 100.0  # Normalized step count
        state[7] = self.total_influence / 1000.0  # Normalized total influence
        
        # Fill remaining dimensions with contextual information
        # This could include network metrics, external events, etc.
        
        return state

    def step(self, action):
        """Execute action and return new state, reward, done"""
        # Action space: 0=generate_positive, 1=generate_negative, 2=generate_counter, 3=do_nothing
        
        reward = 0
        done = False
        
        if action == 0:  # Generate positive narrative
            narrative = self.gan_generator.generate_narrative(sentiment_target=0.7)
            reward = self.add_narrative(narrative, expected_sentiment=0.7)
        
        elif action == 1:  # Generate negative narrative
            narrative = self.gan_generator.generate_narrative(sentiment_target=-0.7)
            reward = self.add_narrative(narrative, expected_sentiment=-0.7)
        
        elif action == 2:  # Generate counter-narrative
            if self.current_narratives:
                dominant_narrative = max(self.current_narratives.values(), 
                                       key=lambda x: x.get('influence', 0))
                counter = self.gan_generator.generate_counter_narrative(dominant_narrative['text'])
                reward = self.add_narrative(counter, is_counter=True)
        
        elif action == 3:  # Do nothing
            reward = -0.1  # Small penalty for inaction
        
        self.step_count += 1
        self.total_influence += reward
        
        # End episode after certain steps or conditions
        if self.step_count >= 100 or self.total_influence < -50:
            done = True
        
        return self.get_state(), reward, done

    def add_narrative(self, text, expected_sentiment=None, is_counter=False):
        """Add narrative and calculate reward"""
        if not text or len(text.strip()) < 5:
            return -1.0  # Penalty for poor quality
        
        # Calculate narrative properties
        sentiment = self.gan_generator.sentiment_analyzer.polarity_scores(text)['compound']
        embedding = self.gan_generator.sentence_model.encode(text)
        
        # Calculate reward based on quality and impact
        reward = 0
        
        # Quality reward
        quality_metrics = self.gan_generator.evaluate_narrative_quality(text)
        reward += quality_metrics['quality'] * 2
        
        # Sentiment alignment reward
        if expected_sentiment is not None:
            sentiment_error = abs(sentiment - expected_sentiment)
            reward += (1 - sentiment_error) * 3
        
        # Diversity reward (avoid repetition)
        if self.current_narratives:
            similarities = []
            for existing_narrative in self.current_narratives.values():
                similarity = np.dot(embedding, existing_narrative['embedding'])
                similarities.append(similarity)
            avg_similarity = np.mean(similarities)
            reward += (1 - avg_similarity) * 2  # Reward for being different
        
        # Counter-narrative bonus
        if is_counter:
            reward += 1.5
        
        # Store narrative
        narrative_id = len(self.current_narratives)
        self.current_narratives[narrative_id] = {
            'text': text,
            'sentiment': sentiment,
            'embedding': embedding,
            'influence': reward,
            'is_counter': is_counter
        }
        
        return max(0, reward)  # Ensure non-negative reward
            '