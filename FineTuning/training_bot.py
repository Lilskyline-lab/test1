"""
Multi-Source Continuous Training System with Quality Filtering
"""

import os
import sys
import json
import time
import requests
import re
from tqdm import tqdm
from typing import List, Dict
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Local imports
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from Tokenizer import MYBPE


# ============================================
# MULTI-SOURCE SCRAPERS
# ============================================

class WikipediaScraper:
    def __init__(self, language='en'):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "WikiQABot/1.0"}

    def get_random_articles(self, count=10):
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': min(count, 20)
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [{"title": a["title"], "id": a["id"]} for a in data["query"]["random"]]
        except Exception as e:
            print(f"⚠️ Wikipedia error: {e}")
            return []

    def get_article_content(self, title: str) -> Dict:
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            page = list(data['query']['pages'].values())[0]
            if 'extract' not in page:
                return None
            text = self._clean_text(page['extract'])
            return {'title': title, 'content': text, 'length': len(text), 'source': 'wikipedia'}
        except:
            return None

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'==+ .*? ==+', '', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()


class RedditScraper:
    """Scrape Reddit for natural conversations (read-only, no auth needed)"""
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.headers = {"User-Agent": "RedditQABot/1.0"}
        # Popular subreddits with good Q&A
        self.subreddits = ['explainlikeimfive', 'AskReddit', 'NoStupidQuestions', 
                          'todayilearned', 'CasualConversation']

    def get_posts(self, count=10):
        """Get top posts from random subreddit"""
        subreddit = random.choice(self.subreddits)
        url = f"{self.base_url}/r/{subreddit}/top.json?limit={min(count, 25)}&t=week"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            for post in data['data']['children']:
                p = post['data']
                if p.get('selftext') and len(p['selftext']) > 100:
                    posts.append({
                        'title': p['title'],
                        'content': p['selftext'][:1000],  # Limit length
                        'source': 'reddit'
                    })
            return posts
        except Exception as e:
            print(f"⚠️ Reddit error: {e}")
            return []


class StackOverflowScraper:
    """Scrape StackOverflow for code Q&A"""
    def __init__(self):
        self.api_url = "https://api.stackexchange.com/2.3/questions"
        self.headers = {"User-Agent": "StackQABot/1.0"}
        self.tags = ['python', 'javascript', 'java', 'machine-learning', 'algorithms']

    def get_questions(self, count=10):
        """Get recent questions with accepted answers"""
        tag = random.choice(self.tags)
        params = {
            'order': 'desc',
            'sort': 'votes',
            'tagged': tag,
            'site': 'stackoverflow',
            'filter': 'withbody',
            'pagesize': min(count, 20)
        }
        
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            questions = []
            for item in data.get('items', []):
                if item.get('body') and item.get('accepted_answer_id'):
                    # Get answer
                    answer = self._get_answer(item['accepted_answer_id'])
                    if answer:
                        questions.append({
                            'title': item['title'],
                            'question': self._clean_html(item['body'])[:800],
                            'answer': answer,
                            'source': 'stackoverflow'
                        })
            return questions
        except Exception as e:
            print(f"⚠️ StackOverflow error: {e}")
            return []

    def _get_answer(self, answer_id):
        """Get answer by ID"""
        url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
        params = {'site': 'stackoverflow', 'filter': 'withbody'}
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            if data.get('items'):
                return self._clean_html(data['items'][0]['body'])[:800]
        except:
            return None

    def _clean_html(self, text: str) -> str:
        """Basic HTML tag removal"""
        text = re.sub(r'<code>.*?</code>', '[CODE]', text, flags=re.DOTALL)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()


# ============================================
# ENHANCED Q&A GENERATOR WITH TEMPLATES
# ============================================

class QAGenerator:
    def __init__(self):
        self.templates = {
            'definition': [
                "What is {subject}?",
                "Define {subject}.",
                "Can you explain {subject}?",
                "Tell me about {subject}.",
            ],
            'how': [
                "How does {subject} work?",
                "How to use {subject}?",
                "Explain how {subject} functions.",
                "What's the process of {subject}?",
            ],
            'why': [
                "Why is {subject} important?",
                "Why do we use {subject}?",
                "What's the purpose of {subject}?",
            ],
            'where': [
                "Where is {subject} used?",
                "In what context is {subject} applied?",
            ],
            'when': [
                "When was {subject} created?",
                "When should I use {subject}?",
            ]
        }
        
        self.conversation_templates = [
            ("Hello", "Hello! How can I help you today?"),
            ("Hi there", "Hi! Great to chat with you!"),
            ("How are you?", "I'm doing well, thank you for asking! How about you?"),
            ("Thank you", "You're very welcome!"),
            ("Good morning", "Good morning! Hope you're having a great day!"),
            ("What's your name?", "I'm an AI assistant here to help you."),
            ("Tell me a joke", "Why did the programmer quit? Because they didn't get arrays!"),
            ("What can you do?", "I can answer questions, help with information, and chat with you!"),
        ]

    def _truncate_smart(self, text: str, max_len=350):
        """Smart truncation at sentence boundaries"""
        if len(text) <= max_len:
            return text.strip()
        
        truncated = text[:max_len]
        # Find last sentence boundary
        for punct in ['. ', '! ', '? ', '\n']:
            pos = truncated.rfind(punct)
            if pos > max_len * 0.7:  # At least 70% of max_len
                return truncated[:pos + 1].strip()
        
        return truncated.strip() + "..."

    def generate_wikipedia_qa(self, title: str, content: str, max_pairs=2) -> List[Dict]:
        """Generate Q&A from Wikipedia articles with varied templates"""
        qa_pairs = []
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 150]
        
        template_types = list(self.templates.keys())
        
        for i, paragraph in enumerate(paragraphs[:max_pairs]):
            # Vary question types
            template_type = template_types[i % len(template_types)]
            questions = self.templates[template_type]
            question = random.choice(questions).format(subject=title)
            
            answer = self._truncate_smart(paragraph, 350)
            if len(answer) > 50:
                qa_pairs.append({
                    "human": question, 
                    "assistant": answer,
                    "source": "wikipedia"
                })
        
        return qa_pairs

    def generate_reddit_qa(self, post: Dict) -> List[Dict]:
        """Convert Reddit posts to Q&A format"""
        qa_pairs = []
        title = post['title'].strip()
        content = post['content'].strip()
        
        # If title is a question
        if '?' in title:
            answer = self._truncate_smart(content, 300)
            if len(answer) > 50:
                qa_pairs.append({
                    "human": title,
                    "assistant": answer,
                    "source": "reddit"
                })
        else:
            # Make it conversational
            question = f"Tell me about: {title}"
            answer = self._truncate_smart(content, 300)
            if len(answer) > 50:
                qa_pairs.append({
                    "human": question,
                    "assistant": answer,
                    "source": "reddit"
                })
        
        return qa_pairs

    def generate_stackoverflow_qa(self, item: Dict) -> List[Dict]:
        """Convert StackOverflow Q&A to training format"""
        qa_pairs = []
        
        question = item['title']
        # Add context if available
        if item.get('question'):
            question += f"\n{self._truncate_smart(item['question'], 200)}"
        
        answer = self._truncate_smart(item['answer'], 400)
        
        if len(answer) > 50:
            qa_pairs.append({
                "human": question,
                "assistant": answer,
                "source": "stackoverflow"
            })
        
        return qa_pairs


# ============================================
# QUALITY FILTER
# ============================================

class QualityFilter:
    """Filter low-quality Q&A pairs"""
    
    def filter_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Apply multiple quality checks"""
        filtered = []
        
        for qa in qa_pairs:
            if not self._is_valid_qa(qa):
                continue
            filtered.append(qa)
        
        return filtered

    def _is_valid_qa(self, qa: Dict) -> bool:
        """Check if Q&A meets quality standards"""
        human = qa['human'].strip()
        assistant = qa['assistant'].strip()
        
        # Check minimum length
        if len(assistant) < 30 or len(human) < 5:
            return False
        
        # Check maximum length (avoid very long responses)
        if len(assistant) > 600 or len(human) > 200:
            return False
        
        # Filter if question is in answer (likely repetitive)
        if human.lower() in assistant.lower():
            return False
        
        # Filter too many special characters
        special_count = sum(1 for c in assistant if c in '[]{}()|<>@#$%^&*')
        if special_count > len(assistant) * 0.1:  # More than 10%
            return False
        
        # Filter if too repetitive
        words = assistant.lower().split()
        if len(words) != len(set(words)) and len(set(words)) < len(words) * 0.5:
            return False
        
        # Filter common bad patterns
        bad_patterns = [
            'this article needs',
            'citation needed',
            'clarification needed',
            '[edit]',
            'http://',
            'https://'
        ]
        if any(pattern in assistant.lower() for pattern in bad_patterns):
            return False
        
        return True


# ============================================
# MULTI-SOURCE SCRAPER
# ============================================

class MultiSourceScraper:
    """Aggregate data from multiple sources"""
    
    def __init__(self, language='en'):
        self.sources = {
            'wikipedia': WikipediaScraper(language),
            'reddit': RedditScraper(),
            'stackoverflow': StackOverflowScraper(),
        }
        self.qa_gen = QAGenerator()
        self.quality_filter = QualityFilter()

    def get_mixed_dataset(self, total=100) -> List[Dict]:
        """Get balanced dataset from all sources"""
        print(f"\n📥 Fetching data from multiple sources (target: {total} Q&A)...")
        
        # Distribution: 50% Wikipedia, 30% Reddit, 20% StackOverflow
        wiki_target = int(total * 0.5)
        reddit_target = int(total * 0.3)
        so_target = int(total * 0.2)
        
        all_qa = []
        
        # Wikipedia
        print("📚 Fetching Wikipedia articles...")
        wiki_articles = self.sources['wikipedia'].get_random_articles(wiki_target // 2)
        for article in tqdm(wiki_articles, desc="Wikipedia"):
            data = self.sources['wikipedia'].get_article_content(article['title'])
            if data and data['length'] > 300:
                qa_pairs = self.qa_gen.generate_wikipedia_qa(data['title'], data['content'], max_pairs=2)
                all_qa.extend(qa_pairs)
            time.sleep(0.3)
        
        # Reddit
        print("💬 Fetching Reddit posts...")
        reddit_posts = self.sources['reddit'].get_posts(reddit_target)
        for post in tqdm(reddit_posts, desc="Reddit"):
            qa_pairs = self.qa_gen.generate_reddit_qa(post)
            all_qa.extend(qa_pairs)
            time.sleep(0.5)
        
        # StackOverflow
        print("💻 Fetching StackOverflow Q&A...")
        so_questions = self.sources['stackoverflow'].get_questions(so_target)
        for item in tqdm(so_questions, desc="StackOverflow"):
            qa_pairs = self.qa_gen.generate_stackoverflow_qa(item)
            all_qa.extend(qa_pairs)
            time.sleep(0.5)
        
        # Add some conversational examples
        for q, a in self.qa_gen.conversation_templates:
            all_qa.append({"human": q, "assistant": a, "source": "conversation"})
        
        # Apply quality filter
        print("🔍 Applying quality filters...")
        filtered_qa = self.quality_filter.filter_qa_pairs(all_qa)
        
        # Shuffle for diversity
        random.shuffle(filtered_qa)
        
        # Show statistics
        sources_count = {}
        for qa in filtered_qa:
            source = qa.get('source', 'unknown')
            sources_count[source] = sources_count.get(source, 0) + 1
        
        print(f"\n✅ Dataset ready: {len(filtered_qa)} Q&A pairs")
        print(f"📊 Source distribution: {sources_count}")
        
        return filtered_qa


# ============================================
# DATASET + TRAINING
# ============================================

class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=384):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.valid_pairs = []
        for pair in pairs:
            h = pair['human'].strip()
            a = pair['assistant'].strip()
            if len(h) > 0 and len(a) > 0:
                self.valid_pairs.append((h, a))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        h, a = self.valid_pairs[idx]
        
        # Format: Human: ... Assistant: ...
        text = f"Human: {h}\nAssistant: {a}"
        
        ids = self.tokenizer.encoder(text)
        
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        
        # Calculate where assistant response starts
        prefix = f"Human: {h}\nAssistant:"
        ids_prefix = self.tokenizer.encoder(prefix)
        assist_start = min(len(ids_prefix), len(ids) - 1)
        
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "assist_start": assist_start,
            "length": len(ids)
        }


def collate_fn(batch, pad_id=0):
    input_ids_list = [b["input_ids"] for b in batch]
    assist_starts = [b["assist_start"] for b in batch]
    max_len = max([t.size(0) for t in input_ids_list])
    
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        
        start = assist_starts[i]
        if start < L - 1:
            labels[i, start:L] = input_ids[i, start:L]
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }


# ============================================
# CONTINUOUS TRAINING SYSTEM
# ============================================

class ContinuousTrainer:
    def __init__(self, model_dir, tokenizer_path, device, language='en'):
        self.model_dir = model_dir
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.scraper = MultiSourceScraper(language)
        
        os.makedirs(model_dir, exist_ok=True)
        
        self.model, self.tokenizer, self.config = self._load_or_init_model()
        
        self.history_file = os.path.join(model_dir, "training_history.json")
        self.history = self._load_history()
        
        self.best_loss = self.history.get('best_loss', float('inf'))

    def _load_or_init_model(self):
        cfg_path = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "model.pt")
        
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            # Optimized config for your 5000 vocab
            cfg = {
                "vocab_size": 5000,
                "embed_dim": 320,      # Reduced from 384
                "num_heads": 8,        # Better than 6 for 320 dim
                "num_layers": 6,       # Keep depth
                "max_seq_len": 384,    # Reduced from 512
                "dropout": 0.15        # Increased dropout
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        tokenizer = MYBPE(vocab_size=cfg["vocab_size"])
        tokenizer.load_tokenizer(self.tokenizer_path)
        
        model_kwargs = {
            "vocab_size": cfg["vocab_size"],
            "embed_dim": cfg["embed_dim"],
            "num_heads": cfg["num_heads"],
            "num_layers": cfg["num_layers"],
            "max_seq_len": cfg["max_seq_len"]
        }
        if "dropout" in cfg:
            model_kwargs["dropout"] = cfg["dropout"]
        
        model = GPT2Model(**model_kwargs)
        
        if os.path.exists(model_path):
            print(f"✅ Loading existing model: {model_path}")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print("🆕 Initializing new model")
        
        model.to(self.device)
        print(f"📊 Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        
        return model, tokenizer, cfg

    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"cycles": [], "total_qa_trained": 0, "best_loss": float('inf')}

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def train_on_dataset(self, dataset, epochs=3, batch_size=8, lr=3e-4):
        """Train model on dataset with improved hyperparameters"""
        print("\n" + "="*60)
        print("🚀 TRAINING ON DATASET")
        print("="*60)
        
        if len(dataset) < 10:
            print("⚠️ Dataset too small, skipping...")
            return [0.0]
        
        # 85/15 split
        split = int(len(dataset) * 0.85)
        train_data = dataset[:split]
        val_data = dataset[split:]
        
        train_ds = ChatDataset(train_data, self.tokenizer, max_length=self.config["max_seq_len"])
        val_ds = ChatDataset(val_data, self.tokenizer, max_length=self.config["max_seq_len"])
        
        pad_id = getattr(self.tokenizer, "eos_id", 0)
        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=lambda b: collate_fn(b, pad_id)
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda b: collate_fn(b, pad_id)
        )
        
        # Optimizer with weight decay
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader)*2, T_mult=1)
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        cycle_losses = []
        
        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
            
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                try:
                    logits, _ = self.model(input_ids, attention_mask=attention_mask)
                except TypeError:
                    logits, _ = self.model(input_ids)
                
                lm_logits = logits[:, :-1, :].contiguous()
                lm_labels = labels[:, 1:].contiguous()
                loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}', 
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            cycle_losses.append(avg_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    try:
                        logits, _ = self.model(input_ids, attention_mask=attention_mask)
                    except TypeError:
                        logits, _ = self.model(input_ids)
                    
                    lm_logits = logits[:, :-1, :].contiguous()
                    lm_labels = labels[:, 1:].contiguous()
                    loss = loss_fn(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val = val_loss / val_batches if val_batches > 0 else 0.0
            print(f"Epoch {ep} - Train Loss: {avg_loss:.4f} | Val Loss: {avg_val:.4f}")
            
            # Early stopping check
            if avg_val > avg_loss * 2.5:
                print("⚠️ Overfitting detected, stopping early")
                break
        
        # Save best model
        final_loss = cycle_losses[-1] if cycle_losses else float('inf')
        if final_loss < self.best_loss:
            self.best_loss = final_loss
            model_path = os.path.join(self.model_dir, "model.pt")
            torch.save(self.model.state_dict(), model_path)
            print(f"💾 Best model saved: {model_path} (loss: {final_loss:.4f})")
        
        return cycle_losses

    def run_continuous_training(self, num_cycles=30, qa_per_cycle=100, 
                                epochs=3, batch_size=8, lr=3e-4):
        """Run continuous training loop with multi-source data"""
        print("\n" + "="*70)
        print("🤖 MULTI-SOURCE CONTINUOUS TRAINING")
        print("="*70)
        print(f"📊 Cycles: {num_cycles}")
        print(f"💬 Q&A per cycle: {qa_per_cycle}")
        print(f"🔁 Epochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"📉 Learning rate: {lr}")
        print("="*70)
        
        for cycle in range(1, num_cycles + 1):
            print(f"\n\n{'='*70}")
            print(f"🔄 CYCLE {cycle}/{num_cycles}")
            print(f"{'='*70}")
            
            # Generate mixed dataset
            dataset = self.scraper.get_mixed_dataset(total=qa_per_cycle)
            
            if not dataset or len(dataset) < 10:
                print("⚠️ Insufficient data, skipping cycle...")
                continue
            
            # Train on this dataset
            losses = self.train_on_dataset(dataset, epochs, batch_size, lr)
            
            # Update history
            avg_loss = sum(losses) / len(losses) if losses else 0
            self.history["cycles"].append({
                "cycle": cycle,
                "num_qa": len(dataset),
                "avg_loss": avg_loss,
                "best_loss": self.best_loss,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            self.history["total_qa_trained"] += len(dataset)
            self.history["best_loss"] = self.best_loss
            self._save_history()
            
            print(f"\n✅ Cycle {cycle} complete - Loss: {avg_loss:.4f} | Best: {self.best_loss:.4f}")
            print(f"📊 Total Q&A trained: {self.history['total_qa_trained']}")
            
            # Progressive pause
            if cycle % 5 == 0:
                print(f"⏸️ Cooling down for 15 seconds...")
                time.sleep(15)
        
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print(f"📊 {num_cycles} cycles | Best Loss: {self.best_loss:.4f}")
        print("="*70)


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Source Continuous Training System")
    parser.add_argument("--model-dir", type=str, default="./my_chatbot_multisource")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_model.bin")
    parser.add_argument("--cycles", type=int, default=30,
                       help="Number of training cycles")
    parser.add_argument("--qa-per-cycle", type=int, default=100,
                       help="Target Q&A pairs per cycle (will fetch from multiple sources)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Epochs per cycle")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--language", type=str, default='en',
                       help="Wikipedia language (en/fr/es/de)")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"🖥️ Using device: {device}")
    
    trainer = ContinuousTrainer(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer,
        device=device,
        language=args.language
    )
    
    trainer.run_continuous_training(
        num_cycles=args.cycles,
        qa_per_cycle=args.qa_per_cycle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()