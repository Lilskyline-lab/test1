"""
Système d'entraînement continu AMÉLIORÉ
- Multi-sources (Wikipedia, Wikihow, Simple Wikipedia)
- Tokenizer 5000 vocab optimisé
- Dataset de qualité
"""

import os
import sys
import json
import time
import requests
import re
from tqdm import tqdm
from typing import List, Dict
from bs4 import BeautifulSoup

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Imports locaux
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from Tokenizer import MYBPE


# ============================================
# MULTI-SOURCE SCRAPER
# ============================================

class WikipediaScraper:
    """Scraper Wikipedia (source principale)"""
    def __init__(self, language='fr'):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "EduBot/1.0"}

    def get_random_articles(self, count=10):
        print(f"\n📚 Wikipedia: {count} articles...")
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': min(count, 50)  # Max 50 par requête
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [{"title": a["title"], "id": a["id"]} for a in data["query"]["random"]]
        except Exception as e:
            print(f"⚠️ Erreur: {e}")
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
            data = response.json()
            page = list(data['query']['pages'].values())[0]
            if 'extract' not in page:
                return None
            text = self._clean_text(page['extract'])
            if len(text) < 200:
                return None
            return {'title': title, 'content': text}
        except:
            return None

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'==+ .*? ==+', '', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()


class WikiHowScraper:
    """Scraper WikiHow (instructions pratiques)"""
    def __init__(self):
        self.base_url = "https://fr.wikihow.com"
        self.headers = {"User-Agent": "EduBot/1.0"}
    
    def get_random_articles(self, count=5):
        print(f"\n🛠️ WikiHow: {count} articles...")
        articles = []
        for _ in range(count):
            try:
                response = requests.get(f"{self.base_url}/Special:Randomizer", 
                                      headers=self.headers, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title_elem = soup.find('h1', class_='firstHeading')
                    if title_elem:
                        articles.append({'title': title_elem.text.strip(), 'url': response.url})
                time.sleep(1)  # Rate limiting
            except:
                continue
        return articles
    
    def get_article_content(self, article_info: Dict) -> Dict:
        try:
            response = requests.get(article_info['url'], headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            steps = soup.find_all('div', class_='step')
            content = []
            for step in steps[:10]:  # Max 10 étapes
                text = step.get_text(strip=True)
                if len(text) > 50:
                    content.append(text)
            
            if content:
                return {
                    'title': article_info['title'],
                    'content': '\n'.join(content)
                }
        except:
            pass
        return None


class SimpleWikiScraper:
    """Simple Wikipedia (texte plus simple)"""
    def __init__(self, language='simple'):
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": "EduBot/1.0"}
    
    def get_random_articles(self, count=10):
        print(f"\n📖 Simple Wiki: {count} articles...")
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': min(count, 50)
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            return [{"title": a["title"], "id": a["id"]} for a in data["query"]["random"]]
        except:
            return []
    
    def get_article_content(self, title: str) -> Dict:
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True
        }
        try:
            response = requests.get(self.api_url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            page = list(data['query']['pages'].values())[0]
            if 'extract' in page:
                text = page['extract'].strip()
                if len(text) > 100:
                    return {'title': title, 'content': text}
        except:
            pass
        return None


# ============================================
# Q&A GENERATOR AMÉLIORÉ
# ============================================

class EnhancedQAGenerator:
    """Générateur Q&A avec plus de diversité"""
    
    def __init__(self):
        self.question_templates = {
            'definition': [
                "Qu'est-ce que {s} ?",
                "Définis {s}.",
                "C'est quoi {s} ?",
                "Explique-moi {s}.",
            ],
            'how': [
                "Comment fonctionne {s} ?",
                "Comment faire {s} ?",
                "Explique comment {s}.",
            ],
            'why': [
                "Pourquoi {s} ?",
                "Pourquoi {s} est important ?",
            ],
            'general': [
                "Parle-moi de {s}.",
                "Que sais-tu sur {s} ?",
                "Décris {s}.",
                "Donne-moi des infos sur {s}.",
            ]
        }
        
        # Conversations basiques (pour le côté chat)
        self.chat_pairs = [
            ("Bonjour", "Bonjour ! Comment vas-tu ?"),
            ("Salut", "Salut ! Ravi de te revoir !"),
            ("Ça va ?", "Oui très bien, et toi ?"),
            ("Comment tu t'appelles ?", "Je suis un assistant IA."),
            ("Merci", "De rien ! 😊"),
            ("Au revoir", "Au revoir ! À bientôt !"),
            ("Aide-moi", "Bien sûr ! Que puis-je faire pour toi ?"),
            ("C'est quoi l'IA ?", "L'intelligence artificielle est un domaine qui crée des systèmes intelligents."),
        ]
    
    def _truncate_clean(self, text: str, max_len=400):
        """Tronque intelligemment"""
        if len(text) <= max_len:
            return text.strip()
        
        truncated = text[:max_len]
        # Trouver la fin de phrase
        for char in ['. ', '! ', '? ', '\n']:
            idx = truncated.rfind(char)
            if idx > max_len // 2:  # Au moins 50% du texte
                return truncated[:idx + 1].strip()
        
        return truncated.strip() + "..."
    
    def generate_qa_pairs(self, title: str, content: str, source_type='wikipedia') -> List[Dict]:
        """Génère des Q&A depuis un article"""
        qa_pairs = []
        
        # Découper en paragraphes
        if source_type == 'wikihow':
            paragraphs = content.split('\n')
        else:
            paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 80]
        
        # Limiter le nombre de paragraphes
        paragraphs = paragraphs[:5]
        
        # Générer plusieurs types de questions
        for i, para in enumerate(paragraphs):
            if len(para) < 50:
                continue
            
            # Choisir un template
            if i == 0:
                templates = self.question_templates['definition']
            elif 'comment' in para.lower() or source_type == 'wikihow':
                templates = self.question_templates['how']
            elif 'pourquoi' in para.lower():
                templates = self.question_templates['why']
            else:
                templates = self.question_templates['general']
            
            question = templates[i % len(templates)].format(s=title)
            answer = self._truncate_clean(para, 400)
            
            qa_pairs.append({
                'human': question,
                'assistant': answer,
                'source': title,
                'type': source_type
            })
        
        return qa_pairs
    
    def get_chat_pairs(self) -> List[Dict]:
        """Retourne les paires de chat basiques"""
        return [{'human': h, 'assistant': a} for h, a in self.chat_pairs]


# ============================================
# DATASET + TRAINING (inchangé)
# ============================================

class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        h = self.pairs[idx]['human'].strip()
        a = self.pairs[idx]['assistant'].strip()
        
        # Encode silencieusement
        import io
        import sys as sys_mod
        old_stdout = sys_mod.stdout
        sys_mod.stdout = io.StringIO()
        
        prefix = f"Human: {h}\nBot:"
        text = prefix + " " + a
        
        ids_all = self.tokenizer.encoder(text)
        sys_mod.stdout = old_stdout
        
        # Tronquer si trop long
        if len(ids_all) > self.max_length:
            ids_all = ids_all[:self.max_length]
        
        return {"input_ids": torch.tensor(ids_all, dtype=torch.long)}


def collate_fn(batch, pad_id=0):
    input_ids_list = [b["input_ids"] for b in batch]
    max_len = max([t.size(0) for t in input_ids_list])
    
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, ids in enumerate(input_ids_list):
        L = ids.size(0)
        input_ids[i, :L] = ids
        labels[i, :L] = ids
    
    return {"input_ids": input_ids, "labels": labels}


# ============================================
# CONTINUOUS TRAINER AMÉLIORÉ
# ============================================

class EnhancedContinuousTrainer:
    def __init__(self, model_dir, tokenizer_path, device, language='fr'):
        self.model_dir = model_dir
        self.device = device
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Scrapers multiples
        self.scrapers = {
            'wikipedia': WikipediaScraper(language),
            'wikihow': WikiHowScraper(),
            'simple_wiki': SimpleWikiScraper()
        }
        
        self.qa_gen = EnhancedQAGenerator()
        
        # Charger modèle
        self.model, self.tokenizer, self.config = self._load_or_init_model(tokenizer_path)
        
        # Historique
        self.history_file = os.path.join(model_dir, "training_history.json")
        self.history = self._load_history()
    
    def _load_or_init_model(self, tokenizer_path):
        cfg_path = os.path.join(self.model_dir, "config.json")
        model_path = os.path.join(self.model_dir, "model.pt")
        
        # Config
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        else:
            cfg = {
                "vocab_size": 5000,  # ← 5000 vocab !
                "embed_dim": 256,
                "num_heads": 8,
                "num_layers": 4,
                "max_seq_len": 256
            }
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        # Tokenizer
        tokenizer = MYBPE(vocab_size=cfg["vocab_size"])
        tokenizer.load_tokenizer(tokenizer_path)
        print(f"✅ Tokenizer chargé: {cfg['vocab_size']} vocab")
        
        # Modèle
        model = GPT2Model(
            vocab_size=cfg["vocab_size"],
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"]
        )
        
        if os.path.exists(model_path):
            print(f"✅ Chargement modèle existant")
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            print("🆕 Nouveau modèle initialisé")
        
        model.to(self.device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Modèle: {num_params:,} paramètres")
        
        return model, tokenizer, cfg
    
    def _load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"cycles": [], "total_qa": 0}
    
    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def generate_mixed_dataset(self, total_articles=30):
        """Génère un dataset mixte depuis plusieurs sources"""
        print("\n" + "="*60)
        print("🔄 GÉNÉRATION DATASET MIXTE")
        print("="*60)
        
        dataset = []
        
        # 1. Wikipedia FR (60%)
        wiki_count = int(total_articles * 0.6)
        wiki_articles = self.scrapers['wikipedia'].get_random_articles(wiki_count)
        for art in tqdm(wiki_articles, desc="Wikipedia"):
            data = self.scrapers['wikipedia'].get_article_content(art['title'])
            if data:
                qa = self.qa_gen.generate_qa_pairs(data['title'], data['content'], 'wikipedia')
                dataset.extend(qa)
            time.sleep(0.3)
        
        # 2. WikiHow (20%)
        how_count = int(total_articles * 0.2)
        how_articles = self.scrapers['wikihow'].get_random_articles(how_count)
        for art in tqdm(how_articles, desc="WikiHow"):
            data = self.scrapers['wikihow'].get_article_content(art)
            if data:
                qa = self.qa_gen.generate_qa_pairs(data['title'], data['content'], 'wikihow')
                dataset.extend(qa)
            time.sleep(1)
        
        # 3. Simple Wiki (20%)
        simple_count = int(total_articles * 0.2)
        simple_articles = self.scrapers['simple_wiki'].get_random_articles(simple_count)
        for art in tqdm(simple_articles, desc="Simple Wiki"):
            data = self.scrapers['simple_wiki'].get_article_content(art['title'])
            if data:
                qa = self.qa_gen.generate_qa_pairs(data['title'], data['content'], 'simple_wiki')
                dataset.extend(qa)
            time.sleep(0.3)
        
        # 4. Ajouter conversations basiques (10 paires)
        dataset.extend(self.qa_gen.get_chat_pairs())
        
        print(f"\n✅ Dataset généré: {len(dataset)} paires Q&A")
        return dataset
    
    def train_on_dataset(self, dataset, epochs=2, batch_size=8, lr=3e-4):
        """Entraîne le modèle"""
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT")
        print("="*60)
        
        if not dataset:
            print("⚠️ Dataset vide!")
            return []
        
        train_ds = ChatDataset(dataset, self.tokenizer, max_length=self.config["max_seq_len"])
        pad_id = 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_fn(b, pad_id))
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        
        losses = []
        
        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
            
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits, _ = self.model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch {ep} - Loss: {avg_loss:.4f}")
        
        # Sauvegarder
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model.pt"))
        print("💾 Modèle sauvegardé")
        
        return losses
    
    def run_training(self, num_cycles=5, articles_per_cycle=30, epochs=2, batch_size=8, lr=3e-4):
        """Lance l'entraînement continu"""
        print("\n" + "="*70)
        print("🤖 ENTRAÎNEMENT CONTINU MULTI-SOURCES")
        print("="*70)
        print(f"📊 Cycles: {num_cycles}")
        print(f"📚 Articles/cycle: {articles_per_cycle}")
        print(f"🔁 Epochs: {epochs}")
        print("="*70)
        
        for cycle in range(1, num_cycles + 1):
            print(f"\n\n{'='*70}")
            print(f"🔄 CYCLE {cycle}/{num_cycles}")
            print(f"{'='*70}")
            
            # Générer dataset
            dataset = self.generate_mixed_dataset(articles_per_cycle)
            
            if not dataset:
                print("⚠️ Aucune donnée, skip")
                continue
            
            # Entraîner
            losses = self.train_on_dataset(dataset, epochs, batch_size, lr)
            
            # Historique
            self.history["cycles"].append({
                "cycle": cycle,
                "num_qa": len(dataset),
                "avg_loss": sum(losses) / len(losses) if losses else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            self.history["total_qa"] += len(dataset)
            self._save_history()
            
            print(f"\n✅ Cycle {cycle} terminé!")
            print(f"📊 Total Q&A: {self.history['total_qa']}")
        
        print("\n" + "="*70)
        print("🎉 ENTRAÎNEMENT TERMINÉ!")
        print(f"📊 {num_cycles} cycles, {self.history['total_qa']} Q&A")
        print("="*70)


# ============================================
# MAIN
# ============================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./my_5k_chatbot")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_5000.bin")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--articles", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--language", type=str, default='fr')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    trainer = EnhancedContinuousTrainer(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer,
        device=torch.device(args.device),
        language=args.language
    )
    
    trainer.run_training(
        num_cycles=args.cycles,
        articles_per_cycle=args.articles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()