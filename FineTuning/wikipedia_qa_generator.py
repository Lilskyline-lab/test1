"""
Bot de génération automatique de Dataset Q&A depuis Wikipedia
Scrape Wikipedia → Transforme en paires Question/Réponse propres
"""

import requests
import json
import re
import time
from typing import List, Dict
from tqdm import tqdm
import argparse

# ============================================
# WIKIPEDIA SCRAPER
# ============================================

class WikipediaScraper:
    """Scrape des articles Wikipedia"""

    def __init__(self, language='fr'):
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.headers = {
            "User-Agent": "WikiQABot/1.0 (https://github.com/yourname; contact@example.com)"
        }

    def get_random_articles(self, count=10):
        """Récupère N articles aléatoires"""
        print(f"\n📥 Récupération de {count} articles Wikipedia aléatoires...")

        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnnamespace': 0,
            'rnlimit': count
        }

        try:
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return [{"title": a["title"], "id": a["id"]} for a in data["query"]["random"]]
        except requests.RequestException as e:
            print(f"⚠️  Erreur réseau : {e}")
            return []

    def get_article_content(self, title: str) -> Dict:
        """Récupère et nettoie le contenu d'un article"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }

        try:
            response = requests.get(self.api_url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            page = list(data['query']['pages'].values())[0]

            if 'extract' not in page:
                return None

            text = self._clean_text(page['extract'])
            return {'title': title, 'content': text, 'length': len(text)}

        except requests.RequestException as e:
            print(f"⚠️  Erreur récupération article '{title}': {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte Wikipedia"""
        text = re.sub(r'\[\d+\]', '', text)  # [1], [2], etc.
        text = re.sub(r'==+ .*? ==+', '', text)  # En-têtes de sections
        text = re.sub(r'\n{2,}', '\n', text)  # Sauts de ligne multiples
        text = re.sub(r'\s{2,}', ' ', text)  # Espaces multiples
        text = text.strip()
        return text


# ============================================
# Q&A GENERATOR
# ============================================

class QAGenerator:
    """Génère des paires Q&A plus variées (description + conversation + général)"""

    def __init__(self):
        # Templates Wikipédia classiques
        self.wiki_templates = [
            "Qu'est-ce que {subject} ?",
            "Parle-moi de {subject}.",
            "Explique-moi {subject}.",
            "Que sais-tu sur {subject} ?",
            "Décris {subject}.",
        ]

        # Nouvelles catégories conversationnelles
        self.conversation_templates = [
            ("Bonjour", "Bonjour ! Comment vas-tu aujourd’hui ? 😊"),
            ("Salut", "Salut ! Heureux de te revoir !"),
            ("Comment ça va ?", "Je vais super bien, merci ! Et toi ?"),
            ("Merci", "Avec plaisir 😄"),
            ("Bonne nuit", "Bonne nuit 🌙 fais de beaux rêves !"),
            ("Tu es qui ?", "Je suis un assistant conçu pour t’aider à apprendre et à discuter."),
        ]

        # Catégories explicatives générales
        self.logic_templates = [
            ("Pourquoi le ciel est bleu ?", "Le ciel est bleu à cause de la diffusion de la lumière solaire par l’atmosphère. Les courtes longueurs d’onde, comme le bleu, se dispersent plus facilement."),
            ("Combien de continents y a-t-il ?", "Il y a généralement sept continents : Afrique, Amérique du Nord, Amérique du Sud, Antarctique, Asie, Europe et Océanie."),
            ("Quelle est la vitesse de la lumière ?", "La lumière voyage à environ 299 792 kilomètres par seconde dans le vide."),
        ]

    def _truncate_sentence(self, text: str, max_len=500):
        """Coupe le texte à la fin d'une phrase complète."""
        if len(text) <= max_len:
            return text.strip()
        truncated = text[:max_len]
        end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if end != -1:
            truncated = truncated[:end + 1]
        return truncated.strip()

    def generate_qa_pairs(self, title: str, content: str, max_pairs=3) -> List[Dict]:
        """Génère des Q&A enrichies"""
        qa_pairs = []

        # --------------------------
        # 1️⃣ Génération Wikipédia
        # --------------------------
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 100]
        for i, paragraph in enumerate(paragraphs[:max_pairs]):
            question = self.wiki_templates[i % len(self.wiki_templates)].format(subject=title)
            answer = self._truncate_sentence(paragraph, 600)
            qa_pairs.append({
                "human": question,
                "assistant": answer,
                "source": title,
                "type": "description"
            })

        # --------------------------
        # 2️⃣ Ajouter des Q&A conversationnelles
        # --------------------------
        for q, a in self.conversation_templates:
            qa_pairs.append({
                "human": q,
                "assistant": a,
                "source": "conversation",
                "type": "chat"
            })

        # --------------------------
        # 3️⃣ Ajouter des Q&A logiques / générales
        # --------------------------
        for q, a in self.logic_templates:
            qa_pairs.append({
                "human": q,
                "assistant": a,
                "source": "connaissance_générale",
                "type": "general"
            })

        return qa_pairs

# ============================================
# DATASET BUILDER
# ============================================

class DatasetBuilder:
    def __init__(self, language='fr'):
        self.scraper = WikipediaScraper(language)
        self.qa_generator = QAGenerator()
        self.dataset = []

    def build_dataset(self, num_articles=10, qa_per_article=3):
        print("="*60)
        print("🤖 GÉNÉRATION AUTOMATIQUE DE DATASET Q&A")
        print("="*60)

        articles = self.scraper.get_random_articles(num_articles)
        print(f"🔄 Génération des Q&A à partir de {len(articles)} articles...\n")

        for article in tqdm(articles, desc="Articles"):
            data = self.scraper.get_article_content(article['title'])
            if not data or data['length'] < 200:
                continue
            qa_pairs = self.qa_generator.generate_qa_pairs(
                data['title'], data['content'], max_pairs=qa_per_article
            )
            self.dataset.extend(qa_pairs)
            time.sleep(0.5)

        print(f"\n✅ Dataset généré : {len(self.dataset)} paires Q&A\n")
        return self.dataset

    def save_dataset(self, filepath='wikipedia_qa_dataset.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        print(f"💾 Dataset sauvegardé : {filepath}")

    def show_examples(self, n=3):
        print("\n📝 EXEMPLES DU DATASET\n" + "="*60)
        for i, ex in enumerate(self.dataset[:n], 1):
            print(f"\n--- Exemple {i} ---")
            print(f"Q: {ex['human']}")
            print(f"R: {ex['assistant']}\n")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Générateur de dataset Q&A Wikipédia")
    parser.add_argument('--language', default='fr', help="Langue de Wikipédia ('fr' ou 'en')")
    parser.add_argument('--articles', type=int, default=10, help="Nombre d'articles à scraper")
    parser.add_argument('--qa', type=int, default=3, help="Nombre de Q&A par article")
    args = parser.parse_args()

    builder = DatasetBuilder(language=args.language)
    dataset = builder.build_dataset(num_articles=args.articles, qa_per_article=args.qa)
    builder.show_examples(n=3)
    builder.save_dataset(f"wikipedia_qa_{args.language}.json")

    print("\n✅ GÉNÉRATION TERMINÉE !")


if __name__ == "__main__":
    main()
