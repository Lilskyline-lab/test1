"""
Fine-Tuning de GPT-2 pour créer un Chatbot
Transforme GPT-2 (completion) en modèle conversationnel
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import json

# ============================================
# ÉTAPE 1 : DATASET DE CONVERSATIONS
# ============================================

class ChatDataset(Dataset):
    """
    Dataset pour l'entraînement de chat
    Format: conversations avec Human/Assistant
    """
    def __init__(self, conversations, tokenizer, max_length=512):
        """
        Args:
            conversations: Liste de dialogues
                [
                    {"human": "Bonjour!", "assistant": "Bonjour! Comment puis-je vous aider?"},
                    {"human": "C'est quoi Python?", "assistant": "Python est un langage..."}
                ]
            tokenizer: GPT2Tokenizer
            max_length: Longueur max des séquences
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokens spéciaux pour structurer le dialogue
        self.human_token = "\n\nHuman: "
        self.assistant_token = "\n\nAssistant: "
        self.eos_token = tokenizer.eos_token
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Formater le dialogue
        text = (
            f"{self.human_token}{conv['human']}"
            f"{self.assistant_token}{conv['assistant']}"
            f"{self.eos_token}"
        )
        
        # Tokenizer
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Labels = input_ids (pour autoregressive training)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ============================================
# ÉTAPE 2 : CRÉER UN DATASET D'EXEMPLE
# ============================================

def create_example_dataset():
    """
    Crée un dataset d'exemple pour le fine-tuning
    Dans la vraie vie, utilisez un dataset plus grand !
    """
    conversations = [
        # Salutations
        {"human": "Bonjour", "assistant": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"},
        {"human": "Salut", "assistant": "Salut ! Que puis-je faire pour vous ?"},
        {"human": "Hey", "assistant": "Hey ! En quoi puis-je vous être utile ?"},
        
        # Questions générales
        {"human": "Comment ça va ?", "assistant": "Je vais bien, merci de demander ! Et vous ?"},
        {"human": "Qui es-tu ?", "assistant": "Je suis un assistant IA conçu pour répondre à vos questions et vous aider."},
        {"human": "Que peux-tu faire ?", "assistant": "Je peux répondre à des questions, donner des informations et avoir des conversations naturelles."},
        
        # Questions techniques
        {"human": "C'est quoi Python ?", "assistant": "Python est un langage de programmation populaire, facile à apprendre et très polyvalent."},
        {"human": "Comment apprendre la programmation ?", "assistant": "Commencez par les bases d'un langage comme Python, pratiquez régulièrement et faites des projets concrets."},
        {"human": "Qu'est-ce que l'IA ?", "assistant": "L'intelligence artificielle est un domaine qui crée des systèmes capables d'effectuer des tâches nécessitant habituellement l'intelligence humaine."},
        
        # Demandes d'aide
        {"human": "Aide-moi", "assistant": "Bien sûr ! Dites-moi ce dont vous avez besoin et je ferai de mon mieux pour vous aider."},
        {"human": "J'ai besoin d'aide", "assistant": "Je suis là pour ça ! Quelle est votre question ?"},
        
        # Au revoir
        {"human": "Merci", "assistant": "De rien ! N'hésitez pas si vous avez d'autres questions."},
        {"human": "Au revoir", "assistant": "Au revoir ! À bientôt !"},
    ]
    
    return conversations


# ============================================
# ÉTAPE 3 : FINE-TUNING
# ============================================

class ChatFineTuner:
    """
    Classe pour fine-tuner GPT-2 en chatbot
    """
    def __init__(
        self,
        model_name='gpt2',
        learning_rate=5e-5,
        batch_size=4,
        num_epochs=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        print("="*60)
        print("🤖 FINE-TUNING GPT-2 POUR CHAT")
        print("="*60)
        
        self.device = device
        self.num_epochs = num_epochs
        
        # Charger le modèle et tokenizer
        print(f"\n📥 Chargement de {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Ajouter un pad token (GPT-2 n'en a pas par défaut)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(device)
        
        print(f"✅ Modèle chargé sur {device}")
        print(f"✅ Paramètres: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
    def train(self, train_dataset):
        """
        Entraîne le modèle sur le dataset de chat
        """
        print("\n" + "="*60)
        print("🏋️  DÉBUT DU FINE-TUNING")
        print("="*60)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True
        )
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch in pbar:
                # Déplacer sur device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f"\n✅ Epoch {epoch+1} - Loss moyenne: {avg_loss:.4f}")
        
        print("\n" + "="*60)
        print("✅ FINE-TUNING TERMINÉ!")
        print("="*60)
    
    def save(self, path='./chatbot_model'):
        """Sauvegarde le modèle fine-tuné"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"\n✅ Modèle sauvegardé dans: {path}")
    
    def chat(self, prompt, max_length=100, temperature=0.7):
        """
        Génère une réponse de chat
        """
        self.model.eval()
        
        # Formater le prompt
        formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
        
        # Encoder
        input_ids = self.tokenizer.encode(
            formatted_prompt,
            return_tensors='pt'
        ).to(self.device)
        
        # Générer
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Décoder
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extraire seulement la réponse de l'assistant
        response = response.split("Assistant:")[-1].strip()
        
        # Couper au premier retour à la ligne (éviter de continuer la conversation)
        response = response.split("\n")[0].strip()
        
        return response


# ============================================
# SCRIPT PRINCIPAL
# ============================================

if __name__ == "__main__":
    
    print("\n🚀 CRÉATION D'UN CHATBOT GPT-2\n")
    
    # 1. Créer le dataset
    print("📚 Création du dataset...")
    conversations = create_example_dataset()
    print(f"✅ {len(conversations)} conversations créées")
    
    # 2. Initialiser
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = ChatDataset(conversations, tokenizer)
    print(f"✅ Dataset préparé: {len(dataset)} exemples")
    
    # 3. Fine-tuner
    trainer = ChatFineTuner(
        model_name='gpt2',
        learning_rate=5e-5,
        num_epochs=5  # Plus d'epochs pour bien apprendre
    )
    
    trainer.train(dataset)
    
    # 4. Sauvegarder
    trainer.save('./my_chatbot')
    
    # 5. Tester en mode interactif
    print("\n" + "="*60)
    print("💬 MODE CHAT INTERACTIF")
    print("="*60)
    print("\n💡 Le modèle est maintenant un chatbot!")
    print("💡 Tapez 'quit' pour quitter\n")
    
    while True:
        user_input = input("Vous: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Au revoir!\n")
            break
        
        if not user_input.strip():
            continue
        
        response = trainer.chat(user_input)
        print(f"Bot: {response}\n")
    
    print("="*60)
    print("✅ Session terminée!")
    print("="*60)
    print("\n📁 Modèle sauvegardé dans: ./my_chatbot/")
    print("💡 Pour le recharger plus tard:")
    print("   model = GPT2LMHeadModel.from_pretrained('./my_chatbot')")
    print("   tokenizer = GPT2Tokenizer.from_pretrained('./my_chatbot')")
    print("="*60)