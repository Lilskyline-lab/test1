"""
Fine-Tuning ULTRA-LÉGER pour CPU
- Utilise VOTRE modèle local (pas Hugging Face)
- Dataset réduit (100 conversations)
- Optimisé pour vieux PC
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import sys
from tqdm import tqdm
import os

# Importer VOTRE modèle
sys.path.append('../Model')
sys.path.append('../Tokenizer')

from gpt2_model import GPT2Model
from Tokenizer import MYBPE

print("="*60)
print("🚀 FINE-TUNING ULTRA-LÉGER (CPU)")
print("="*60)
print("\n⚠️  Version optimisée pour vieux PC!")
print("📊 Modèle: VOTRE GPT-2 local")
print("📊 Dataset: 100 conversations seulement")
print("📊 Device: CPU\n")

# ============================================
# CONFIGURATION MINI
# ============================================

CONFIG = {
    # Votre modèle (PETIT!)
    'vocab_size': 300,
    'embed_dim': 128,       # TRÈS petit
    'num_heads': 4,
    'num_layers': 2,        # Seulement 2 layers
    'max_seq_len': 64,      # Séquences courtes
    
    # Entraînement
    'batch_size': 4,        # 1 seul exemple à la fois
    'num_epochs': 12,      # Peu d'epochs
    'learning_rate': 3e-4,
    
    # Dataset
    'max_conversations': 100000,  # ← 1K au lieu de 100
         
    # Sauvegarde
    'save_dir': './my_tiny_chatbot',
}

# ============================================
# DATASET SIMPLIFIÉ
# ============================================

class SimpleChatDataset(Dataset):
    """Dataset ultra-simplifié pour CPU"""
    
    def __init__(self, conversations, tokenizer, max_length=64):
        # Limiter le nombre de conversations
        self.conversations = conversations[:CONFIG['max_conversations']]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"✅ Dataset: {len(self.conversations)} conversations")
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Format simple
        text = f"Human: {conv['human']}\nBot: {conv['assistant']}"
        
        # Encoder (sans les barres de progression)
        import io
        import sys as sys_module
        old_stdout = sys_module.stdout
        sys_module.stdout = io.StringIO()
        
        tokens = self.tokenizer.encoder(text)
        
        sys_module.stdout = old_stdout
        
        # Tronquer si trop long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Padding
        while len(tokens) < self.max_length:
            tokens.append(0)  # Pad avec 0
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, targets

# ============================================
# FINE-TUNING
# ============================================

def train():
    # 1. Charger VOTRE tokenizer
    print("\n🔤 Chargement du tokenizer...")
    tokenizer = MYBPE(vocab_size=CONFIG['vocab_size'])
    tokenizer.load_tokenizer("../Tokenizer/tokenizer_model.bin")
    print("✅ Tokenizer chargé")
    
    # 2. Charger les conversations
    print("\n📚 Chargement des conversations...")
    with open('train_conversations.json', 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Filtrer les conversations courtes (plus facile à apprendre)
    conversations = [
        c for c in conversations 
        if len(c['human']) < 200 and len(c['assistant']) < 200
    ]
    
    print(f"✅ {len(conversations)} conversations chargées")
    print(f"💡 On va utiliser seulement les {CONFIG['max_conversations']} premières")
    
    # 3. Créer le dataset
    dataset = SimpleChatDataset(conversations, tokenizer, CONFIG['max_seq_len'])
    
    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    print(f"✅ Train: {train_size} | Val: {val_size}")
    
    # 4. Créer VOTRE modèle (PETIT!)
    print("\n🤖 Création du modèle...")
    model = GPT2Model(
        vocab_size=CONFIG['vocab_size'],
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        max_seq_len=CONFIG['max_seq_len']
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modèle créé: {num_params:,} paramètres")
    print(f"   (BEAUCOUP plus petit que GPT-2 officiel!)")
    
    # 5. Loss et optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 6. ENTRAÎNER
    print("\n" + "="*60)
    print(f"🏋️  DÉBUT DU FINE-TUNING ({CONFIG['num_epochs']} epochs)")
    print("="*60)
    print("\n⏱️  Temps estimé: 30-60 minutes sur CPU\n")
    
    for epoch in range(CONFIG['num_epochs']):
        # TRAIN
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for input_ids, targets in pbar:
            # Forward
            logits, _ = model(input_ids)
            
            # Reshape pour la loss
            loss = criterion(
                logits.view(-1, CONFIG['vocab_size']),
                targets.view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for input_ids, targets in val_loader:
                logits, _ = model(input_ids)
                loss = criterion(
                    logits.view(-1, CONFIG['vocab_size']),
                    targets.view(-1)
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"\n✅ Epoch {epoch+1}:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f}")
    
    # 7. Sauvegarder
    print(f"\n💾 Sauvegarde du modèle...")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    torch.save(model.state_dict(), f"{CONFIG['save_dir']}/model.pt")
    
    # Sauvegarder aussi la config
    with open(f"{CONFIG['save_dir']}/config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"✅ Modèle sauvegardé dans {CONFIG['save_dir']}/")
    
    return model, tokenizer

# ============================================
# TEST INTERACTIF
# ============================================

def chat(model, tokenizer):
    """Mode chat simple"""
    print("\n" + "="*60)
    print("💬 MODE CHAT")
    print("="*60)
    print("\n⚠️  Modèle TRÈS petit, résultats limités!")
    print("💡 Tapez 'quit' pour quitter\n")
    
    model.eval()
    
    while True:
        user_input = input("Vous: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input.strip():
            continue
        
        # Format
        prompt = f"Human: {user_input}\nBot:"
        
        # Encoder (silencieux)
        import io
        import sys as sys_module
        old_stdout = sys_module.stdout
        sys_module.stdout = io.StringIO()
        
        tokens = tokenizer.encoder(prompt)
        
        sys_module.stdout = old_stdout
        
        input_ids = torch.tensor([tokens])
        
        # Générer
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8
            )
        
        # Décoder (silencieux)
        sys_module.stdout = io.StringIO()
        text = tokenizer.decoder(generated[0].tolist())
        sys_module.stdout = old_stdout
        
        # Extraire la réponse
        if "Bot:" in text:
            response = text.split("Bot:")[-1].strip()
        else:
            response = text
        
        print(f"Bot: {response}\n")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n💡 Ce script utilise:")
    print("   - VOTRE modèle GPT-2 local (pas Hugging Face)")
    print("   - VOTRE tokenizer BPE")
    print("   - Les conversations OpenAssistant (locales)")
    print("\n⏱️  Patience... Le CPU est lent!\n")
    
    try:
        # Fine-tuner
        model, tokenizer = train()
        
        print("\n" + "="*60)
        print("✅ FINE-TUNING TERMINÉ!")
        print("="*60)
        
        # Tester
        chat(model, tokenizer)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompu par l'utilisateur")
    
    print("\n👋 Au revoir!")