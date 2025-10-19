#!/usr/bin/env python3
"""
Script d'entraînement GPT-2 "mini-large" (~20M paramètres)
Optimisé pour CPU / Codespaces
Durée estimée: 3–5 h sur CPU standard
"""

import torch
import sys
import os
from pathlib import Path
import time

# Ajout des chemins locaux
sys.path.append('./Tokenizer')
sys.path.append('./Training')
sys.path.append('./Model')

from Tokenizer import MYBPE
from training import TextDataset, GPT2Trainer
from gpt2_model import GPT2Model

print("="*60)
print("🚀 ENTRAÎNEMENT GPT-2 (≈20M paramètres) sur CPU")
print("="*60)

# Configuration "mini-large"
CONFIG = {
    'vocab_size': 300,
    'embed_dim': 256,      # ↑ Taille d'embedding
    'num_heads': 8,        # ↑ Nombre de têtes d'attention
    'num_layers': 6,       # ↑ Nombre de blocs Transformer
    'max_seq_len': 64,
    'batch_size': 2,       # CPU: petit batch
    'num_epochs': 3,
    'learning_rate': 3e-4,
}

print("\n⚙️  Configuration:")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# 1. Dataset
print("\n📥 Téléchargement du dataset...")
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/train.txt"):
    print("Téléchargement de TinyStories...")
    import urllib.request
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    urllib.request.urlretrieve(url, "data/train.txt")
    print("✓ Dataset téléchargé")
else:
    print("✓ Dataset déjà présent")

# 2. Tokenizer
print("\n🔤 Tokenizer...")
tokenizer = MYBPE(vocab_size=CONFIG['vocab_size'])

try:
    tokenizer.load_tokenizer("Tokenizer/tokenizer_model.bin")
    print("✓ Tokenizer chargé")
except:
    print("⚠️  Entraînement d'un tokenizer...")
    with open("data/train.txt", "r") as f:
        sample = f.read(1_000_000)
    tokenizer = MYBPE(vocab_size=CONFIG['vocab_size'], dataset=sample)
    tokenizer.train_tokenizer()
    tokenizer.build_vocabulary()
    tokenizer.save_tokenizer("Tokenizer/tokenizer_model.bin")
    print("✓ Tokenizer entraîné")

# 3. Dataset réduit (pour CPU)
print("\n📚 Préparation du dataset...")
with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read(4_000_000)  # 4MB ≈ suffisant pour test CPU

train_dataset = TextDataset(text, tokenizer, seq_len=CONFIG['max_seq_len'])
print(f"✓ {len(train_dataset)} séquences prêtes")

# 4. Modèle GPT-2 (20M)
print("\n🤖 Création du modèle...")
model = GPT2Model(
    vocab_size=CONFIG['vocab_size'],
    embed_dim=CONFIG['embed_dim'],
    num_heads=CONFIG['num_heads'],
    num_layers=CONFIG['num_layers'],
    max_seq_len=CONFIG['max_seq_len']
)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modèle initialisé: {num_params:,} paramètres ({num_params/1e6:.1f}M)")

# 5. Sauvegardes
os.makedirs("checkpoints", exist_ok=True)

# 6. Trainer
print("\n🏋️  Initialisation du Trainer...")
trainer = GPT2Trainer(
    model=model,
    train_dataset=train_dataset,
    learning_rate=CONFIG['learning_rate'],
    batch_size=CONFIG['batch_size'],
    num_epochs=CONFIG['num_epochs'],
    device='cpu',
    checkpoint_dir='./checkpoints'
)

# 7. Entraînement
print("\n" + "="*60)
print("🚀 DÉBUT DE L'ENTRAÎNEMENT (CPU)")
print("="*60)
print("⏱️  Estimé: 3–5 heures")
print("💡 Astuce: Fermez tout ce qui consomme CPU dans Codespaces.")
print("="*60 + "\n")

start = time.time()

try:
    trainer.train(save_every=1)
except KeyboardInterrupt:
    print("\n⚠️  Interrompu par l'utilisateur")
except Exception as e:
    print(f"\n❌ Erreur: {e}")

elapsed = time.time() - start

# 8. Test de génération
print("\n" + "="*60)
print("🎉 TEST DE GÉNÉRATION")
print("="*60)

model.eval()
prompt = "Once upon a time"
print(f"\nPrompt: '{prompt}'")

tokens = tokenizer.encoder(prompt)
input_ids = torch.tensor([tokens])

with torch.no_grad():
    generated = model.generate(input_ids, max_new_tokens=30, temperature=0.8)

text = tokenizer.decoder(generated[0].tolist())
print(f"\nGénéré:\n{text}\n")

# 9. Statistiques
print("="*60)
print("📊 STATISTIQUES")
print("="*60)
print(f"✓ Paramètres: {num_params:,}")
print(f"✓ Séquences: {len(train_dataset)}")
print(f"✓ Epochs: {CONFIG['num_epochs']}")
if trainer.train_losses:
    print(f"✓ Loss initiale: {trainer.train_losses[0]:.4f}")
    print(f"✓ Loss finale: {trainer.train_losses[-1]:.4f}")
print(f"✓ Temps: {elapsed/60:.1f} minutes")
print("="*60)

# 10. Sauvegarde finale
final_path = './checkpoints/final_model.pt'
torch.save(model.state_dict(), final_path)
print(f"\n✓ Modèle sauvegardé: {final_path}")

print("\n" + "="*60)
print("✅ TERMINÉ!")
print("="*60)
print("\n💡 Pour un entraînement sérieux:")
print("   → Passe sur Google Colab (GPU)")
print("   → GPU = 50–100× plus rapide qu’un CPU Codespaces")
print("="*60)
