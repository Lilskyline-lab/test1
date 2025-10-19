import torch
import sys
import os

# Désactiver les barres de progression tqdm du tokenizer
os.environ['TQDM_DISABLE'] = '1'

sys.path.append('./Tokenizer')
sys.path.append('./Model')

from Tokenizer import MYBPE
from gpt2_model import GPT2Model

print("\n" + "="*60)
print("🎉 TEST DE GÉNÉRATION GPT-2")
print("="*60)

# 1. Charger le tokenizer
print("\n📦 Chargement des composants...")
tokenizer = MYBPE(vocab_size=300)
tokenizer.load_tokenizer("Tokenizer/tokenizer_model.bin")

# 2. Créer le modèle
model = GPT2Model(
    vocab_size=300,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=64
)

# 3. Charger les poids
model.load_state_dict(torch.load('./checkpoints/final_model.pt'))
model.eval()

print("✅ Tokenizer chargé")
print("✅ Modèle chargé (~1M paramètres)")

# 4. GÉNÉRER DU TEXTE
prompts = [
    "Once upon a time",
    "The little girl",
    "One day",
]

temperatures = [0.5, 0.8, 1.2]

for i, prompt in enumerate(prompts, 1):
    print("\n" + "="*60)
    print(f"TEST {i}/{len(prompts)}")
    print("="*60)
    print(f"📝 Prompt: '{prompt}'\n")
    
    # Encoder (silencieux)
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    tokens = tokenizer.encoder(prompt)
    input_ids = torch.tensor([tokens])
    
    sys.stdout = old_stdout
    
    # Générer avec différentes températures
    for temp in temperatures:
        print(f"🌡️  Temperature = {temp}")
        
        # Génération silencieuse
        sys.stdout = io.StringIO()
        generated = model.generate(
            input_ids,
            max_new_tokens=50,
            temperature=temp,
            top_k=30
        )
        text = tokenizer.decoder(generated[0].tolist())
        sys.stdout = old_stdout
        
        print(f"✨ {text}")
        print("-"*60 + "\n")

print("="*60)
print("✅ Tests terminés!")
print("="*60 + "\n")