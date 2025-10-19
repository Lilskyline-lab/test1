import torch
import sys
import os
import io

# Désactiver les sorties verboses
os.environ['TQDM_DISABLE'] = '1'

sys.path.append('./Tokenizer')
sys.path.append('./Model')

from Tokenizer import MYBPE
from gpt2_model import GPT2Model

def silent_operation(func, *args, **kwargs):
    """Exécute une fonction sans afficher les logs"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return result

print("\n" + "="*60)
print("🎮 GPT-2 - MODE INTERACTIF")
print("="*60)

# Charger le tokenizer (silencieux)
print("\n📦 Chargement...")
tokenizer = MYBPE(vocab_size=300)
tokenizer.load_tokenizer("Tokenizer/tokenizer_model.bin")

# Charger le modèle
model = GPT2Model(
    vocab_size=300,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    max_seq_len=64
)
model.load_state_dict(torch.load('./checkpoints/final_model.pt'))
model.eval()

print("✅ Modèle prêt (~1M paramètres)\n")

print("="*60)
print("💬 MODE INTERACTIF")
print("="*60)
print("\n💡 Conseils:")
print("  • Temperature 0.5 = cohérent mais répétitif")
print("  • Temperature 0.8 = équilibré (recommandé)")
print("  • Temperature 1.5 = créatif mais chaotique")
print("\n  Tapez 'quit' pour quitter\n")

while True:
    print("-"*60)
    prompt = input("📝 Votre prompt: ")
    
    if prompt.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Au revoir!\n")
        break
    
    if not prompt.strip():
        print("⚠️  Prompt vide!\n")
        continue
    
    try:
        # Paramètres
        try:
            temp_input = input("🌡️  Temperature (appuyez Entrée pour 0.8): ")
            temp = float(temp_input) if temp_input.strip() else 0.8
            
            tokens_input = input("📏 Nombre de tokens (appuyez Entrée pour 50): ")
            max_tokens = int(tokens_input) if tokens_input.strip() else 50
        except:
            temp = 0.8
            max_tokens = 50
        
        print(f"\n⏳ Génération en cours...")
        
        # Encoder (silencieux)
        tokens = silent_operation(tokenizer.encoder, prompt)
        input_ids = torch.tensor([tokens])
        
        # Générer (silencieux)
        generated = silent_operation(
            model.generate,
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=40
        )
        
        # Décoder (silencieux)
        text = silent_operation(tokenizer.decoder, generated[0].tolist())
        
        # Afficher le résultat
        print("\n" + "="*60)
        print("✨ RÉSULTAT:")
        print("="*60)
        print(text)
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}\n")

print("="*60)
print("✅ Session terminée!")
print("="*60 + "\n")