import sys
sys.path.append('./Tokenizer')
sys.path.append('./Embeddings_Layer')

from Tokenizer import MYBPE
from Embeddings import GPT2Embeddings
import torch

# Charger tokenizer
tokenizer = MYBPE(vocab_size=300)
tokenizer.load_tokenizer("./Tokenizer/tokenizer_model.bin")

# Créer embeddings
embeddings = GPT2Embeddings(300, 768, 1024)

# Pipeline complet
text = "Bonjour, je teste mon GPT-2!"
tokens = tokenizer.encoder(text)
input_ids = torch.tensor([tokens])
output = embeddings(input_ids)

print(f"✓ Pipeline fonctionne!")
print(f"  Texte: {text}")
print(f"  Tokens: {tokens}")
print(f"  Embeddings: {output.shape}")