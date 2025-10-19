import os
import json
import torch
import argparse
import io
import sys

# importer votre modèle/tokenizer local (même chemins que dans test1.py)
sys.path.append('../Model')
sys.path.append('../Tokenizer')
from gpt2_model import GPT2Model
from Tokenizer import MYBPE

def silence_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    return old

def restore_stdout(old):
    sys.stdout = old

def load_config(model_dir):
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # fallback minimal config compatible avec test1.py
    return {
        "vocab_size": 300,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": 64
    }

def load_model_and_tokenizer(model_dir, tokenizer_path, device):
    # charger config
    config = load_config(model_dir)

    # tokenizer
    print("🔤 Chargement du tokenizer...")
    tokenizer = MYBPE(vocab_size=config.get("vocab_size", 300))
    tokenizer.load_tokenizer(tokenizer_path)
    print("✅ Tokenizer chargé")

    # modèle
    print("🤖 Initialisation du modèle...")
    model = GPT2Model(
        vocab_size=config.get("vocab_size", 300),
        embed_dim=config.get("embed_dim", 128),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        max_seq_len=config.get("max_seq_len", 64)
    )

    model_file = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Fichier de poids introuvable: {model_file}")

    # Essayer de charger en mode "weights_only" pour éviter les warnings/security (PyTorch récent)
    try:
        state = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:
        # Anciennes versions de torch n'ont pas weights_only -> fallback
        state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Modèle chargé depuis {model_file} sur {device}")
    return model, tokenizer, config

def generate_response(model, tokenizer, prompt, device, max_new_tokens=40, temperature=0.8):
    # encoder silencieusement
    old = silence_stdout()
    try:
        tokens = tokenizer.encoder(prompt)
    finally:
        restore_stdout(old)

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # si votre GPT2Model propose generate(), on l'utilise ; sinon tentative auto-regressive basique
    if hasattr(model, "generate"):
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        generated_ids = gen[0].tolist()
    else:
        # génération autoregressive simple (échantillonnage)
        generated_ids = input_ids[0].tolist()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([generated_ids[-model.max_seq_len:]], device=device)
                logits, _ = model(inp)
                next_logits = logits[0, -1, :].float() / max(1e-8, temperature)
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                generated_ids.append(next_id)

    # décoder silencieusement
    old = silence_stdout()
    try:
        text = tokenizer.decoder(generated_ids)
    finally:
        restore_stdout(old)

    # extraire la partie Bot:
    if "Bot:" in text:
        return text.split("Bot:")[-1].strip()
    # fallback: renvoyer la totalité décodée après prompt
    if prompt in text:
        return text[len(prompt):].strip()
    return text.strip()

def chat_loop(model, tokenizer, device, max_new_tokens, temperature):
    print("\n" + "="*60)
    print("💬 MODE CHAT — Tapez 'quit' pour quitter")
    print("="*60)
    while True:
        try:
            user = input("Vous: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            break
        prompt = f"Human: {user}\nBot:"
        try:
            resp = generate_response(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, temperature=temperature)
        except Exception as e:
            resp = f"[Erreur génération] {e}"
        print(f"Bot: {resp}\n")

def main():
    parser = argparse.ArgumentParser(description="Charger un modèle entraîné et chatter")
    parser.add_argument("--model-dir", type=str, default="./my_tiny_chatbot", help="Répertoire contenant model.pt et config.json")
    parser.add_argument("--tokenizer", type=str, default="../Tokenizer/tokenizer_model.bin", help="Chemin vers le tokenizer BPE")
    parser.add_argument("--device", type=str, default="cpu", help="Device (ex: cpu)")
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Nombre de tokens à générer")
    parser.add_argument("--temperature", type=float, default=0.8, help="Température de génération")
    args = parser.parse_args()

    device = torch.device(args.device)

    try:
        model, tokenizer, _ = load_model_and_tokenizer(args.model_dir, args.tokenizer, device)
    except Exception as e:
        print(f"Erreur au chargement: {e}")
        return

    chat_loop(model, tokenizer, device, args.max_new_tokens, args.temperature)

if __name__ == "__main__":
    main()