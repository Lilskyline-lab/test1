import os
import json
import argparse
import random
from collections import defaultdict

def filter_pairs(pairs, min_len=5, max_len=1000):
    out = []
    seen = set()
    for p in pairs:
        h = ' '.join(p['human'].split())
        a = ' '.join(p['assistant'].split())
        if not h or not a:
            continue
        if len(h) < min_len or len(a) < min_len:
            continue
        if len(h) > max_len or len(a) > max_len:
            continue
        key = (h, a)
        if key in seen:
            continue
        seen.add(key)
        out.append({'human': h, 'assistant': a})
    return out

def save_json_combined(train_list, val_list, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    out = {"train": train_list, "val": val_list}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def extract_pairs_from_oasst(messages):
    """
    messages : liste de dict avec (message_id, parent_id, role, text)
    On veut trouver pour chaque message d’assistant, qui est le prompter précédent, et faire une paire.
    """
    # indexer les messages par id
    by_id = {m['message_id']: m for m in messages}
    pairs = []
    for msg in messages:
        role = msg.get('role')
        if role != 'assistant':
            continue
        parent_id = msg.get('parent_id')
        if not parent_id:
            continue
        parent = by_id.get(parent_id)
        if not parent:
            continue
        # parent doit être “prompter”
        if parent.get('role') in ('prompter', 'human', 'user'):
            human = parent.get('text')
            assistant = msg.get('text')
            if human and assistant:
                pairs.append({'human': human, 'assistant': assistant})
    return pairs

def load_and_extract(hf_id, max_examples=None, debug=False):
    from datasets import load_dataset
    print(f"-> Chargement du dataset : {hf_id}")
    ds = None
    try:
        ds = load_dataset(hf_id, split='train')
    except Exception as e:
        try:
            ds_all = load_dataset(hf_id)
            if isinstance(ds_all, dict):
                # prendre le premier split existant
                for s in ('train', 'validation', 'test'):
                    if s in ds_all:
                        ds = ds_all[s]
                        break
                else:
                    ds = list(ds_all.values())[0]
            else:
                ds = ds_all
        except Exception as e2:
            print(f"  [WARN] impossible de charger {hf_id}: {e2}")
            return [], []
    if ds is None:
        return [], []

    # Convertir en liste de dicts (limitation)
    msgs = []
    cnt = 0
    for ex in ds:
        cnt += 1
        if max_examples and cnt > max_examples:
            break
        # Pour OASST, le format plat : ex a 'message_id', 'parent_id', 'role', 'text'
        # On extrait ces champs si présents
        if isinstance(ex, dict) and 'message_id' in ex and 'role' in ex and 'text' in ex:
            msgs.append({
                'message_id': ex['message_id'],
                'parent_id': ex.get('parent_id'),
                'role': ex['role'],
                'text': ex['text']
            })
        # Sinon, ignorer
    if debug:
        print("  Quelques messages bruts :")
        for i, m in enumerate(msgs[:5]):
            print(f"   - {i}: id={m['message_id']}, parent={m.get('parent_id')}, role={m['role']}, text={m['text'][:100]}")

    # Extraire les paires human → assistant
    pairs = extract_pairs_from_oasst(msgs)
    print(f"  Extrait {len(pairs)} paires du dataset {hf_id}")
    return pairs, msgs

def main():
    parser = argparse.ArgumentParser(description="Préparer des paires human/assistant à partir de datasets")
    parser.add_argument("--out-dir", type=str, default="./output", help="Répertoire de sortie")
    parser.add_argument("--total", type=int, default=100000, help="Total de paires souhaitées (par défaut 100000)")
    parser.add_argument("--train-size", type=int, default=50000, help="Nombre de paires pour train (par défaut 50000)")
    parser.add_argument("--debug", action="store_true", help="Afficher debug")
    parser.add_argument("--hf-oasst", type=str, default="OpenAssistant/oasst1", help="Identifiant HuggingFace pour OASST")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_pairs = []

    # Charger OASST
    p_oasst, _ = load_and_extract(args.hf_oasst, max_examples=None, debug=args.debug)
    all_pairs.extend(p_oasst)

    # Potentiellement, tu pourrais ajouter d’autres datasets ici, avec leurs propres extracteurs

    print(f"Total paires brutes avant filtrage : {len(all_pairs)}")
    filtered = filter_pairs(all_pairs)
    print(f"Paires après filtrage et déduplication : {len(filtered)}")

    if len(filtered) == 0:
        raise RuntimeError("Aucune paire extraite après filtrage. Vérifiez les datasets / formats.")

    needed = args.total
    if len(filtered) >= needed:
        sampled = random.sample(filtered, needed)
    else:
        # Si pas assez, répéter
        sampled = filtered[:] + [random.choice(filtered) for _ in range(needed - len(filtered))]
    random.shuffle(sampled)

    # afficher combien d'exemples ont été répliqués (approx)
    num_unique = len(filtered)
    num_duplicates = max(0, needed - num_unique)
    if num_duplicates > 0:
        print(f"[INFO] Pas assez d'exemples uniques ({num_unique}) — {num_duplicates} exemples seront répliqués (échantillonnage avec remise).")

    train_n = min(args.train_size, len(sampled))
    train = sampled[:train_n]
    val = sampled[train_n:]

    out_path = os.path.join(out_dir, "conversations.json")
    save_json_combined(train, val, out_path)
    print(f"Conversations sauvegardées dans : {out_path} (train={len(train)}, val={len(val)})")

if __name__ == "__main__":
    main()
