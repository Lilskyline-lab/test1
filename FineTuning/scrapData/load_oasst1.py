"""
Charger et préparer le dataset OpenAssistant (oasst1)
161K conversations de haute qualité
"""

from datasets import load_dataset
import json
from tqdm import tqdm

print("="*60)
print("📥 CHARGEMENT DU DATASET OPENASSISTANT")
print("="*60)

# Charger le dataset
print("\n⏳ Téléchargement du dataset (peut prendre quelques minutes)...")
dataset = load_dataset("OpenAssistant/oasst1")

print(f"✅ Dataset chargé!")
print(f"   Train: {len(dataset['train'])} exemples")
print(f"   Validation: {len(dataset['validation'])} exemples")

# Examiner la structure
print("\n📊 Structure du dataset:")
print(f"   Colonnes: {dataset['train'].column_names}")

# Voir un exemple
print("\n📖 Exemple de données:")
example = dataset['train'][0]
for key, value in example.items():
    print(f"   {key}: {value}")

def extract_conversations(dataset_split, max_conversations=10000):
    """
    Extrait les conversations du dataset OpenAssistant
    
    Le dataset OASST1 est structuré en arbres de conversations.
    On va extraire les paires question/réponse.
    """
    print(f"\n🔄 Extraction des conversations...")
    
    conversations = []
    
    # Créer un dictionnaire pour mapper les IDs
    messages = {}
    for item in tqdm(dataset_split, desc="Indexation"):
        messages[item['message_id']] = item
    
    # Extraire les paires Human/Assistant
    for item in tqdm(dataset_split, desc="Extraction"):
        # Si c'est une réponse de l'assistant
        if item['role'] == 'assistant' and item['parent_id']:
            parent = messages.get(item['parent_id'])
            
            # Si le parent est une question humaine
            if parent and parent['role'] == 'prompter':
                # Filtrer par langue (optionnel)
                if item['lang'] == 'en' or item['lang'] == 'fr':
                    conversations.append({
                        'human': parent['text'],
                        'assistant': item['text'],
                        'lang': item['lang']
                    })
                    
                    if len(conversations) >= max_conversations:
                        break
    
    return conversations

# Extraire les conversations
print("\n🎯 Extraction du train set...")
train_conversations = extract_conversations(dataset['train'], max_conversations=10000)

print("\n🎯 Extraction du validation set...")
val_conversations = extract_conversations(dataset['validation'], max_conversations=1000)

print(f"\n✅ Conversations extraites:")
print(f"   Train: {len(train_conversations)}")
print(f"   Validation: {len(val_conversations)}")

# Statistiques
print("\n📊 Statistiques:")
total_langs = {}
for conv in train_conversations:
    lang = conv['lang']
    total_langs[lang] = total_langs.get(lang, 0) + 1

for lang, count in total_langs.items():
    print(f"   {lang}: {count} conversations")

# Afficher quelques exemples
print("\n📝 Exemples de conversations:")
for i, conv in enumerate(train_conversations[:3], 1):
    print(f"\n--- Exemple {i} ({conv['lang']}) ---")
    print(f"Human: {conv['human'][:100]}...")
    print(f"Assistant: {conv['assistant'][:100]}...")

# Sauvegarder
print("\n💾 Sauvegarde des conversations...")
with open('train_conversations.json', 'w', encoding='utf-8') as f:
    json.dump(train_conversations, f, ensure_ascii=False, indent=2)

with open('val_conversations.json', 'w', encoding='utf-8') as f:
    json.dump(val_conversations, f, ensure_ascii=False, indent=2)

print(f"✅ Sauvegardé dans:")
print(f"   - train_conversations.json ({len(train_conversations)} conversations)")
print(f"   - val_conversations.json ({len(val_conversations)} conversations)")

print("\n" + "="*60)
print("✅ PRÉPARATION TERMINÉE!")
print("="*60)
print("\n💡 Vous pouvez maintenant lancer le fine-tuning avec:")
print("   python chat_finetune.py")
print("="*60)