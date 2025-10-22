from tqdm import tqdm
import argparse
import pickle
import os
import time
from collections import Counter

class MYBPE():
    def __init__(self, vocab_size, dataset=None):
        self.vocab_size = vocab_size
        if dataset is not None:
            self.dataset = list(dataset.encode("utf-8"))
    
    def get_pairs(self, dataset):
        """
        Generate and count adjacent token pairs efficiently using Counter.
        """
        pairs = Counter()
        for i in range(len(dataset) - 1):
            pairs[(dataset[i], dataset[i+1])] += 1
        return pairs
    
    def merge_tokens(self, tokens, pair, new_id):
        """
        Replace all occurrences of a token pair with a new token id.
        Optimized with list comprehension avoidance for better performance.
        """
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                merged_tokens.append(new_id)
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens
    
    def train_tokenizer(self, checkpoint_path=None, checkpoint_freq=500):
        """
        Train the BPE tokenizer by iteratively merging frequent token pairs.
        Optimized for large vocabularies with progress tracking and checkpointing.
        """
        print("\n------ Training BPE Tokenizer ------")
        print(f"Target vocabulary size: {self.vocab_size}")
        print(f"Number of merges needed: {self.vocab_size - 256}")
        print(f"Dataset size: {len(self.dataset):,} bytes ({len(self.dataset)/1e9:.2f} GB)\n")
        
        # Estimate time based on dataset size
        estimated_hours = (len(self.dataset) * (self.vocab_size - 256)) / 600_000_000
        
        if checkpoint_path:
            print(f"💾 Checkpointing enabled: saving every {checkpoint_freq} iterations to {checkpoint_path}\n")
        
        num_merged_tokens = self.vocab_size - 256
        tokens = self.dataset
        self.merging_rules = {}
        
        start_time = time.time()
        last_checkpoint_time = start_time
        
        with tqdm(total=num_merged_tokens, desc="Training Progress") as pbar:
            for i in range(num_merged_tokens):
                pair_details = self.get_pairs(tokens)
                
                if not pair_details:
                    print(f"\nWarning: No more pairs to merge at iteration {i}")
                    break
                
                top_pair = max(pair_details, key=pair_details.get)
                new_token_id = i + 256
                
                # Calculate time remaining
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time_per_iter = elapsed / i
                    eta_seconds = avg_time_per_iter * (num_merged_tokens - i)
                    eta_hours = eta_seconds / 3600
                else:
                    eta_hours = 0
                
                # Log progress
                if i % 50 == 0 or i < 10:
                    pbar.set_postfix({
                        'freq': pair_details[top_pair],
                        'tokens': f"{len(tokens):,}",
                        'ETA': f"{eta_hours:.2f}h"
                    })
                
                tokens = self.merge_tokens(tokens, top_pair, new_token_id)
                self.merging_rules[top_pair] = new_token_id
                pbar.update(1)
                
                # Checkpoint saving
                if checkpoint_path and (i + 1) % checkpoint_freq == 0:
                    checkpoint_time = time.time()
                    time_since_last = checkpoint_time - last_checkpoint_time
                    temp_path = f"{checkpoint_path}.checkpoint_{i+1}"
                    self.build_vocabulary()
                    self.save_tokenizer(temp_path)
                    print(f"\n💾 Checkpoint saved at iteration {i+1} (took {time_since_last:.1f}s)")
                    last_checkpoint_time = time.time()
        
        total_time = time.time() - start_time
        hours = total_time / 3600
        minutes = (total_time % 3600) / 60
        print(f"\n✓ Training completed in {hours:.0f}h {minutes:.0f}m ({total_time:.1f} seconds)")
        print(f"  Created {len(self.merging_rules)} merge rules.")
        return self.merging_rules
    
    def build_vocabulary(self):
        """
        Build the vocabulary mapping from token ID to actual byte sequence.
        """
        print("\n------ Building Vocabulary ------")
        self.voc = {i: bytes([i]) for i in range(256)}
        
        for pair, val in tqdm(self.merging_rules.items(), desc="Building vocab"):
            self.voc[val] = self.voc[pair[0]] + self.voc[pair[1]]
        
        print(f"Vocabulary size: {len(self.voc)} tokens")
    
    def save_tokenizer(self, path):
        """Save tokenizer with additional metadata."""
        with open(path, "wb") as f:
            pickle.dump({
                "merging_rules": self.merging_rules,
                "vocabulary": self.voc,
                "vocab_size": self.vocab_size
            }, f)
    
    def load_tokenizer(self, path):
        """Load tokenizer with validation."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.merging_rules = data["merging_rules"]
            self.voc = data["vocabulary"]
            if "vocab_size" in data:
                self.vocab_size = data["vocab_size"]
        print(f"✓ Tokenizer loaded from {path}")
        print(f"  - Vocabulary size: {len(self.voc)}")
    
    def decoder(self, ids):
        """
        Decode a list of token IDs into a UTF-8 string using the vocabulary.
        """
        text = b"".join(self.voc[i] for i in ids)
        text = text.decode("utf-8", errors="replace")
        return text
    
    def encoder(self, text):
        """
        Encode raw UTF-8 text into token IDs using trained merges.
        Optimized with early stopping and better pair finding.
        """
        byte_tokens = list(text.encode("utf-8"))
        
        # Precompute minimum merge priority for faster lookup
        merge_priority = {pair: idx for pair, idx in self.merging_rules.items()}
        
        with tqdm(desc="Encoding", unit=" merge") as pbar:
            while len(byte_tokens) > 1:
                pairs = self.get_pairs(byte_tokens)
                
                # Find the pair with highest priority (lowest index in merge rules)
                replace_pair = min(
                    pairs.keys(), 
                    key=lambda p: merge_priority.get(p, float('inf')),
                    default=None
                )
                
                if replace_pair is None or replace_pair not in self.merging_rules:
                    break
                
                byte_tokens = self.merge_tokens(
                    byte_tokens, 
                    replace_pair, 
                    self.merging_rules[replace_pair]
                )
                pbar.update(1)
        
        return byte_tokens

def valid_tokenizer_model(name: str):
    """Validate tokenizer model file."""
    if not name.endswith(".bin"):
        raise argparse.ArgumentTypeError("File must have a '.bin' extension.")
    
    if os.path.exists(name):
        try:
            with open(name, "rb") as f:
                data = pickle.load(f)
            required_keys = ["merging_rules", "vocabulary"]
            if not isinstance(data, dict) or not all(k in data for k in required_keys):
                raise argparse.ArgumentTypeError(
                    "The .bin file must contain 'merging_rules' and 'vocabulary'."
                )
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid .bin file content: {e}")
    return name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized Byte Pair Encoding Tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a tokenizer with 5000 vocab size (with checkpointing for long training)
  python tokenizer.py --train --vocab_size 5000 --dataset train.txt --save tokenizer_5k.bin --checkpoint checkpoint
  
  # Train on Google Colab (recommended for 2GB dataset)
  python tokenizer.py --train --vocab_size 5000 --dataset large_data.txt --save tokenizer_5k.bin
  
  # Use the tokenizer to encode text
  python tokenizer.py --use_tokenizer --load tokenizer_5k.bin --input "Hello world!"
  
  # Encode a file
  python tokenizer.py --use_tokenizer --load tokenizer_5k.bin --input input.txt
        """
    )
    
    parser.add_argument("--dataset", type=str, default="./train.txt",
                        help="Dataset for training (text file)")
    parser.add_argument("--save", default="./tokenizer_model.bin", type=valid_tokenizer_model,
                        help="Path to save the tokenizer")
    parser.add_argument("--load", default="./tokenizer_model.bin", type=valid_tokenizer_model,
                        help="Path to load the tokenizer")
    parser.add_argument("--use_tokenizer", action="store_true",
                        help="Run the tokenizer on input")
    parser.add_argument("--vocab_size", default=5000, type=int,
                        help="Desired vocabulary size (>= 256, default: 5000)")
    parser.add_argument("--train", action="store_true",
                        help="Train a new tokenizer")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Enable checkpointing - save progress every N iterations")
    parser.add_argument("--checkpoint_freq", type=int, default=500,
                        help="Checkpoint frequency (default: 500 iterations)")
    parser.add_argument("--input", type=str,
                        help="File path or raw input string to tokenize")
    
    args = parser.parse_args()
    
    # Validate vocab_size
    if args.vocab_size < 256:
        parser.error("vocab_size must be at least 256")
    
    if args.train:
        print(f"\n{'='*50}")
        print(f"TRAINING MODE")
        print(f"{'='*50}")
        
        with open(args.dataset, "r", encoding="utf-8") as f:
            data = f.read()
        
        print(f"Dataset size: {len(data):,} characters ({len(data)/1e6:.2f} MB)")
        
        tokenizer = MYBPE(args.vocab_size, data)
        tokenizer.train_tokenizer(checkpoint_path=args.checkpoint, checkpoint_freq=args.checkpoint_freq)
        tokenizer.build_vocabulary()
        tokenizer.save_tokenizer(args.save)
        
        print(f"\n{'='*50}")
        print("Training completed successfully!")
        print(f"{'='*50}\n")
    
    if args.use_tokenizer:
        
        tokenizer = MYBPE(args.vocab_size)
        tokenizer.load_tokenizer(args.load)
        
        if not args.input:
            parser.error("--input is required when using --use_tokenizer")
        
        if os.path.isfile(args.input):
            with open(args.input, "r", encoding="utf-8") as f:
                input_data = f.read()
        else:
            input_data = args.input
       