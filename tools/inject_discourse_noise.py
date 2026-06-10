import json
import re
import random
import numpy as np
import os
import subprocess
from collections import Counter
from lxml import etree

def apply_lexical_noise(text):
    """Legacy surface noise: character/word level typos and casing."""
    if not text:
        return ""
    # Casual/Lowercase - low frequency
    if random.random() < 0.1:
        text = text.lower()
    
    # Filler words - minimal to avoid hedge_rate spike
    fillers = ["um", "uh", "well", "like", "actually"]
    if random.random() < 0.02:
        words = text.split()
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(fillers))
        text = " ".join(words)
            
    return text

def load_empirical_message_bank(xml_path, predator_ids, limit=5000):
    context = etree.iterparse(xml_path, events=('end',), tag='conversation')
    bank = []
    for event, elem in context:
        messages = elem.xpath('message')
        for msg in messages:
            author = msg.xpath('author/text()')[0] if msg.xpath('author/text()') else ''
            if author not in predator_ids:
                text = msg.xpath('text/text()')[0] if msg.xpath('text/text()') else ''
                if text and len(text.split()) < 15: # Only use short/mid neutral fragments
                    bank.append(text)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if len(bank) >= limit:
            break
    return bank

class LatentPerturbationField:
    def __init__(self, empirical_stats, msg_bank):
        self.stats = empirical_stats
        self.msg_bank = msg_bank
        self.empirical_reciprocity = self.stats.get('reciprocity', [0.8])
        self.empirical_latencies = self.stats.get('latencies', [1.0])
        self.empirical_lengths = self.stats['distributions'].get('msg_lengths', [10])

    def _sample_asymmetry(self):
        # A: Global participation skew (0.0 to 1.0)
        return random.choice(self.empirical_reciprocity)

    def _sample_entropy(self, prev_E, A, progress):
        # E_t: Semantic focus drift. Increases with progress and lower A (chaotic skew)
        drift = random.gauss(0.05, 0.02)
        noise = random.uniform(-0.01, 0.01)
        return min(1.0, max(0.0, prev_E + drift * (1.1 - A) + noise))

    def _sample_delay(self, E_t, A, prev_D):
        # D_t: Cadence. High entropy or low reciprocity increases delay
        base_delay = random.choice(self.empirical_latencies)
        multiplier = 1.0 + (E_t * 0.5) + ((1.0 - A) * 0.3)
        return base_delay * multiplier

    def _compute_realization_kernel(self, D_t, E_t, A, role_index):
        # Returns [pi_full, pi_trunc, pi_swap, pi_null]
        # role_index: 0 or 1. If role_index == 1 and A is low, pi_null increases.
        
        # Base probabilities
        pi_swap = min(0.4, E_t * 0.5)
        pi_null = 0.0
        if role_index == 1:
            pi_null = (1.0 - A) * 0.4
        
        pi_trunc = min(0.3, (D_t / max(self.empirical_latencies)) * 0.2)
        
        # Normalize
        total = pi_swap + pi_null + pi_trunc
        if total >= 1.0:
            # Scale down
            pi_swap /= (total + 0.1)
            pi_null /= (total + 0.1)
            pi_trunc /= (total + 0.1)
        
        pi_full = 1.0 - (pi_swap + pi_null + pi_trunc)
        return [pi_full, pi_trunc, pi_swap, pi_null]

    def _execute_strategy(self, strategy_idx, content, E_t, D_t):
        # strategies: 0: full, 1: trunc, 2: swap, 3: null
        if strategy_idx == 3: # Null
            return ""
        
        if strategy_idx == 2: # Swap
            return random.choice(self.msg_bank)
        
        if strategy_idx == 1: # Truncate
            words = content.split()
            if len(words) > 3:
                # Truncate to a short clip or acknowledgment
                if random.random() < 0.5:
                    return random.choice(["k", "yep", "ok", "cool", "maybe", "idk"])
                return " ".join(words[:random.randint(1, 3)]) + "..."
            return content
            
        # Full (0) - minimal pruning based on empirical lengths if content is huge
        return content

    def apply_field(self, trajectory):
        turns = trajectory['turns']
        T = len(turns)
        perturbed_turns = []
        
        # 1. Sample Global Asymmetry
        A = self._sample_asymmetry()
        
        E_t = 0.1
        prev_D = 0.0
        
        # Identify roles to map asymmetry
        authors = list(set(t['role'] for t in turns))
        
        for t, original_turn in enumerate(turns):
            role_idx = 0 if original_turn['role'] == authors[0] else 1
            
            # 2. Evolve Latent State
            E_t = self._sample_entropy(E_t, A, t/T)
            D_t = self._sample_delay(E_t, A, prev_D)
            
            # 3. Kernel Sampling
            kernel = self._compute_realization_kernel(D_t, E_t, A, role_idx)
            strategy = np.random.choice([0, 1, 2, 3], p=kernel)
            
            # 4. Execute
            new_content = self._execute_strategy(strategy, original_turn['content'], E_t, D_t)
            
            if new_content:
                new_turn = original_turn.copy()
                new_turn['content'] = apply_lexical_noise(new_content)
                new_turn['latency'] = D_t
                new_turn['latent_stats'] = {
                    'entropy': round(E_t, 3),
                    'asymmetry': round(A, 3),
                    'strategy': int(strategy)
                }
                perturbed_turns.append(new_turn)
            
            prev_D = D_t
            
        return {'turns': perturbed_turns}

def inject_discourse_noise(gen_path, stats_path, out_path):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    train_xml = 'Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    predators_file = 'Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
    with open(predators_file, 'r') as f:
        predator_ids = set(line.strip() for line in f)
        
    print("Loading empirical message bank...")
    msg_bank = load_empirical_message_bank(train_xml, predator_ids)
    
    perturber = LatentPerturbationField(stats, msg_bank)
    noisy_trajectories = []
    
    with open(gen_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            traj = json.loads(line)
            noisy_traj = perturber.apply_field(traj)
            noisy_trajectories.append(noisy_traj)

    with open(out_path, 'w') as f:
        for traj in noisy_trajectories:
            f.write(json.dumps(traj) + '\n')
            
    return len(noisy_trajectories)

def main():
    gen_path = 'Adversarial-Intent-Telemetry/data/pan_annotated/regenerated_trajectories.jsonl'
    stats_path = 'Adversarial-Intent-Telemetry/tools/pan12_empirical_stats.json'
    out_path = 'Adversarial-Intent-Telemetry/data/pan_annotated/regenerated_trajectories_noisy.jsonl'
    
    if not os.path.exists(gen_path) or not os.path.exists(stats_path):
        print(f"Missing input files. Checked: {gen_path}, {stats_path}")
        return

    print("Injecting latent discourse noise...")
    count = inject_discourse_noise(gen_path, stats_path, out_path)
    print(f"Noise injection complete for {count} trajectories.")
    
    print("Running discrimination audit on noisy data...")
    subprocess.run([
        "python3", "Adversarial-Intent-Telemetry/tools/audit_human_vs_generated.py", out_path
    ])
    
if __name__ == "__main__":
    main()
