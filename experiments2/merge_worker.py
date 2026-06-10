import json
import random
import numpy as np
import os
import sys
from lxml import etree
from pathlib import Path

# Paths
ROOT = Path("Adversarial-Intent-Telemetry")
ADAPTED_PATH = ROOT / "data" / "pan_annotated" / "adapted_trajectories.jsonl"
NCMEC_PATH = ROOT / "data" / "agentic_ncmec" / "pan_ncmec_trajectories.jsonl"
STATS_PATH = ROOT / "tools" / "pan12_empirical_stats.json"
XML_PATH = ROOT / "data" / "pan12" / "train" / "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
PREDATORS_PATH = ROOT / "data" / "pan12" / "train" / "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"

# Logic from inject_discourse_noise.py
def apply_lexical_noise(text):
    if not text: return ""
    if random.random() < 0.1: text = text.lower()
    fillers = ["um", "uh", "well", "like", "actually"]
    if random.random() < 0.02:
        words = text.split()
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(fillers))
        text = " ".join(words)
    return text

def load_empirical_message_bank(xml_path, predator_ids, limit=5000):
    context = etree.iterparse(str(xml_path), events=('end',), tag='conversation')
    bank = []
    for event, elem in context:
        messages = elem.xpath('message')
        for msg in messages:
            author = msg.xpath('author/text()')[0] if msg.xpath('author/text()') else ''
            if author not in predator_ids:
                text = msg.xpath('text/text()')[0] if msg.xpath('text/text()') else ''
                if text and len(text.split()) < 15:
                    bank.append(text)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if len(bank) >= limit: break
    return bank

class LatentPerturbationField:
    def __init__(self, empirical_stats, msg_bank):
        self.stats = empirical_stats
        self.msg_bank = msg_bank
        self.empirical_reciprocity = self.stats.get('reciprocity', [0.8])
        self.empirical_latencies = self.stats.get('latencies', [1.0])
        self.empirical_lengths = self.stats['distributions'].get('msg_lengths', [10])

    def apply_field(self, trajectory):
        turns = trajectory['turns']
        T = len(turns)
        perturbed_turns = []
        A = random.choice(self.empirical_reciprocity)
        E_t = 0.1
        prev_D = 0.0
        authors = list(set(t['role'] for t in turns))
        
        for t, original_turn in enumerate(turns):
            role_idx = 0 if original_turn['role'] == authors[0] else 1
            # Latent state evolution logic
            drift = random.gauss(0.05, 0.02)
            noise = random.uniform(-0.01, 0.01)
            E_t = min(1.0, max(0.0, E_t + drift * (1.1 - A) + noise))
            
            base_delay = random.choice(self.empirical_latencies)
            D_t = base_delay * (1.0 + (E_t * 0.5) + ((1.0 - A) * 0.3))
            
            pi_swap = min(0.4, E_t * 0.5)
            pi_null = (1.0 - A) * 0.4 if role_idx == 1 else 0.0
            pi_trunc = min(0.3, (D_t / max(self.empirical_latencies)) * 0.2)
            
            total = pi_swap + pi_null + pi_trunc
            if total >= 1.0:
                pi_swap /= (total + 0.1)
                pi_null /= (total + 0.1)
                pi_trunc /= (total + 0.1)
            
            pi_full = 1.0 - (pi_swap + pi_null + pi_trunc)
            strategy = np.random.choice([0, 1, 2, 3], p=[pi_full, pi_trunc, pi_swap, pi_null])
            
            # Execute strategy
            new_content = original_turn['content']
            if strategy == 3: new_content = ""
            elif strategy == 2: new_content = random.choice(self.msg_bank)
            elif strategy == 1:
                words = original_turn['content'].split()
                if len(words) > 3:
                    if random.random() < 0.5: new_content = random.choice(["k", "yep", "ok", "cool", "maybe", "idk"])
                    else: new_content = " ".join(words[:random.randint(1, 3)]) + "..."
            
            if new_content:
                new_turn = original_turn.copy()
                new_turn['content'] = apply_lexical_noise(new_content)
                new_turn['latency'] = round(D_t, 3)
                new_turn['latent_stats'] = {
                    'entropy': round(E_t, 3),
                    'asymmetry': round(A, 3),
                    'strategy': int(strategy)
                }
                perturbed_turns.append(new_turn)
            prev_D = D_t
        
        new_traj = trajectory.copy()
        new_traj['turns'] = perturbed_turns
        return new_traj

def main():
    # Load assets
    try:
        with open(STATS_PATH) as f: stats = json.load(f)
        with open(PREDATORS_PATH) as f: predator_ids = set(line.strip() for line in f)
        msg_bank = load_empirical_message_bank(XML_PATH, predator_ids)
        perturber = LatentPerturbationField(stats, msg_bank)
    except Exception as e:
        sys.stderr.write(f"Error loading assets: {e}\n")
        sys.exit(1)
    
    # Load NCMEC data
    ncmec_data = {}
    try:
        with open(NCMEC_PATH) as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                ncmec_data[obj['trajectory_id']] = obj
    except Exception as e:
        sys.stderr.write(f"Error loading NCMEC data: {e}\n")
        sys.exit(1)
            
    # Process adapted trajectories
    results = []
    try:
        with open(ADAPTED_PATH) as f:
            for line in f:
                if not line.strip(): continue
                traj = json.loads(line)
                # Phase 1: Noise injection
                noisy_traj = perturber.apply_field(traj)
                
                # Phase 2: NCMEC Enrichment
                tid = noisy_traj['trajectory_id']
                enrichment = None
                if tid in ncmec_data:
                    e = ncmec_data[tid]
                    enrichment = {
                        "events_injected": e.get("events_injected"),
                        "injection_boundary_after_turn": e.get("injection_boundary_after_turn"),
                        "injection_seed": e.get("injection_seed"),
                        "ncmec_prior_version": e.get("ncmec_prior_version")
                    }
                noisy_traj['ncmec_enrichment'] = enrichment
                results.append(noisy_traj)
    except Exception as e:
        sys.stderr.write(f"Error processing trajectories: {e}\n")
        sys.exit(1)
            
    # Output
    for r in results:
        print(json.dumps(r))

if __name__ == "__main__":
    main()
