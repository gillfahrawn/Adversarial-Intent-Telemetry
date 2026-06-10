import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re
import os

def get_ngrams(text, n):
    words = re.findall(r'\w+', text.lower())
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def normalized_levenshtein(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def get_lexical_entropy(text):
    if not text:
        return 0
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((count/total) * np.log2(count/total) for count in counts.values())
    return entropy

def get_cadence(traj):
    return [len(t['content'].split()) for t in traj['turns']]

def get_motifs(traj):
    # Abstract roles and length bins as motifs
    # e.g. (user, short), (assistant, long)
    motifs = []
    for t in traj['turns']:
        length_bin = 'short' if len(t['content'].split()) < 5 else 'medium' if len(t['content'].split()) < 15 else 'long'
        motifs.append((t['role'], length_bin))
    return motifs

def audit_diversity(gen_path, stats_path):
    trajectories = []
    with open(gen_path, 'r') as f:
        for line in f:
            trajectories.append(json.loads(line))
    
    with open(stats_path, 'r') as f:
        empirical = json.load(f)
    
    n = len(trajectories)
    
    # ... (previous code)
    all_texts = [" ".join([t['content'] for t in traj['turns']]) for traj in trajectories]
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    cos_sim = cosine_similarity(vectorizer)
    
    jaccard_sims = []
    lev_sims = []
    ngram_overlaps = {2: [], 3: [], 5: []}
    cadence_sims = []
    motif_overlaps = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # ... (jaccard, lev, ngrams)
            set1 = set(re.findall(r'\w+', all_texts[i].lower()))
            set2 = set(re.findall(r'\w+', all_texts[j].lower()))
            jaccard_sims.append(jaccard_similarity(set1, set2))
            lev_sims.append(normalized_levenshtein(all_texts[i], all_texts[j]))
            for deg in [2, 3, 5]:
                ng1 = set(get_ngrams(all_texts[i], deg))
                ng2 = set(get_ngrams(all_texts[j], deg))
                ngram_overlaps[deg].append(jaccard_similarity(ng1, ng2))

            # Cadence Similarity (Correlation of turn lengths if same turn count)
            c1 = get_cadence(trajectories[i])
            c2 = get_cadence(trajectories[j])
            if len(c1) == len(c2):
                cadence_sims.append(np.corrcoef(c1, c2)[0, 1])
            
            # Motif Overlap
            m1 = set(get_motifs(trajectories[i]))
            m2 = set(get_motifs(trajectories[j]))
            motif_overlaps.append(jaccard_similarity(m1, m2))

    # Structural Reuse
    migration_positions = []
    for traj in trajectories:
        m_pos = []
        for i, turn in enumerate(traj['turns']):
            if re.search(r'\b(skype|yahoo|msn|aim|kik|whatsapp|telegram|discord|email|add me|move to)\b', turn['content'].lower()):
                m_pos.append(i / len(traj['turns']))
        migration_positions.append(m_pos)
    
    flat_m_pos = [p for sub in migration_positions for p in sub]
    m_pos_variance = np.var(flat_m_pos) if flat_m_pos else None
    
    # Distributional Diversity
    turn_counts = [len(t['turns']) for t in trajectories]
    msg_lengths = [len(turn['content'].split()) for t in trajectories for turn in t['turns']]
    lex_entropies = [get_lexical_entropy(turn['content']) for t in trajectories for turn in t['turns']]
    topics = [t['topic'] for t in trajectories]
    
    # Memorization Flags
    flags = []
    lev_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if cos_sim[i, j] > 0.8:
                flags.append({'type': 'high_cosine_similarity', 'ids': [trajectories[i]['trajectory_id'], trajectories[j]['trajectory_id']], 'value': cos_sim[i, j]})
            if lev_sims[lev_idx] > 0.8:
                flags.append({'type': 'high_levenshtein_similarity', 'ids': [trajectories[i]['trajectory_id'], trajectories[j]['trajectory_id']], 'value': lev_sims[lev_idx]})
            lev_idx += 1

    # Report Generation
    report = {
        'diversity_metrics': {
            'avg_jaccard_similarity': float(np.mean(jaccard_sims)),
            'avg_cosine_similarity': float(np.mean(cos_sim[np.triu_indices(n, k=1)])),
            'avg_normalized_levenshtein': float(np.mean(lev_sims)),
            'avg_ngram_overlap': {k: float(np.mean(v)) for k, v in ngram_overlaps.items()},
            'avg_cadence_correlation': float(np.nanmean(cadence_sims)) if cadence_sims else None,
            'avg_motif_overlap': float(np.mean(motif_overlaps)),
            'migration_position_variance': float(m_pos_variance) if m_pos_variance is not None else None,
            'turn_count_variance': float(np.var(turn_counts)),
            'msg_length_variance': float(np.var(msg_lengths)),
            'avg_lexical_entropy': float(np.mean(lex_entropies)),
            'unique_topics_count': len(set(topics))
        },
        'pan12_comparison': {
            'turn_count_diff': float(abs(np.mean(turn_counts) - empirical['avg_turn_count'])),
            'msg_length_diff': float(abs(np.mean(msg_lengths) - empirical['avg_msg_length'])),
            'lexical_entropy_diff': float(abs(np.mean(lex_entropies) - empirical['avg_lexical_entropy']))
        }
    }
    
    with open('Adversarial-Intent-Telemetry/experiments/results/diversity_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(cos_sim).to_csv('Adversarial-Intent-Telemetry/experiments/results/structural_overlap_matrix.csv', index=False)
    with open('Adversarial-Intent-Telemetry/experiments/results/memorization_flags.json', 'w') as f:
        json.dump(flags, f, indent=2)
        
    print("Diversity audit complete. Reports generated in experiments/results/")
    return report

if __name__ == "__main__":
    gen_path = 'Adversarial-Intent-Telemetry/data/pan_annotated/regenerated_trajectories.jsonl'
    stats_path = 'Adversarial-Intent-Telemetry/tools/pan12_empirical_stats.json'
    if os.path.exists(gen_path) and os.path.exists(stats_path):
        audit_diversity(gen_path, stats_path)
    else:
        print(f"Required files missing: {gen_path} or {stats_path}")
