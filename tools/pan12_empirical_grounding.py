#!/usr/bin/env python3
"""
PAN12 Empirical Grounding Tool
Goal: Extract structural conversational distributions from PAN12 and use them to condition neutral-topic analogue generation.
Avoids explicit CSAE or sexual-content generation.

Usage:
  # Analyze PAN12 corpus and export stats/templates
  python3 pan12_empirical_grounding.py --analyze
  
  # Validate generated trajectories against empirical stats
  python3 pan12_empirical_grounding.py --validate
"""
import os
import json
import math
import re
import random
from collections import Counter, defaultdict
from lxml import etree
import numpy as np

def parse_time(time_str):
    try:
        if not time_str:
            return None
        # Format is HH:MM
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    except:
        return None

def get_lexical_entropy(text):
    if not text:
        return 0
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
    return entropy

def analyze_pan12(xml_path, predator_ids=None):
    stats = {
        'turn_counts': [],
        'msg_lengths': [],
        'latencies': [],
        'reciprocity': [],
        'lexical_entropy': [],
        'migration_timing': [],
        'repair_freq': 0,
        'hedge_freq': 0,
        'total_msgs': 0,
        'templates': [] # Store structural templates of predator convs
    }
    
    # Hedges and Repairs (Heuristic-based)
    hedges = r'\b(maybe|perhaps|i think|possibly|i guess|not sure|sort of|kind of|actually)\b'
    repairs = r'\b(sorry|i mean|no i meant|my mistake|wait)\b'
    migration_keywords = r'\b(skype|yahoo|msn|aim|kik|whatsapp|telegram|discord|email|add me|move to)\b'

    context = etree.iterparse(xml_path, events=('end',), tag='conversation')
    
    count = 0
    for event, elem in context:
        messages = elem.xpath('message')
        authors = [msg.xpath('author/text()')[0] if msg.xpath('author/text()') else '' for msg in messages]
        
        is_predator_conv = any(a in predator_ids for a in authors) if predator_ids else False
        
        num_turns = len(messages)
        stats['turn_counts'].append(num_turns)
        
        times = [parse_time(msg.xpath('time/text()')[0]) if msg.xpath('time/text()') else None for msg in messages]
        texts = [msg.xpath('text/text()')[0] if msg.xpath('text/text()') else '' for msg in messages]
        
        author_counts = Counter(authors)
        if len(author_counts) >= 2:
            counts = list(author_counts.values())
            stats['reciprocity'].append(min(counts) / max(counts))
        
        template = []
        prev_time = None
        for i, text in enumerate(texts):
            stats['total_msgs'] += 1
            words = text.split()
            length = len(words)
            stats['msg_lengths'].append(length)
            stats['lexical_entropy'].append(get_lexical_entropy(text))
            
            has_hedge = bool(re.search(hedges, text.lower()))
            has_repair = bool(re.search(repairs, text.lower()))
            has_migration = bool(re.search(migration_keywords, text.lower()))
            
            if has_hedge: stats['hedge_freq'] += 1
            if has_repair: stats['repair_freq'] += 1
            if has_migration: stats['migration_timing'].append(i / num_turns if num_turns > 0 else 0)
            
            latency = 0
            current_time = times[i]
            if prev_time is not None and current_time is not None:
                latency = current_time - prev_time
                if latency < 0: latency += 1440
                if latency < 60:
                    stats['latencies'].append(latency)
            prev_time = current_time
            
            if is_predator_conv:
                template.append({
                    'role': 'predator' if authors[i] in predator_ids else 'victim',
                    'length': length,
                    'latency': latency,
                    'has_hedge': has_hedge,
                    'has_repair': has_repair,
                    'has_migration': has_migration
                })
        
        if is_predator_conv and template:
            stats['templates'].append(template)
            
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} conversations...")

    return stats

def select_templates(templates, n=100, min_turns=8, max_turns=30):
    eligible = [t for t in templates if min_turns <= len(t) <= max_turns]
    if len(eligible) < n:
        return eligible
    return random.sample(eligible, n)

def generate_analogue_prompt(template, topic):
    turns_desc = []
    for i, t in enumerate(template):
        desc = f"Turn {i+1} ({t['role']}): approx {t['length']} words"
        if t['has_hedge']: desc += ", use a hedge"
        if t['has_repair']: desc += ", include a repair/clarification"
        if t['has_migration']: desc += ", mention moving to another topic/platform (neutral)"
        turns_desc.append(desc)
    
    prompt = (
        f"Generate a neutral-topic conversation about '{topic}' following this exact structural template:\n"
        + "\n".join(turns_desc) + "\n\n"
        "Ensure the conversation is mundane, professional, or hobby-focused. "
        "Preserve the cadence and relative message lengths. "
        "Output ONLY the JSON format:\n"
        '{"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}'
    )
    return prompt

def calculate_kl_divergence(p, q, bins=20):
    p_hist, edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=edges, density=True)
    
    # Add small epsilon to avoid division by zero or log(0)
    p_hist += 1e-10
    q_hist += 1e-10
    
    return np.sum(p_hist * np.log(p_hist / q_hist))

def validate_trajectories(gen_path, stats_path):
    with open(stats_path, 'r') as f:
        empirical = json.load(f)
    
    gen_turn_counts = []
    gen_msg_lengths = []
    gen_lexical_entropy = []
    gen_migration_positions = []
    
    with open(gen_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            turns = data['turns']
            gen_turn_counts.append(len(turns))
            for i, turn in enumerate(turns):
                text = turn['content']
                gen_msg_lengths.append(len(text.split()))
                gen_lexical_entropy.append(get_lexical_entropy(text))
                
                # Check for migration keywords in generated text
                migration_keywords = r'\b(skype|yahoo|msn|aim|kik|whatsapp|telegram|discord|email|add me|move to)\b'
                if re.search(migration_keywords, text.lower()):
                    gen_migration_positions.append(i / len(turns))

    # Metrics
    kl_length = calculate_kl_divergence(empirical['distributions']['msg_lengths'], gen_msg_lengths)
    kl_entropy = calculate_kl_divergence(empirical['distributions']['lexical_entropy'], gen_lexical_entropy)
    
    # Let's re-calculate lexical entropy distribution for empirical if needed, but for now we'll use mean comparison
    mean_len_diff = abs(np.mean(empirical['distributions']['msg_lengths']) - np.mean(gen_msg_lengths))
    mean_entropy_diff = abs(empirical['avg_lexical_entropy'] - np.mean(gen_lexical_entropy))
    
    metrics = {
      'kl_divergence_msg_length': float(kl_length),
      'mean_msg_length_diff': float(mean_len_diff),
      'mean_lexical_entropy_diff': float(mean_entropy_diff),
      'avg_migration_pos_diff': float(abs(empirical['avg_migration_pos'] - np.mean(gen_migration_positions))) if gen_migration_positions else None
    }
    
    print("\nValidation Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Thresholds for aborting
    if kl_length > 5.0 or mean_len_diff > 10:
        print("ABORT: Generated distributions drift substantially from PAN12 empirical distributions.")
        return False
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    if args.analyze:
        train_xml = 'Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
        predators_file = 'Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
        
        with open(predators_file, 'r') as f:
            predator_ids = set(line.strip() for line in f)
        
        print("Analyzing PAN12 training corpus...")
        stats = analyze_pan12(train_xml, predator_ids)
        
        aggregated = {
            'avg_turn_count': np.mean(stats['turn_counts']),
            'std_turn_count': np.std(stats['turn_counts']),
            'avg_msg_length': np.mean(stats['msg_lengths']),
            'avg_latency': np.mean(stats['latencies']),
            'avg_reciprocity': np.mean(stats['reciprocity']),
            'avg_lexical_entropy': np.mean(stats['lexical_entropy']),
            'hedge_rate': stats['hedge_freq'] / stats['total_msgs'],
            'repair_rate': stats['repair_freq'] / stats['total_msgs'],
            'avg_migration_pos': np.mean(stats['migration_timing']) if stats['migration_timing'] else 0,
            'distributions': {
                'msg_lengths': stats['msg_lengths'],
                'lexical_entropy': stats['lexical_entropy']
            },
            'templates': stats['templates']
        }
        
        with open('Adversarial-Intent-Telemetry/tools/pan12_empirical_stats.json', 'w') as f:
            json.dump(aggregated, f, indent=2)
        print("Stats and templates exported.")

    if args.validate:
        gen_path = 'Adversarial-Intent-Telemetry/data/pan_annotated/regenerated_trajectories.jsonl'
        stats_path = 'Adversarial-Intent-Telemetry/tools/pan12_empirical_stats.json'
        success = validate_trajectories(gen_path, stats_path)
        if not success:
            exit(1)

if __name__ == "__main__":
    main()
