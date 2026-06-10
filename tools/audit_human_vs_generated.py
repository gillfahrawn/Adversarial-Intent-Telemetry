import json
import re
import numpy as np
import pandas as pd
from collections import Counter
from lxml import etree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

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

def extract_structural_features(turns):
    """
    Extracts structural features from a list of turns.
    Each turn is a dict with 'role' and 'content'.
    """
    if not turns:
        return None
    
    num_turns = len(turns)
    lengths = [len(re.findall(r'\w+', t['content'])) for t in turns]
    entropies = [get_lexical_entropy(t['content']) for t in turns]
    
    # Hedges and Repairs (Heuristic-based)
    hedges_regex = r'\b(maybe|perhaps|i think|possibly|i guess|not sure|sort of|kind of|actually)\b'
    repairs_regex = r'\b(sorry|i mean|no i meant|my mistake|wait)\b'
    migration_regex = r'\b(skype|yahoo|msn|aim|kik|whatsapp|telegram|discord|email|add me|move to)\b'
    
    hedge_count = sum(1 for t in turns if re.search(hedges_regex, t['content'].lower()))
    repair_count = sum(1 for t in turns if re.search(repairs_regex, t['content'].lower()))
    
    migration_pos = -1
    for i, t in enumerate(turns):
        if re.search(migration_regex, t['content'].lower()):
            migration_pos = i / num_turns
            break
            
    # Asymmetry & Reciprocity
    roles = [t['role'] for t in turns]
    role_counts = Counter(roles)
    if len(role_counts) >= 2:
        counts = list(role_counts.values())
        reciprocity = min(counts) / max(counts)
    else:
        reciprocity = 0
        
    # Feature vector
    features = {
        'turn_count': num_turns,
        'avg_len': np.mean(lengths),
        'std_len': np.std(lengths),
        'max_len': np.max(lengths),
        'avg_entropy': np.mean(entropies),
        'hedge_rate': hedge_count / num_turns,
        'repair_rate': repair_count / num_turns,
        'reciprocity': reciprocity,
        'migration_pos': migration_pos,
        # Cadence transitions (diff in lengths)
        'avg_len_diff': np.mean(np.abs(np.diff(lengths))) if len(lengths) > 1 else 0
    }
    return features

def load_pan12_features(xml_path, predator_ids, limit=2000):
    context = etree.iterparse(xml_path, events=('end',), tag='conversation')
    features_list = []
    count = 0
    for event, elem in context:
        messages = elem.xpath('message')
        authors = [msg.xpath('author/text()')[0] if msg.xpath('author/text()') else '' for msg in messages]
        # Only use predator convs for more relevant comparison? 
        # Requirement says "PAN12 predator conversation corpus"
        if not any(a in predator_ids for a in authors):
            elem.clear()
            continue
            
        turns = []
        for msg in messages:
            author = msg.xpath('author/text()')[0] if msg.xpath('author/text()') else ''
            text = msg.xpath('text/text()')[0] if msg.xpath('text/text()') else ''
            role = 'predator' if author in predator_ids else 'victim'
            turns.append({'role': role, 'content': text})
            
        feat = extract_structural_features(turns)
        if feat:
            features_list.append(feat)
            count += 1
            
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        
        if count >= limit:
            break
    return features_list

def load_generated_features(jsonl_path):
    features_list = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            feat = extract_structural_features(data['turns'])
            if feat:
                features_list.append(feat)
    return features_list

def run_discrimination_audit(gen_path=None):
    train_xml = 'Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    predators_file = 'Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
    if gen_path is None:
        gen_path = 'Adversarial-Intent-Telemetry/data/pan_annotated/regenerated_trajectories.jsonl'
    
    with open(predators_file, 'r') as f:
        predator_ids = set(line.strip() for line in f)
        
    print(f"Loading PAN12 features...")
    pan_features = load_pan12_features(train_xml, predator_ids, limit=1000)
    print(f"Loading generated features from {gen_path}...")
    gen_features = load_generated_features(gen_path)
    
    df_pan = pd.DataFrame(pan_features)
    df_pan['label'] = 0 # Empirical
    
    df_gen = pd.DataFrame(gen_features)
    df_gen['label'] = 1 # Generated
    
    df = pd.concat([df_pan, df_gen], ignore_index=True)
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Fill NaNs (like migration_pos if never found)
    X = X.fillna(-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'shallow_mlp': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    }
    
    results = {}
    feature_importance = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'auc': float(auc),
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'confusion_matrix': cm
        }
        
        if name == 'random_forest':
            importances = clf.feature_importances_
            feature_importance = dict(zip(X.columns, [float(i) for i in importances]))

    # Identify top divergent features
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Analyze distributions of top features
    divergence_analysis = {}
    for feat, _ in sorted_importance[:5]:
        divergence_analysis[feat] = {
            'empirical_mean': float(df_pan[feat].mean()),
            'generated_mean': float(df_gen[feat].mean()),
            'empirical_std': float(df_pan[feat].std()),
            'generated_std': float(df_gen[feat].std())
        }
    
    report = {
        'summary': "Lower separability (AUC closer to 0.5) indicates stronger behavioral grounding.",
        'auc_score': float(results['random_forest']['auc']),
        'top_divergent_features': sorted_importance[:5],
        'divergence_analysis': divergence_analysis,
        'interpretation': "High AUC indicates structural distinguishability. Generated trajectories may be more consistent/less noisy than empirical ones."
    }
    
    # Save outputs
    with open('Adversarial-Intent-Telemetry/experiments/results/discrimination_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    with open('Adversarial-Intent-Telemetry/experiments/results/feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    with open('Adversarial-Intent-Telemetry/experiments/results/classifier_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Discrimination audit complete. Results saved in experiments/results/")
    
    # Print summary
    print(f"\nRandom Forest AUC: {results['random_forest']['auc']:.4f}")
    print(f"Top divergent feature: {sorted_importance[0][0]}")

if __name__ == "__main__":
    import sys
    gen_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_discrimination_audit(gen_path)
