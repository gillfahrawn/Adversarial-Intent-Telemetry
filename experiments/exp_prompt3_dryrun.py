import json
import re
import random
import numpy as np
import pandas as pd
from collections import Counter
from lxml import etree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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
    """Extracts ONLY structural features from a list of turns."""
    if not turns:
        return None
    
    num_turns = len(turns)
    lengths = [len(re.findall(r'\w+', t['content'])) for t in turns]
    entropies = [get_lexical_entropy(t['content']) for t in turns]
    
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
            
    roles = [t.get('role', 'unknown') for t in turns]
    role_counts = Counter(roles)
    if len(role_counts) >= 2:
        counts = list(role_counts.values())
        reciprocity = min(counts) / max(counts)
    else:
        reciprocity = 0
        
    # Escalation timing proxy: first turn > mean length
    avg_l = np.mean(lengths)
    escalation_timing = -1
    for i, l in enumerate(lengths):
        if l > avg_l:
            escalation_timing = i / num_turns
            break

    features = {
        'turn_count': float(num_turns),
        'avg_msg_len': float(np.mean(lengths)),
        'std_msg_len': float(np.std(lengths)),
        'max_msg_len': float(np.max(lengths)),
        'reciprocity_ratio': float(reciprocity),
        'avg_lexical_entropy': float(np.mean(entropies)),
        'hedge_rate': float(hedge_count / num_turns),
        'repair_rate': float(repair_count / num_turns),
        'migration_position': float(migration_pos),
        'escalation_timing': float(escalation_timing),
        'cadence_variance': float(np.std(np.diff(lengths))) if len(lengths) > 1 else 0.0
    }
    return features

def load_pan12_baseline(xml_path, limit=500):
    """Loads baseline features from raw PAN12 XML."""
    features_list = []
    try:
        context = etree.iterparse(xml_path, events=('end',), tag='conversation')
        count = 0
        for event, elem in context:
            messages = elem.xpath('message')
            turns = []
            for msg in messages:
                author = msg.xpath('author/text()')[0] if msg.xpath('author/text()') else 'unknown'
                text = msg.xpath('text/text()')[0] if msg.xpath('text/text()') else ''
                turns.append({'role': author, 'content': text})
            
            feat = extract_structural_features(turns)
            if feat:
                features_list.append(feat)
                count += 1
            
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            if count >= limit:
                break
        return features_list, True
    except Exception as e:
        print(f"Error loading PAN12: {e}")
        return [], False

def load_experimental_dataset(jsonl_path):
    """Loads experimental features from NCMEC trajectories."""
    features_list = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                feat = extract_structural_features(data['turns'])
                if feat:
                    features_list.append(feat)
        return features_list, True
    except Exception as e:
        print(f"Error loading experimental dataset: {e}")
        return [], False

def run_dryrun():
    pan12_xml = 'Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    ncmec_jsonl = 'Adversarial-Intent-Telemetry/data/agentic_ncmec/pan_ncmec_trajectories.jsonl'
    
    integrity_report = {
        "dataset_load_success": False,
        "schema_compatibility": "NOT_CHECKED",
        "feature_extraction_success": False,
        "classifier_pipeline_execution": "NOT_STARTED",
        "warnings": [],
        "recommendation": "NOT_READY"
    }

    # 1. Data Ingestion
    baseline_feats, pan_ok = load_pan12_baseline(pan12_xml, limit=200)
    exp_feats, ncmec_ok = load_experimental_dataset(ncmec_jsonl)
    
    if pan_ok and ncmec_ok:
        integrity_report["dataset_load_success"] = True
    else:
        integrity_report["recommendation"] = "NOT_READY"
        return integrity_report

    # 2. Schema Normalization & Feature Matrix
    df_base = pd.DataFrame(baseline_feats)
    df_exp = pd.DataFrame(exp_feats)
    
    # Assert identical schemas
    if list(df_base.columns) == list(df_exp.columns):
        integrity_report["schema_compatibility"] = "MATCH"
    else:
        integrity_report["schema_compatibility"] = "MISMATCH"
        integrity_report["warnings"].append(f"Schema mismatch: {set(df_base.columns) ^ set(df_exp.columns)}")
    
    integrity_report["feature_extraction_success"] = True
    
    # Check for NaN/Inf
    if df_base.isnull().values.any() or df_exp.isnull().values.any():
        integrity_report["warnings"].append("NaN values detected in features")
    
    # 3. Pipeline Run
    df_base['label'] = 0
    df_exp['label'] = 1
    df = pd.concat([df_base, df_exp], ignore_index=True).fillna(0)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'logistic_regression': LogisticRegression(random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=SEED),
        'shallow_mlp': MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=SEED)
    }
    
    metrics = {}
    try:
        for name, clf in classifiers.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            auc = roc_auc_score(y_test, y_prob)
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            metrics[name] = {
                "auc": float(auc),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1)
            }
        integrity_report["classifier_pipeline_execution"] = "SUCCESS"
    except Exception as e:
        integrity_report["classifier_pipeline_execution"] = "FAILED"
        integrity_report["warnings"].append(f"Classifier error: {e}")

    # 4. Logging & Final Validation
    if integrity_report["classifier_pipeline_execution"] == "SUCCESS":
        integrity_report["recommendation"] = "READY_FOR_PROMPT3"
    
    # Export reports
    with open('Adversarial-Intent-Telemetry/experiments/results/classifier_metrics_dryrun.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    feature_report = {
        "feature_names": list(X.columns),
        "X_baseline_shape": df_base.shape,
        "X_experimental_shape": df_exp.shape,
        "normalization": "StandardScaler"
    }
    with open('Adversarial-Intent-Telemetry/experiments/results/feature_schema_report.json', 'w') as f:
        json.dump(feature_report, f, indent=2)
        
    with open('Adversarial-Intent-Telemetry/experiments/results/pipeline_integrity_report.json', 'w') as f:
        json.dump(integrity_report, f, indent=2)

    return integrity_report

if __name__ == "__main__":
    os.makedirs('experiments/results', exist_ok=True)
    report = run_dryrun()
    print(f"Dry-run complete. Status: {report['recommendation']}")
