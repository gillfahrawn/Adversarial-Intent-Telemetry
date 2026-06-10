#!/usr/bin/env python3
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.svm import LinearSVC

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 20260514
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(".")
DATA_XML = ROOT / "Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
DATA_PRED = ROOT / "Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
NCMEC_JSONL = ROOT / "Adversarial-Intent-Telemetry/data/agentic_ncmec/pan_ncmec_trajectories.jsonl"
NOISY_JSONL = ROOT / "Adversarial-Intent-Telemetry/data/pan_annotated/regenerated_trajectories_noisy.jsonl"
OUT_DIR = ROOT / "Adversarial-Intent-Telemetry/experiments/results"
OUT_BASELINE = OUT_DIR / "detection_baseline.json"
OUT_NCMEC = OUT_DIR / "detection_ncmec.json"

# ── Parameters (DO NOT MODIFY - Hard Constraints) ─────────────────────────────
FPR_TARGET = 0.05
L = 256
BAND_B = 16
BAND_R = 16
VOCAB_SIZE = 4096
FPR_CEILING = 0.005

# ── MinHash Logic (from exp_m3_federation_lift.py) ───────────────────────────
def make_hash_funcs(L, vocab_size, seed=SEED):
    rng = np.random.default_rng(seed)
    p = 2**61 - 1
    a = rng.integers(1, p - 1, size=L, dtype=np.int64)
    b = rng.integers(0, p - 1, size=L, dtype=np.int64)
    return a, b, p

def minhash(token_set, a, b, p):
    if token_set.size == 0: return None
    h = ((a[:, None] * token_set[None, :] + b[:, None]) % p)
    return h.min(axis=1).astype(np.int64)

def build_band_index(sigs, b, r):
    index = [{} for _ in range(b)]
    for sig in sigs:
        if sig is None: continue
        bands = sig.reshape(b, r)
        for bi in range(b):
            index[bi][bands[bi].tobytes()] = True
    return index

def query_band_index(index, sig, b, r):
    if sig is None: return False
    bands = sig.reshape(b, r)
    for bi in range(b):
        if bands[bi].tobytes() in index[bi]:
            return True
    return False

def text_to_tokens(text):
    tokens = set()
    for i in range(len(text) - 3):
        gram = text[i:i + 4]
        digest = hashlib.md5(gram.encode("utf-8", errors="replace")).digest()
        tokens.add(int.from_bytes(digest[:4], "little") % VOCAB_SIZE)
    return np.array(sorted(tokens), dtype=np.int64)

def conv_to_sig(messages, a_hf, b_hf, p):
    full_text = " ".join(txt for _, txt in messages)
    tokens = text_to_tokens(full_text)
    return minhash(tokens, a_hf, b_hf, p)

# ── TF-IDF Sequence Logic (from exp_trajectory_lift.py) ─────────────────────
def conv_to_weighted_vec(msgs, vect):
    L_conv = len(msgs)
    texts = [txt if txt else " " for _, txt in msgs]
    if not texts: return None
    X = vect.transform(texts)
    weights = np.array([1.0 + 0.5 * (i / max(L_conv - 1, 1)) for i in range(L_conv)], dtype=np.float64)
    w_sum = weights.sum()
    weighted = X.multiply(weights[:, np.newaxis])
    return np.asarray(weighted.sum(axis=0)) / w_sum

def operating_point_at_fpr(y_true, scores, fpr_target):
    fpr_arr, tpr_arr, thr_arr = roc_curve(y_true, scores)
    mask = fpr_arr <= fpr_target
    if not mask.any(): idx = 0
    else: idx = int(np.argmax(tpr_arr * mask))
    threshold = float(thr_arr[idx]) if idx < len(thr_arr) else 0.0
    actual_fpr = float(fpr_arr[idx])
    recall_op = float(tpr_arr[idx])
    preds = (scores >= threshold).astype(int)
    prec = float((preds & y_true).sum() / max(preds.sum(), 1))
    f1 = float(2 * prec * recall_op / max(prec + recall_op, 1e-9))
    return threshold, recall_op, prec, f1, actual_fpr

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_predators(path):
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}

def parse_pan12(xml_path, predators):
    conversations = []
    for _event, elem in ET.iterparse(str(xml_path), events=["end"]):
        if elem.tag == "conversation":
            conv_id = elem.get("id", "")
            messages = []
            for msg in elem.findall("message"):
                author = (msg.findtext("author") or "").strip()
                text = (msg.findtext("text") or "").strip()
                messages.append((author, text))
            label = 1 if any(a in predators for a, _ in messages) else 0
            conversations.append((conv_id, label, messages))
            elem.clear()
    return conversations

def stratified_split(conversations, test_frac=0.2, seed=SEED):
    conversations = sorted(conversations, key=lambda x: x[0])
    pos = [c for c in conversations if c[1] == 1]
    neg = [c for c in conversations if c[1] == 0]
    rng = np.random.default_rng(seed)
    pos_perm = rng.permutation(len(pos))
    neg_perm = rng.permutation(len(neg))
    n_pos_tr = int(len(pos) * (1.0 - test_frac))
    n_neg_tr = int(len(neg) * (1.0 - test_frac))
    train = [pos[i] for i in pos_perm[:n_pos_tr]] + [neg[i] for i in neg_perm[:n_neg_tr]]
    test = [pos[i] for i in pos_perm[n_pos_tr:]] + [neg[i] for i in neg_perm[n_neg_tr:]]
    return train, test

def load_perturbed_dataset():
    paths = [NCMEC_JSONL, NOISY_JSONL]
    all_convs = []
    for p in paths:
        if not p.exists(): continue
        with open(p) as f:
            for line in f:
                data = json.loads(line)
                conv_id = data.get("trajectory_id", f"perturbed_{len(all_convs)}")
                label = 1 # All in these sets are positives
                turns = data.get("turns", [])
                msgs = [(t.get("role", "unknown"), t.get("content", "")) for t in turns]
                all_convs.append((conv_id, label, msgs))
    return all_convs

# ── Main Execution ───────────────────────────────────────────────────────────
def main():
    # 1. Load Baseline Data
    print("Loading PAN12 Baseline data...")
    predators = load_predators(DATA_PRED)
    conversations = parse_pan12(DATA_XML, predators)
    train, test = stratified_split(conversations)
    test_neg = [c for c in test if c[1] == 0]
    
    # 2. Train Detector Pipeline on Baseline
    print("Training detector pipeline on Baseline (PAN12 Train)...")
    # TF-IDF fitting
    train_messages_text = [txt if txt else " " for _, _, msgs in train for _, txt in msgs]
    vectorizer = TfidfVectorizer(max_features=10000, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vectorizer.fit(train_messages_text)
    
    # Baseline LinearSVC (per-message)
    train_msg_texts = []
    train_msg_labels = []
    for _, label, msgs in train:
        for _, txt in msgs:
            train_msg_texts.append(txt if txt else " ")
            train_msg_labels.append(label)
    X_msg_train = vectorizer.transform(train_msg_texts)
    y_msg_train = np.array(train_msg_labels, dtype=int)
    svc = LinearSVC(random_state=SEED)
    svc.fit(X_msg_train, y_msg_train)
    
    # Sequence LogisticRegression (position-weighted)
    X_conv_train_rows = []
    y_conv_train = []
    for _, label, msgs in train:
        vec = conv_to_weighted_vec(msgs, vectorizer)
        if vec is not None:
            X_conv_train_rows.append(vec.flatten())
            y_conv_train.append(label)
    X_conv_train = np.vstack(X_conv_train_rows)
    y_conv_train = np.array(y_conv_train, dtype=int)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    lr.fit(X_conv_train, y_conv_train)
    
    # M3 MinHash Pool
    a_hf, b_hf, p_hf = make_hash_funcs(L=L, vocab_size=VOCAB_SIZE, seed=SEED)
    train_pos = [c for c in train if c[1] == 1]
    train_pos_sigs = [conv_to_sig(msgs, a_hf, b_hf, p_hf) for _, _, msgs in train_pos]
    fed_index = build_band_index(train_pos_sigs, BAND_B, BAND_R)

    # 3. Evaluate on Baseline
    print("Evaluating on Baseline (PAN12 Test)...")
    results_baseline = evaluate_on_set(test, svc, lr, vectorizer, fed_index, a_hf, b_hf, p_hf, "Baseline (PAN12 Test)")
    
    # 4. Evaluate on Perturbed
    print("Evaluating on Perturbed (NCMEC + noisy)...")
    perturbed_pos = load_perturbed_dataset()
    perturbed_set = perturbed_pos + test_neg # Combine perturbed positives with baseline negatives
    results_ncmec = evaluate_on_set(perturbed_set, svc, lr, vectorizer, fed_index, a_hf, b_hf, p_hf, "Perturbed (NCMEC/Realism)")
    
    # 5. Save and Summary
    with open(OUT_BASELINE, 'w') as f: json.dump(results_baseline, f, indent=2)
    with open(OUT_NCMEC, 'w') as f: json.dump(results_ncmec, f, indent=2)
    
    print("\n=== Comparative Evaluation Summary ===")
    print(f"{'Metric':<25} | {'Baseline':<10} | {'Perturbed':<10} | {'Delta':<10}")
    print("-" * 65)
    for m in ["AUC (Sequence)", "F1 @ FPR=0.05", "Recall @ FPR=0.05", "M3 Recall"]:
        v_b = results_baseline["metrics"][m]
        v_p = results_ncmec["metrics"][m]
        print(f"{m:<25} | {v_b:<10.4f} | {v_p:<10.4f} | {v_p-v_b:<10.4f}")
    
    print(f"\nResults saved to:\n  - {OUT_BASELINE}\n  - {OUT_NCMEC}")

def evaluate_on_set(dataset, svc, lr, vectorizer, fed_index, a_hf, b_hf, p_hf, name):
    labels = np.array([c[1] for c in dataset], dtype=int)
    
    # Baseline scores
    base_scores = []
    for _, _, msgs in dataset:
        texts = [txt if txt else " " for _, txt in msgs]
        if not texts: base_scores.append(0.0)
        else: base_scores.append(float(svc.decision_function(vectorizer.transform(texts)).mean()))
    base_scores = np.array(base_scores)
    
    # Sequence scores
    seq_scores = []
    for _, _, msgs in dataset:
        vec = conv_to_weighted_vec(msgs, vectorizer)
        if vec is not None: score = float(lr.decision_function(vec.reshape(1, -1))[0])
        else: score = 0.0
        seq_scores.append(score)
    seq_scores = np.array(seq_scores)
    
    # M3 hits
    tp = fp = fn = tn = 0
    for _, label, msgs in dataset:
        sig = conv_to_sig(msgs, a_hf, b_hf, p_hf)
        hit = query_band_index(fed_index, sig, BAND_B, BAND_R)
        if label == 1:
            if hit: tp += 1
            else: fn += 1
        else:
            if hit: fp += 1
            else: tn += 1
    m3_recall = tp / max(tp + fn, 1)
    m3_fpr = fp / max(fp + tn, 1)
    
    # Operating point metrics
    thr, recall, prec, f1, act_fpr = operating_point_at_fpr(labels, seq_scores, FPR_TARGET)
    
    return {
        "dataset": name,
        "metrics": {
            "AUC (Baseline)": float(roc_auc_score(labels, base_scores)),
            "AUC (Sequence)": float(roc_auc_score(labels, seq_scores)),
            "F1 @ FPR=0.05": f1,
            "Recall @ FPR=0.05": recall,
            "Precision @ FPR=0.05": prec,
            "Actual FPR": act_fpr,
            "M3 Recall": m3_recall,
            "M3 FPR": m3_fpr
        },
        "counts": {
            "total": len(dataset),
            "positives": int(labels.sum()),
            "negatives": int(len(labels) - labels.sum())
        }
    }

if __name__ == "__main__":
    main()
