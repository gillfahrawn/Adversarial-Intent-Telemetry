import pandas as pd
import numpy as np

def analyze_tri_metric():
    df = pd.read_csv("Adversarial-Intent-Telemetry/experiments/results/continuity_summary.csv")
    
    # Thresholds for "Recovery"
    # LSH: > 0.5 (standard)
    # Cosine: > 0.85 (observed high baseline)
    # Euclidean: > 0.05 (based on exp(-dist) with dist around 2-3)
    # Let's check ranges first to be sure
    print(f"Cosine Range: [{df['cosine_stabilized'].min():.4f}, {df['cosine_stabilized'].max():.4f}]")
    print(f"Euclidean Range: [{df['euclidean_stabilized'].min():.4f}, {df['euclidean_stabilized'].max():.4f}]")
    
    # Adaptive thresholds for the sake of comparison logic (if range is narrow)
    cos_thresh = 0.90
    euc_thresh = 0.05
    
    df['lsh_rec'] = (df['recovery_rate_stabilized'] > 0.5).astype(int)
    df['cos_rec'] = (df['cosine_stabilized'] > cos_thresh).astype(int)
    df['euc_rec'] = (df['euclidean_stabilized'] > euc_thresh).astype(int)
    
    print("\n--- Tri-Metric Classification Agreement ---")
    metrics = ['lsh_rec', 'cos_rec', 'euc_rec']
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            m1, m2 = metrics[i], metrics[j]
            agreement = (df[m1] == df[m2]).mean()
            print(f"Agreement({m1}, {m2}): {agreement:.2%}")

    print("\n--- Rank-Order Correlation (Spearman) ---")
    corr_lc = df['recovery_rate_stabilized'].corr(df['cosine_stabilized'], method='spearman')
    corr_le = df['recovery_rate_stabilized'].corr(df['euclidean_stabilized'], method='spearman')
    corr_ce = df['cosine_stabilized'].corr(df['euclidean_stabilized'], method='spearman')
    
    print(f"Correlation(LSH, Cosine): {corr_lc:.4f}")
    print(f"Correlation(LSH, Euclidean): {corr_le:.4f}")
    print(f"Correlation(Cosine, Euclidean): {corr_ce:.4f}")

    print("\n--- 3-Way Consistency Summary ---")
    # Count occurrences of (1,1,1), (0,0,0) etc.
    df['trip'] = df['lsh_rec'].astype(str) + df['cos_rec'].astype(str) + df['euc_rec'].astype(str)
    counts = df['trip'].value_counts().to_dict()
    print(f"Pattern Counts (LSH, Cos, Euc): {counts}")
    
    # Identify outlier
    # Agreement between Cos and Euc is the "latent truth" baseline
    latent_agreement = (df['cos_rec'] == df['euc_rec']).mean()
    print(f"Latent Consensus (Cos == Euc): {latent_agreement:.2%}")
    
    lsh_consensus_with_latent = (df['lsh_rec'] == df['cos_rec']).mean()
    print(f"LSH Alignment with Latent: {lsh_consensus_with_latent:.2%}")

if __name__ == "__main__":
    analyze_tri_metric()
