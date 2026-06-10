import pandas as pd
import numpy as np

def analyze_boundaries():
    df = pd.read_csv("Adversarial-Intent-Telemetry/experiments/results/continuity_summary.csv")
    
    # We define "recovered" as:
    # LSH: recovery_rate_stabilized > 0.5
    # Cosine: cosine_stabilized > 0.85 (picking a reasonable threshold for "stable")
    
    df['lsh_recovered'] = df['recovery_rate_stabilized'] > 0.5
    df['cosine_recovered'] = df['cosine_stabilized'] > 0.85
    
    print("--- Recovery Classification Agreement ---")
    agreement = (df['lsh_recovered'] == df['cosine_recovered']).mean()
    print(f"Total Agreement: {agreement:.2%}")
    
    lsh_pos = df['lsh_recovered'].sum()
    cos_pos = df['cosine_recovered'].sum()
    print(f"LSH Positive (Recovered): {lsh_pos} / {len(df)}")
    print(f"Cosine Positive (Recovered): {cos_pos} / {len(df)}")
    
    # Check for LSH-only recovery
    lsh_only = ((df['lsh_recovered'] == True) & (df['cosine_recovered'] == False)).sum()
    # Check for Cosine-only recovery
    cos_only = ((df['lsh_recovered'] == False) & (df['cosine_recovered'] == True)).sum()
    
    print(f"LSH-only recovery: {lsh_only}")
    print(f"Cosine-only recovery: {cos_only}")
    
    print("\n--- Phase Boundary Estimation (Threshold 0.5) ---")
    # For a fixed eg, find the ir where it crosses 0.5
    for eg in sorted(df['entropy_growth_rate'].unique()):
        sub = df[df['entropy_growth_rate'] == eg]
        if sub.empty: continue
        
        lsh_cross = sub[sub['recovery_rate_stabilized'] > 0.5]['invariant_ratio'].min()
        cos_cross = sub[sub['cosine_stabilized'] > 0.5]['invariant_ratio'].min()
        
        print(f"EG={eg}: LSH Boundary IR={lsh_cross}, Cosine Boundary IR={cos_cross}")

    print("\n--- Migration Shock Recovery (LSH) ---")
    for eg in sorted(df['entropy_growth_rate'].unique()):
        sub = df[df['entropy_growth_rate'] == eg]
        lsh_cross = sub[sub['migration_shock_recovery'] > 0.5]['invariant_ratio'].min()
        print(f"EG={eg}: LSH Shock Boundary IR={lsh_cross}")

if __name__ == "__main__":
    analyze_boundaries()
