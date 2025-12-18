import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
file_path = '/Users/yunxiuxu/Documents/tetfemcpp/out/experiment2/20251218_013656/experiment2_volume.csv'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

try:
    df = pd.read_csv(file_path)
    
    # Filter data by run_index or poisson ratio
    # Baseline: poisson = 0.28
    baseline = df[df['poisson'] < 0.3]
    # Incompressible: poisson = 0.49
    incomp = df[df['poisson'] > 0.4]

    print(f"Baseline samples: {len(baseline)}")
    print(f"Incompressible samples: {len(incomp)}")

    # Analysis
    def analyze_volume(data, name):
        if data.empty:
            print(f"{name}: No data")
            return None
        
        # volume_ratio is V / V0
        ratios = data['volume_ratio']
        max_ratio = ratios.max()
        min_ratio = ratios.min()
        mean_ratio = ratios.mean()
        std_ratio = ratios.std()
        
        # Max deviation from 1.0 (in percentage)
        max_dev_pct = max(abs(max_ratio - 1.0), abs(min_ratio - 1.0)) * 100
        
        print(f"\n--- {name} ---")
        print(f"Range: [{min_ratio:.6f}, {max_ratio:.6f}]")
        print(f"Max Deviation: {max_dev_pct:.4f}%")
        print(f"Standard Dev : {std_ratio:.6f}")
        return max_dev_pct

    dev_base = analyze_volume(baseline, "Baseline (nu=0.28)")
    dev_incomp = analyze_volume(incomp, "Incompressible (nu=0.49)")

    # Conclusion logic
    print("\n--- Conclusion ---")
    if dev_incomp is not None and dev_base is not None:
        if dev_incomp < 0.1:
            print("PERFECT: Incompressible volume deviation is extremely low (< 0.1%).")
            print("This strongly supports the 'Volume Preservation' claim.")
        elif dev_incomp < 0.5:
            print("GOOD: Incompressible volume deviation is low (< 0.5%).")
            print("This is acceptable for real-time simulation papers.")
        else:
            print(f"WARNING: Incompressible deviation is {dev_incomp:.2f}%. Check if this is expected.")
            
        if dev_incomp < dev_base:
            print(f"COMPARISON: Incompressible mode is {dev_base/dev_incomp:.1f}x more stable in volume than Baseline.")
        else:
            print("COMPARISON: Incompressible mode did not show reduced volume change (unexpected).")

except Exception as e:
    print(f"Error analyzing data: {e}")
