import csv
import math
import os

file_path = '/Users/yunxiuxu/Documents/tetfemcpp/out/experiment2/20251218_013656/experiment2_volume.csv'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

baseline_ratios = []
incomp_ratios = []

try:
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                poisson = float(row['poisson'])
                ratio = float(row['volume_ratio'])
                
                if poisson < 0.3:
                    baseline_ratios.append(ratio)
                elif poisson > 0.4:
                    incomp_ratios.append(ratio)
            except ValueError:
                continue

    print(f"Baseline samples: {len(baseline_ratios)}")
    print(f"Incompressible samples: {len(incomp_ratios)}")

    def analyze(ratios, name):
        if not ratios:
            print(f"\n{name}: No data")
            return None
            
        min_val = min(ratios)
        max_val = max(ratios)
        mean_val = sum(ratios) / len(ratios)
        
        # Standard deviation
        variance = sum([((x - mean_val) ** 2) for x in ratios]) / len(ratios)
        std_dev = math.sqrt(variance)
        
        max_dev_pct = max(abs(max_val - 1.0), abs(min_val - 1.0)) * 100
        
        print(f"\n--- {name} ---")
        print(f"Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"Max Deviation: {max_dev_pct:.4f}%")
        print(f"Standard Dev : {std_dev:.6f}")
        return max_dev_pct

    dev_base = analyze(baseline_ratios, "Baseline (nu=0.28)")
    dev_incomp = analyze(incomp_ratios, "Incompressible (nu=0.49)")

    print("\n--- Conclusion ---")
    if dev_incomp is not None and dev_base is not None:
        # Check strictness
        if dev_incomp < 0.1:
            print("PERFECT: Incompressible volume deviation is extremely low (< 0.1%).")
            print("This strongly supports the 'Volume Preservation' claim.")
        elif dev_incomp < 1.0:
            print("GOOD: Incompressible volume deviation is low (< 1.0%).")
            print("This is acceptable for real-time simulation.")
        else:
            print(f"WARNING: Incompressible deviation is {dev_incomp:.2f}%. Check if this is expected.")
            
        # Comparison logic
        # For nu=0.28, volume SHOULD increase under tension (Poisson effect).
        # For nu=0.49, it should stay near 1.0.
        # So we expect dev_incomp to be smaller than dev_base IF dev_base represents significant volume gain.
        
        print(f"Baseline Max Deviation: {dev_base:.4f}% (Expected to be higher due to compressibility)")
        print(f"Incomp   Max Deviation: {dev_incomp:.4f}% (Expected to be near 0)")
        
        if dev_incomp < dev_base:
            print(f"SUCCESS: Incompressible setting reduced volume error.")
        else:
             # It's possible baseline didn't stretch enough to gain much volume, 
             # or incomp failed to hold volume.
             if dev_base < 0.5:
                 print("NOTE: Baseline volume change was also very small. Maybe the drag wasn't aggressive enough?")
             else:
                 print("WARNING: Incompressible mode did not perform better than baseline.")

except Exception as e:
    print(f"Error: {e}")
