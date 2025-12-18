import csv
import os

# 使用相对路径，假设我们在项目根目录
file_path = 'out/experiment2/20251218_013656/experiment2_volume.csv'

if not os.path.exists(file_path):
    # 尝试绝对路径
    file_path = '/Users/yunxiuxu/Documents/tetfemcpp/out/experiment2/20251218_013656/experiment2_volume.csv'
    if not os.path.exists(file_path):
        print(f"Error: CSV file not found at {file_path}")
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
        # Max deviation from 1.0 (in percentage)
        max_dev_pct = max(abs(max_val - 1.0), abs(min_val - 1.0)) * 100
        
        print(f"\n--- {name} ---")
        print(f"Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"Max Deviation: {max_dev_pct:.4f}%")
        return max_dev_pct

    dev_base = analyze(baseline_ratios, "Baseline (nu=0.28)")
    dev_incomp = analyze(incomp_ratios, "Incompressible (nu=0.49)")

    print("\n--- Conclusion ---")
    if dev_incomp is not None and dev_base is not None:
        if dev_incomp < 0.1:
            print("PERFECT: Incompressible error < 0.1%. Strong evidence.")
        elif dev_incomp < 0.5:
            print("GOOD: Incompressible error < 0.5%. Acceptable.")
        else:
            print(f"NOTE: Incompressible error is {dev_incomp:.2f}%.")
            
        if dev_incomp < dev_base:
             print("SUCCESS: Incompressible mode is more stable than Baseline.")
        else:
             print("WARNING: Incompressible mode deviation is not smaller than Baseline.")

except Exception as e:
    print(f"Error: {e}")
