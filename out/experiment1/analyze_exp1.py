import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
base_path = "/Users/yunxiuxu/Documents/tetfemcpp/out/experiment1"
ours_path = os.path.join(base_path, "20251218_002006/experiment1_sweep_summary.csv")
xpbd_path = os.path.join(base_path, "20251220_032312_xpbd/experiment1_sweep_summary.csv")
vega_path = os.path.join(base_path, "VegaFEM_20251221_010234/VegaFEM_liver_HD_Low_sweep_summary.csv")

# Load data
ours_df = pd.read_csv(ours_path)
xpbd_df = pd.read_csv(xpbd_path)
vega_df = pd.read_csv(vega_path)

# Filter Vega data for reference (runType == 'reference')
vega_ref = vega_df[vega_df['runType'] == 'reference'].copy()

# Sync data by accel
accels = [800, 1500, 2000]

# Prepare summary data
summary = []

for accel in accels:
    v_disp = vega_ref[vega_ref['accel'] == accel]['targetDisplacementRef'].values[0]
    
    o_row = ours_df[ours_df['pull_accel'] == accel]
    o_disp_fast = o_row['target_disp_fast'].values[0]
    o_time_fast = 12.0 # From proposal: "在 30 次迭代下仅需约 12ms"
    
    x_row = xpbd_df[xpbd_df['pull_accel'] == accel]
    x_disp_fast = x_row['target_disp_fast'].values[0]
    x_disp_ref = x_row['target_disp_reference'].values[0]
    x_time_fast = x_row['time_fast_ms'].values[0]
    x_time_ref = x_row['time_reference_ms'].values[0]
    
    summary.append({
        'Accel': accel,
        'VegaFEM': v_disp,
        'Proposed Method (Fast)': o_disp_fast,
        'XPBD (Fast)': x_disp_fast,
        'XPBD (Ref)': x_disp_ref,
        'Proposed_Error_%': abs(o_disp_fast - v_disp) / v_disp * 100,
        'XPBD_Fast_Error_%': abs(x_disp_fast - v_disp) / v_disp * 100,
        'XPBD_Ref_Error_%': abs(x_disp_ref - v_disp) / v_disp * 100,
        'Proposed_Time': o_time_fast,
        'XPBD_Fast_Time': x_time_fast,
        'XPBD_Ref_Time': x_time_ref
    })

summary_df = pd.DataFrame(summary)

# --- Plot 1: Displacement Comparison ---
plt.figure(figsize=(10, 6))
x = np.arange(len(accels))
width = 0.2

plt.bar(x - 1.5*width, summary_df['VegaFEM'], width, label='VegaFEM (GT)', color='black')
plt.bar(x - 0.5*width, summary_df['Proposed Method (Fast)'], width, label='Proposed Method (Fast, 30 iter)', color='blue')
plt.bar(x + 0.5*width, summary_df['XPBD (Fast)'], width, label='XPBD (Fast, 5 sub)', color='orange')
plt.bar(x + 1.5*width, summary_df['XPBD (Ref)'], width, label='XPBD (Ref, 50 sub)', color='red')

plt.xlabel('Pulling Force (Accel)')
plt.ylabel('Target Displacement')
plt.title('Displacement Comparison: Proposed Method vs XPBD vs VegaFEM')
plt.xticks(x, accels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(base_path, 'displacement_comparison.png'))

# --- Plot 2: Relative Error Comparison ---
plt.figure(figsize=(10, 6))
plt.plot(accels, summary_df['Proposed_Error_%'], marker='o', label='Proposed Method (Fast) Error %', color='blue', linewidth=2)
plt.plot(accels, summary_df['XPBD_Fast_Error_%'], marker='s', label='XPBD (Fast) Error %', color='orange', linestyle='--')
plt.plot(accels, summary_df['XPBD_Ref_Error_%'], marker='^', label='XPBD (Ref) Error %', color='red', linestyle='--')

plt.xlabel('Pulling Force (Accel)')
plt.ylabel('Relative Error (%)')
plt.title('Relative Error to VegaFEM Ground Truth')
plt.xticks(accels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(base_path, 'error_comparison.png'))

# --- Plot 3: Performance (Time) Comparison ---
plt.figure(figsize=(10, 6))
methods = ['Proposed Method', 'XPBD (Fast)', 'XPBD (Ref)']
times = [summary_df['Proposed_Time'].mean(), summary_df['XPBD_Fast_Time'].mean(), summary_df['XPBD_Ref_Time'].mean()]
colors = ['blue', 'orange', 'red']

plt.bar(methods, times, color=colors)
plt.axhline(y=16.67, color='green', linestyle='--', label='60 FPS Limit (16.7ms)')
plt.ylabel('Average Step Time (ms)')
plt.title('Performance Comparison')
plt.legend()
plt.savefig(os.path.join(base_path, 'performance_comparison.png'))

print("Plots generated successfully.")
print(summary_df.to_string())
