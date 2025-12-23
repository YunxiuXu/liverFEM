import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
base_path = "/Users/yunxiuxu/Documents/tetfemcpp/out/experiment2"
prop_path = os.path.join(base_path, "20251223_012834/experiment2_volume.csv")
xpbd_path = os.path.join(base_path, "20251223_014628_xpbd/experiment2_volume.csv")
vega_path = os.path.join(base_path, "VegaFEM_20251223_144012/VegaFEM_liver_HD_Low_volume.csv")

# Load data
prop_df = pd.read_csv(prop_path)
xpbd_df = pd.read_csv(xpbd_path)
vega_df = pd.read_csv(vega_path)

# Extend VegaFEM data to 480 steps (4.8s) by padding with last value
def extend_vega_to_480(df):
    """Extend VegaFEM data from 420 steps to 480 steps by padding with last value."""
    extended_rows = []
    # Now VegaFEM only has one runName 'incompressible' with nu=0.49
    for nu in [0.49]:
        nu_data = df[df['nu'] == nu].copy()
        if len(nu_data) == 0:
            continue
        
        run_name = 'incompressible'
        last_row = nu_data.iloc[-1]
        last_volume = last_row['volume']
        last_ratio = last_row['volume_ratio']
        
        # Pad from step 420 to 479 (60 more steps)
        for step in range(420, 480):
            extended_rows.append({
                'runName': run_name,
                'step': step,
                'nu': nu,
                'volume': last_volume,
                'volume_ratio': last_ratio
            })
    
    if extended_rows:
        extended_df = pd.DataFrame(extended_rows)
        vega_df_extended = pd.concat([df, extended_df], ignore_index=True)
        return vega_df_extended
    return df

vega_df = extend_vega_to_480(vega_df)

def add_cumulative_time(df):
    """Adds a cumulative time column to handle resetting sim_time across stages."""
    if 'sim_time' not in df.columns:
        return df
    
    df = df.copy()
    df['total_time'] = 0.0
    
    # Group by run_index (or run_name/poisson) to process each sequence
    for (run_idx, poisson), group in df.groupby(['run_index', 'poisson']):
        current_offset = 0.0
        # Get stages in order (drag first, then hold)
        stages = sorted(group['stage'].unique(), key=lambda x: 0 if x == 'drag' else 1)
        
        for stage in stages:
            mask = (df['run_index'] == run_idx) & (df['poisson'] == poisson) & (df['stage'] == stage)
            stage_data = df[mask]
            
            # Check if this stage's sim_time starts from 0 (needs offset) or continues (already cumulative)
            stage_min_time = stage_data['sim_time'].min()
            
            if stage_min_time == 0.0 or stage_min_time < 0.1:
                # This stage resets time, needs offset
                df.loc[mask, 'total_time'] = stage_data['sim_time'] + current_offset
                if len(stage_data) > 1:
                    dt = stage_data['sim_time'].iloc[1] - stage_data['sim_time'].iloc[0]
                    current_offset += stage_data['sim_time'].max() + dt
                else:
                    current_offset += 0.01
            else:
                # This stage's time is already cumulative (like Proposed Method's hold stage)
                df.loc[mask, 'total_time'] = stage_data['sim_time']
                # Update offset for next stage (if any)
                current_offset = stage_data['sim_time'].max()
    return df

prop_df = add_cumulative_time(prop_df)
xpbd_df = add_cumulative_time(xpbd_df)

# Prepare plot
plt.figure(figsize=(12, 8))

# Define colors and styles
styles = {
    'Proposed Method': {'color': 'blue', 'linewidth': 2},
    'XPBD': {'color': 'orange', 'linestyle': '--'},
    'VegaFEM (GT)': {'color': 'black', 'linestyle': ':'}
}

# --- Plot 1: Volume Preservation Comparison ---
# We show Proposed (0.49) vs XPBD (0.49) vs VegaFEM (0.49)

# 1. Proposed Method (Incompressible)
p_data = prop_df[(prop_df['poisson'] == 0.49)]
if not p_data.empty:
    plt.plot(p_data['total_time'], p_data['volume_ratio'], label='Proposed Method (nu=0.49)', 
             color=styles['Proposed Method']['color'], linewidth=2)

# 2. XPBD (Incompressible)
x_data = xpbd_df[xpbd_df['poisson'] == 0.49]
if not x_data.empty:
    plt.plot(x_data['total_time'], x_data['volume_ratio'], label='XPBD (nu=0.49)', 
             color=styles['XPBD']['color'], linestyle='--')

# 3. VegaFEM (Incompressible - Reference for high poisson)
v_incomp = vega_df[vega_df['nu'] == 0.49]
if not v_incomp.empty:
    v_time = v_incomp['step'] * 0.01
    plt.plot(v_time, v_incomp['volume_ratio'], label='VegaFEM (nu=0.49)', 
             color=styles['VegaFEM (GT)']['color'], linestyle='-.')

plt.axhline(y=1.0, color='red', linestyle='-', alpha=0.3, label='Ideal Volume (V/V0=1.0)')
plt.xlabel('Simulation Time (s)')
plt.ylabel('Volume Ratio ($V/V_0$)')
plt.title('Volume Preservation Comparison under Large Deformation (nu=0.49)')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'volume_preservation_comparison.png'))

print("Experiment 2 plots generated successfully.")

# Calculate statistics for the report
stats = []
# Method mappings for stats
method_info = [
    ('Proposed', prop_df, 'poisson', [0.49]),
    ('XPBD', xpbd_df, 'poisson', [0.49]),
    ('VegaFEM', vega_df, 'nu', [0.49])
]

for method_name, df, nu_col, nus in method_info:
    for nu in nus:
        d = df[df[nu_col] == nu]
        if not d.empty:
            max_dev = (d['volume_ratio'] - 1.0).abs().max() * 100
            avg_dev = (d['volume_ratio'] - 1.0).abs().mean() * 100
            stats.append({
                'Method': method_name,
                'Nu': nu,
                'Max Dev (%)': max_dev,
                'Avg Dev (%)': avg_dev
            })

stats_df = pd.DataFrame(stats)
print(stats_df.to_string())
