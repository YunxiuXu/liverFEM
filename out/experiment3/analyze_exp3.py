import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
base_path = "/Users/yunxiuxu/Documents/tetfemcpp/out/experiment3"
data_path = os.path.join(base_path, "20251217_205910/experiment3_force_displacement.csv")

# Load data
df = pd.read_csv(data_path)

# Prepare plot
plt.figure(figsize=(10, 6))

# Define groups to plot
groups = [
    {'material': 'isotropic', 'axis': 'x', 'label': 'Isotropic (X)', 'color': 'gray', 'ls': '--'},
    {'material': 'isotropic', 'axis': 'y', 'label': 'Isotropic (Y)', 'color': 'silver', 'ls': ':'},
    {'material': 'anisotropic', 'axis': 'x', 'label': 'Anisotropic X (Fiber Axis)', 'color': 'red', 'ls': '-'},
    {'material': 'anisotropic', 'axis': 'y', 'label': 'Anisotropic Y (Transverse)', 'color': 'blue', 'ls': '-'},
]

stiffness_results = []

for g in groups:
    mask = (df['material'] == g['material']) & (df['axis'] == g['axis'])
    data = df[mask]
    
    if not data.empty:
        plt.plot(data['actual_displacement'], data['force_target'], 
                 label=g['label'], color=g['color'], linestyle=g['ls'], linewidth=2)
        
        # Calculate stiffness (slope of force-displacement)
        # Use the latter half of data for a more stable linear fit
        fit_data = data[data['actual_displacement'] > data['actual_displacement'].max() * 0.5]
        if len(fit_data) > 1:
            slope, intercept = np.polyfit(fit_data['actual_displacement'], fit_data['force_target'], 1)
            stiffness_results.append({
                'Material': g['material'],
                'Axis': g['axis'],
                'Stiffness': slope
            })

plt.xlabel('Displacement (m)')
plt.ylabel('Target Reaction Force (N)')
plt.title('Force-Displacement Curves: Anisotropic Validation')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(base_path, 'force_displacement_curves.png'))

print("Experiment 3 plots generated successfully.")

# Print stiffness table
stiff_df = pd.DataFrame(stiffness_results)
print(stiff_df.to_string())

# Calculate ratios
if not stiff_df.empty:
    ani_x = stiff_df[(stiff_df['Material'] == 'anisotropic') & (stiff_df['Axis'] == 'x')]['Stiffness'].values[0]
    ani_y = stiff_df[(stiff_df['Material'] == 'anisotropic') & (stiff_df['Axis'] == 'y')]['Stiffness'].values[0]
    iso_x = stiff_df[(stiff_df['Material'] == 'isotropic') & (stiff_df['Axis'] == 'x')]['Stiffness'].values[0]
    iso_y = stiff_df[(stiff_df['Material'] == 'isotropic') & (stiff_df['Axis'] == 'y')]['Stiffness'].values[0]
    
    print(f"\nStiffness Ratio (Ani X / Ani Y): {ani_x / ani_y:.2f}")
    print(f"Stiffness Ratio (Iso X / Iso Y): {iso_x / iso_y:.2f}")
    print(f"Fiber Reinforcement Factor (Ani X / Iso X): {ani_x / iso_x:.2f}")

