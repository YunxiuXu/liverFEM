import matplotlib.pyplot as plt
import numpy as np
import sys

# Hard Coded Data from your successful experiment
# Stiff Axis (E=5.0e6 approx, based on high force)
t_hard = [1.35, 1.36, 1.38, 1.40, 1.42, 1.43, 1.45, 1.46, 1.48, 1.49, 1.50, 1.52, 1.54, 1.56, 1.57, 1.59, 1.61, 1.62, 1.64, 1.66]
f_hard = [0, 15, 12, 23, 43, 2123, 31130, 98197, 142634, 323993, 597994, 967817, 1285680, 1444520, 1440320, 1397560, 1324670, 1247420, 1174370, 1108540]

# Soft Axis (E=1.0e6 approx)
t_soft = [1.46, 1.48, 1.50, 1.51, 1.53, 1.54, 1.56, 1.58, 1.60, 1.61, 1.63, 1.65, 1.66, 1.68, 1.70, 1.71, 1.73, 1.74, 1.76, 1.78, 1.79, 1.80]
f_soft = [0, 61, 117, 169, 214, 252, 284, 310, 1288, 6648, 15545, 20216, 32936, 46099, 60343, 55232, 65255, 64377, 65829, 64871, 61688, 59374]

# Align peaks for better comparison
peak_idx_hard = np.argmax(f_hard)
peak_idx_soft = np.argmax(f_soft)
t_hard_aligned = [t - t_hard[peak_idx_hard] for t in t_hard]
t_soft_aligned = [t - t_soft[peak_idx_soft] for t in t_soft]

# Set up the plot style
plt.style.use('bmh') # Use a clean style
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(t_hard_aligned, f_hard, 'r-o', label='Stiff Axis (Across Fibers)', linewidth=2.5, markersize=6)
ax.plot(t_soft_aligned, f_soft, 'b-s', label='Soft Axis (Along Fibers)', linewidth=2.5, markersize=6)

# Labels and Title
ax.set_title('Haptic Force Feedback: Anisotropic Response Analysis', fontsize=16, pad=20)
ax.set_xlabel('Time relative to Peak Force (s)', fontsize=14)
ax.set_ylabel('Reaction Force Magnitude (N)', fontsize=14)

# Add grid and legend
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

# Annotate the difference
max_hard = max(f_hard)
max_soft = max(f_soft)
ratio = max_hard / max_soft
ax.annotate(f'Peak Ratio ~ {ratio:.1f}x', 
            xy=(0, max_hard), 
            xytext=(0.2, max_hard),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=12)

# Save
output_file = 'Fig5_Anisotropy_Experiment.png'
plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Chart generated successfully: {output_file}")
