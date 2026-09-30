import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15
})

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'data_dec21')
ours = pd.read_csv(os.path.join(data, 'proposed_20251218_002006_sweep_summary.csv'))
xpbd = pd.read_csv(os.path.join(data, 'xpbd_20251220_032312_sweep_summary.csv'))
vega = pd.read_csv(os.path.join(data, 'vega_20251221_010234_sweep_summary.csv'))
vega = vega[vega['runType'] == 'reference']

accels = [800, 1500, 2000]
v = [vega[vega['accel'] == a]['targetDisplacementRef'].values[0] for a in accels]
o = [ours[ours['pull_accel'] == a]['target_disp_fast'].values[0] for a in accels]
xf = [xpbd[xpbd['pull_accel'] == a]['target_disp_fast'].values[0] for a in accels]
xr = [xpbd[xpbd['pull_accel'] == a]['target_disp_reference'].values[0] for a in accels]

plt.figure(figsize=(10, 6.2))
x = np.arange(len(accels))
w = 0.2
plt.bar(x - 1.5 * w, v, w, label='VegaFEM (reference)', color='black')
plt.bar(x - 0.5 * w, o, w, label='Proposed method (30 iter)', color='blue')
plt.bar(x + 0.5 * w, xf, w, label='XPBD Fast (5 substeps)', color='orange')
plt.bar(x + 1.5 * w, xr, w, label='XPBD Ref (50 substeps)', color='red')
plt.xlabel('Prescribed loading acceleration')
plt.ylabel('Target displacement')
plt.xticks(x, accels)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(here, 'displacement_comparison_dec21.png'), dpi=220, bbox_inches='tight')

for a, vv, oo, ff, rr in zip(accels, v, o, xf, xr):
    print(a, vv, oo, ff, rr, 'err%% %.1f %.1f %.1f' % (abs(oo - vv) / vv * 100, abs(ff - vv) / vv * 100, abs(rr - vv) / vv * 100))
