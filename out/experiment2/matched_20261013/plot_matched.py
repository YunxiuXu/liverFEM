import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
DT = 0.01
SETTLE_VEGA = 120


def proposed(nu):
    d = pd.read_csv(os.path.join(here, f'proposed_nu0{int(round(nu * 100))}', 'experiment2_volume.csv'))
    return d['volume_ratio'].to_numpy()


def xpbd(nu):
    d = pd.read_csv(os.path.join(here, 'xpbd_nu028_nu049', 'experiment2_volume.csv'))
    return d[np.isclose(d['poisson'], nu)]['volume_ratio'].to_numpy()


def vega(nu):
    f = glob.glob(os.path.join(here, f'vega_nu0{int(round(nu * 100))}', '*_volume.csv'))[0]
    d = pd.read_csv(f)
    d = d[np.isclose(d['nu'], nu) & (d['step'] >= SETTLE_VEGA)]
    return d['volume_ratio'].to_numpy()


print('Method      nu    max_dev%  mean_dev%  final V/V0')
for nu in (0.28, 0.49):
    for name, fn in (('Proposed', proposed), ('XPBD', xpbd), ('VegaFEM', vega)):
        r = fn(nu)
        dev = np.abs(r - 1.0) * 100
        print(f'{name:10s} {nu:.2f}  {dev.max():7.2f}  {dev.mean():8.2f}   {r[-1]:.4f}')

plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'xtick.labelsize': 16,
                     'ytick.labelsize': 16, 'legend.fontsize': 15})
fig, ax = plt.subplots(figsize=(10, 6.2))
series = (('Proposed method', proposed, 'blue', '-'),
          ('XPBD', xpbd, 'orange', '--'),
          ('VegaFEM (reference)', vega, 'black', '-.'))
for label, fn, color, ls in series:
    r = fn(0.49)
    t = np.arange(len(r)) * DT
    ax.plot(t, r, color=color, linestyle=ls, linewidth=2, label=label)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Ideal volume ($V/V_0=1$)')
ax.axvline(240 * DT, color='gray', linestyle='-', linewidth=0.8, alpha=0.6)
ax.set_ylim(0.975, 1.115)
ax.text(240 * DT + 0.08, 1.004, 'hold phase', va='bottom', fontsize=14, color='gray')
ax.set_xlabel('Time from start of drag (s)')
ax.set_ylabel('Volume ratio $V/V_0$')
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper left', ncol=2, fontsize=13)
fig.tight_layout()
out = os.path.join(here, 'volume_preservation_matched_nu049.png')
fig.savefig(out, dpi=220, bbox_inches='tight')
print('saved', out)
