# Experiment 2 rerun with a matched loading protocol (2026-10-13)

Source of Table 6 and Fig. 8 (`volume_preservation_matched_nu049.png`) in the revised manuscript.

Protocol, identical for all three methods: liver mesh (4,403 nodes, 20,061 tets, unrotated), E = 7000 Pa, nu = 0.28 or 0.49, gravity off, dt = 0.01 s.
Anchor: posterior 5% slice along x (fixed). Pull: anterior 12% slice along x, tracking spring toward +x target 0.5.
Schedule: settle 120, drag 240, hold 480 steps. Max/mean |V/V0 - 1| taken over drag + hold (720 samples).

| Folder | Method | nu | Notes |
|---|---|---|---|
| proposed_nu028, proposed_nu049 | GB-cFEM (TetgenFEM) | 0.28 / 0.49 | spring accel 4000, 60 coupling iterations; nu set via global `poisson` |
| vega_nu028, vega_nu049 | VegaFEM CLFEM | 0.28 / 0.49 | spring accel 4000 (force = mass * accel), anchor file `liver_e2_anchor5.bou`, config `liver_e2_test.config` |
| xpbd_nu028_nu049 | XPBD | both | spring accel 50 (4000 diverged), 10 substeps, scene `patches/xpbd_scene_e2.json` |

Code changes used for the runs are in `patches/` (headless start hooks, Vega pull and anchor change, XPBD env overrides). They are not applied to the main tree.
Run `python3 plot_matched.py` to regenerate the figure and print the Table 6 values.
