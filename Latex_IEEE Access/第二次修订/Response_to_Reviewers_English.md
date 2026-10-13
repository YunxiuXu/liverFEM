# Response to the Editor and Reviewers

**Manuscript ID:** Access-2026-43742

Dear Editor and Reviewers,

Thank you for the careful second review. Our point-by-point responses follow. The main changes are:

1. The haptic update path, actuator response and force mapping are now described precisely (Section III-D).
2. Runtime is reported from a 6,000-frame profiling run with median, percentiles and maximum (Section III-D).
3. The configuration of Experiment 1 is now stated per experiment and per result. While checking it, we found and corrected three reporting errors: the material listed for Experiment 1, the runtime of the proposed method in Table 5, and the data plotted in the displacement figure (now Fig. 6). The conclusions are unchanged.
4. Experiment 2 was re-run with one loading protocol for all three methods, and Table 6 and Fig. 8 now come from the same runs (Sections IV-A and V-A).

All changes are highlighted in the revised manuscript. Figure numbers below refer to the revised manuscript.

---

## Reviewer 1

### R1-1: Haptic update path, latency and force characterization

**Concern:**
> The remaining uncertainty concerns the effective haptic update path. Section III-D reports a 1000-Hz haptic-control loop, while the complete software pipeline produces a new force command at approximately 39.6 FPS, with a mean same-frame latency of 25.2 ms. Please explain what the actuator receives between physics updates: for example, whether the latest command is held, interpolated, filtered, or otherwise resampled. It would also help to distinguish software latency from the true end-to-end response of the physical actuator. The force-sensor characterization should state how commanded and measured force are related, how saturation is handled, and whether repeatability or hysteresis was examined. If these measurements are already available, a compact calibration plot or table would be preferable to reporting only the maximum static force and three frequency-response values.

**Response:**
- *Between physics updates.* New force values arrive at the simulation rate (about 41.2 Hz). The independent 1 kHz control thread holds the latest command as its target (zero-order hold) and applies a slew-rate limiter before each write to the actuator. It does not interpolate or predict. The 1 kHz rate therefore sets actuator command timing, while new force information is limited to about 41.2 Hz and further smoothed by a 10 ms low-pass filter. The text now states the resulting software bandwidth limits. The 41.2 Hz update rate gives a Nyquist limit of 20.6 Hz, and the 10 ms first-order filter has a nominal cutoff of about 15.9 Hz. These values describe the software force-information bandwidth, not a measured end-to-end mechanical bandwidth.
- *Latency.* The former 25.2 ms value is a software frame time, not an end-to-end latency, and is now labeled as such (now 24.3 ms, see R1-3). We separately measured the actuator with a 1000-fps camera. The delay from the serial-command indicator to visible motor rotation was 14.0 ms (SD 1.0 ms, n = 5). Adding the measured and nominal components gives an estimated latency of about 60 ms (24.3 ms frame time, 12.2 ms average wait for the next update, 10 ms filter delay and 14.0 ms motor-motion onset). This estimate excludes tracking, hand-position smoothing, slew limiting and force buildup, and the text states these exclusions.
- *Force characterization.* Fig. 1 reproduces the published current-force measurement for this actuator. Tension increases approximately linearly with current before reaching a plateau near 1.36 N. PWM commands set the motor current, and the PWM-current correspondence was measured before integration. Combined with the current-force curve in Fig. 1, this gives the static command-to-tension mapping. PWM 180, the maximum on a 0 to 180 scale, corresponds to approximately 0.78 A and a static tension of approximately 1.36 N. Simulation force is clamped to the configured range and soft-clipped near its upper bound. A gamma curve (γ = 0.7) then maps it onto 181 PWM levels. The curve represents a single sweep. Repeatability and hysteresis were not tested. To characterize the command range used in practice, we logged output commands during an 80 s bench session. The authors pressed both normal-tissue and lesion regions. Commands ranged from near zero to nearly the configured maximum, with a median of 60% of the configured force range. These values represent commanded output rather than measured force. This session was separate from the expert evaluation, during which force logging was not enabled.

**Changes:** Section III-D (update path, latency, force mapping), Fig. 1 added.

### R1-2: Table 2, Figure 5 and the scope of the comparison

**Concern:**
> Table 2 and Figure 5 require a few precise definitions. "Coupling iterations: 30/150" should indicate which value was used in each experiment, and the relationship between Young's modulus, Poisson's ratio, and the XPBD compliance parameters should be given. Figure 5 labels the horizontal axis "Pulling Force (Accel)," whereas the text refers to loading strengths of 16, 30, and 40. Please define the quantity and its units. Because the methods use different numerical schemes and do not share a common convergence tolerance, the comparison should continue to be described as application-level evidence rather than a standardized demonstration of general superiority.

**Response:**
- *Iterations.* 30 and 150 coupling iterations belong to Experiment 1 (30 is the real-time setting reported in Table 5, and 150 is a convergence reference). Experiment 2 uses 60. Table 2 now lists iterations and materials per experiment.
- *Material.* The previous version incorrectly listed the tissue-level parameters (E = 7,000 Pa, ν = 0.49) for Experiment 1. Experiment 1 used E = 1 × 10^6^ Pa and ν = 0.28. The density was 1,000 kg/m³, and the time step was 0.01 s. Young’s modulus, density, and time step follow the benchmark settings of the original GB-cFEM study. The choice of ν = 0.28 keeps volumetric effects out of the accuracy comparison. Near-incompressible behavior is evaluated in Experiment 2. Because static displacement depends on the ratio of load to stiffness, we also checked proportional scaling with VegaFEM. At the same Poisson ratio, using E = 7,000 Pa with proportionally scaled loads changed the target displacement by less than 2%. We added results at ν = 0.45 and ν = 0.49. The proposed method’s mean errors were 19.5% and 29.3%, respectively. The corresponding errors were 45.2% and 54.5% for XPBD Fast and 32.8% and 32.2% for XPBD Ref. The proposed method therefore retained lower mean errors than both XPBD configurations at these two Poisson ratios.
- *XPBD.* XPBD uses a compliance, the inverse of a solver-specific stiffness parameter, which has no direct mapping to Young's modulus. XPBD Fast and Ref share the same stiffness (0.8) and Poisson's ratio and differ only in substeps.
- *Axis.* The quantity is a per-vertex acceleration with a spatial falloff, in simulation units per second squared, not a force. The axis now reads "Prescribed loading acceleration". While checking this figure we found that it plotted a different run (loads 16, 30, 40) in which the VegaFEM material setting had not taken effect, so it did not match Table 5. We redrew it from the run used for Table 5 (loads 800, 1500, 2000), now Fig. 6. No experiment was re-run for this figure. We also added Fig. 7, showing the final shapes under the largest load.
- *Scope.* The comparison remains described as application-level evidence under a fixed-step schedule without a common convergence tolerance (Section IV-A).

**Changes:** Table 2, Section IV-A (Experiment 1 setup), Section V-A, Fig. 6 redrawn, Fig. 7 added.

### R1-3: Runtime, power and expert observations

**Concern:**
> Runtime and power results are useful, although some wording remains stronger than the measurements warrant. Profiling over 90 frames gives only a short observation of latency. If longer logs exist, reporting a median, a high percentile, and frame-time variability would make the real-time claim more informative. Otherwise, the present measurement can simply be identified as a short profiling run. Similarly, the change from approximately 12 W at idle to 52 W during five minutes of operation supports feasibility on the tested Mac mini, but it is not by itself general evidence of low computational burden or energy efficiency.
>
> No additional participant experiment is requested. The current single-expert walkthrough is acceptable for identifying engineering limitations, provided that observations remain clearly qualitative. Also, consider that statements such as the expert having "extracted physically meaningful stiffness cues" would be safer if expressed as reported or observed perception during the walkthrough, rather than as an objectively measured localization result.

**Response:**
- We re-profiled over 6,000 frames after a 120-frame warmup under the same conditions: mean 24.3 ms (41.2 FPS), median 24.3 ms, SD 1.2 ms, 95th/99th percentiles 26.3/27.2 ms, maximum 43.1 ms.
- The 52 W versus 12 W measurement is now stated only as supporting feasibility on the tested Mac mini. Claims of low computational burden or energy efficiency were removed.
- The Conclusion now states that the expert "reported perceiving physically meaningful stiffness cues".

**Changes:** Section III-D, Section IV-A, Section V-A, Conclusion.

### R1 minor comments

> The response letter describes the supplementary video as approximately 54 seconds long, whereas the supplied file lasts approximately 76.5 seconds.

**Response:** The video is about 76 s. The previous "54 s" was an error.

> The video still uses "tumor" and "target mass." These labels should be changed to "lesion" to match the terminology adopted in the manuscript.

**Response:** The labels in the video are now "lesion".

> Phrases such as "system connectivity" and "soft-tissue mechanical expression" are understandable but not especially natural in English; "closed-loop system integration" and "representation of soft-tissue mechanics" would be clearer.

**Response:** Both phrases were replaced as suggested throughout the manuscript, including Table 8.

---

## Reviewer 2

### R2-1 and R2-2: 1000-Hz loop and force-output validation

**Concern:**
> 1. Please clarify the 1000-Hz haptic loop. The manuscript reports a 1000-Hz haptic-control loop, while the complete hand-input-to-force-command pipeline runs at approximately 39.6 FPS. Please explain what force value is used between consecutive physics updates. Is the latest value held, interpolated, predicted, or updated through another mechanism? The effective haptic-feedback bandwidth should also be clarified.
>
> 2. The force-output validation is still incomplete. The maximum force and frequency-response measurements are useful, but the manuscript should also clarify command-to-force calibration, linearity, resolution, saturation/clipping, and the actual force range used during the palpation task.

**Response:** These comments overlap with R1-1. The held value, effective feedback bandwidth, calibration, linearity, resolution (181 PWM levels), saturation and clipping, and the force range used in practice are addressed in our response to R1-1 and in Section III-D.

### R2-3: Missing anisotropic-response validation

**Concern:**
> Section IV-A lists anisotropic-response validation as one of the technical experiments, but no corresponding result is presented. Please report this experiment or remove it from the stated validation protocol.

**Response:** We re-examined the Experiment 3 data. The isotropic control did not behave isotropically across repeated runs, so the data cannot support an anisotropy claim. We removed this item from the validation protocol.

**Changes:** Section IV-A.

### R2-4: Configuration behind each result in Table 5

**Concern:**
> Different coupling-iteration settings are reported for the proposed method, while Table 5 presents accuracy and runtime together. Please clearly state which configuration produced each displacement error and runtime value so that the reader does not interpret results from different operating conditions as a single accuracy-speed point.

**Response:** The 12.0 ms previously reported for the proposed method was not measured under the configuration that produced the 17.9% error. Re-measured under that same configuration (30 coupling iterations), the runtime is 19.5 ms (51.3 FPS). The Table 5 header now states the configuration of each column, so each error and runtime pair comes from one configuration. The two XPBD errors were previously single-load values. They are now three-load means (52.7% and 42.9%), computed as for the proposed method. The conclusion is unchanged. The proposed method has the lowest error and meets the real-time requirement.

**Changes:** Table 5, Section V-A.

### R2-5: Lesion-localization claims

**Concern:**
> The revised manuscript appropriately acknowledges that only one expert was included and that no quantitative localization metrics were recorded. Please maintain this cautious interpretation throughout the manuscript. If the expert's actual localization success or failure was recorded, the result could also be stated explicitly.

**Response:** We checked the session notes. Without haptics, the expert searched for 2 min 9 s and indicated an incorrect location. With haptics, the expert found the lesion in 36 s. We report these as single-session observations. The lesion positions differed, and the trial order was fixed. We therefore keep the interpretation qualitative.

**Changes:** contribution list (Section I), Section IV-B, Section V-B.

### R2 additional comments

> The whole-machine power result of approximately 52 W versus 12 W idle is interesting, but the phrase "strong evidence" of low computational burden may be too strong without a detailed power-measurement protocol.

**Response:** "Strong evidence" and the low-burden claims were removed. We added the protocol: plug-in wattmeter (SANWA TAP-TST8N, 2% ± 5 W), average from accumulated energy over 5 minutes, idle defined as powered on without the simulation running (Section IV-A).

> Please clarify what "GB-cFEM/PBD" means in the latency breakdown, since GB-cFEM is described as the main simulation solver.

**Response:** This was the full GB-cFEM physics step, whose inter-group coupling uses a PBD-style position constraint. It does not include the XPBD baseline. The coupling step is now defined in Section III-B, and the latency breakdown names its content.

> Please verify the manuscript ID appearing in the response letter, since it differs from the current submission number.

**Response:** Corrected to Access-2026-43742.

> A final proofreading and consistency check is recommended.

**Response:** We checked the consistency of values, figure and table numbers and terminology. This also corrected the default mesh size, previously misstated as 13,800 nodes and 62,897 tetrahedra, to the actual 4,403 nodes and 20,061 tetrahedra. No reported result is affected.

The consistency check also showed that Table 6 and the volume figure (now Fig. 8) came from two different runs of Experiment 2, in which the three methods were not loaded in the same way. We re-ran Experiment 2 with one protocol: the same mesh, E = 7000 Pa, ν = 0.28 and 0.49, the same fixed and pulled slices, and the pulled slice driven toward the same target displacement. Table 6 and Fig. 8 now come from these runs. The proposed method again shows lower maximum and mean volume deviations than XPBD at both Poisson ratios (2.09% and 1.74% versus 7.95% and 4.82% at ν = 0.49). It stays close to the VegaFEM reference (1.83% and 1.26%), and moving from ν = 0.28 to ν = 0.49 reduces the deviation of both the proposed method and VegaFEM.

---

## Reviewer 3

**Concern:**
> Expand User Evaluation: Transition from the preliminary single-expert review ($n=1$) to a multi-participant study (e.g., comparing medical students vs. experienced surgeons) to quantitatively evaluate learning curves, lesion localization accuracy, and diagnostic confidence.
>
> Benchmarking Against Other Physics Engines: Compare the current GB-cFEM implementation against other continuous physics frameworks (e.g., SOFA Framework or Projective Dynamics) to further validate performance, volume preservation, and stability tradeoffs.

**Response:** Thank you for the positive assessment. Two values quoted in the review were corrected in this revision. The frame rate of about 83 FPS is now 51 FPS for the same configuration as the reported accuracy (see R2-4). The 25.2 ms frame time is now 24.3 ms from a 6,000-frame measurement (see R1-3). The conclusions do not change. We agree that a multi-participant study is the necessary next step, as stated in Future Research Directions. That subsection now also plans a comparison with SOFA and Projective Dynamics under matched material, mesh, loading and boundary conditions.

Sincerely,
The Authors
