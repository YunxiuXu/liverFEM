# Response to the Editor and Reviewers

**Manuscript ID:** Access-2026-15540

Dear Editor and Reviewers,

We sincerely appreciate the careful and constructive evaluation of our manuscript. We have revised the manuscript extensively in response to all of the comments. The principal changes include: (1) reframing the study as a system-integration and technical-feasibility study rather than evidence of training effectiveness; (2) adding quantitative characterization of the haptic-rendering pipeline; (3) clarifying the design and limitations of the formative single-expert evaluation; (4) providing additional information needed to reproduce the solver comparisons; (5) strengthening the biomechanical basis of the liver and lesion model; (6) expanding the comparison with related liver and abdominal palpation systems; and (7) carefully editing the manuscript for consistency, readability, and appropriate interpretation of the results.

Our point-by-point responses are provided below. Reviewer comments are shown as headings, followed by our response and a description of the corresponding changes to the manuscript.

---

## Response to Reviewer 1

### Comment R1-1: Reframe the contribution of the study

The manuscript should be positioned as a system-integration and technical-feasibility study rather than as evidence of training effectiveness. An evaluation involving a single surgical expert can provide formative feedback, but it cannot support strong claims regarding clinical-grade training, educational impact, or learning efficiency. The Abstract, Discussion, and Conclusion should be revised accordingly.

**Response:**  
We agree and have substantially narrowed the scope of our claims. The Abstract now explicitly frames the work as “a system-integration and technical-feasibility study rather than as evidence of training effectiveness.” It also states that “this single-participant walkthrough provides diagnostic, exploratory feedback and does not by itself support claims of clinical-grade training efficacy, educational impact, or learning outcomes.”

At the beginning of the Discussion subsection “From Algorithm Validation to System-Prototype Evaluation,” we added the following statement:

> It should be emphasized that this single-expert, walkthrough-based evaluation is formative and diagnostic in nature: it is designed to expose engineering boundaries rather than to establish clinical-grade training efficacy, educational impact, or learning outcomes, and none of the observations discussed below should be read as demonstrating such effects.

We also removed overly strong statements from the Conclusion, including the previous claims that the system “matches the actual demands of modern medical education” and “meets the educational standards of improving learning efficiency.” The revised Conclusion now states:

> Establishing actual training effectiveness, learning-efficiency gains, or clinical-grade fidelity will require the structured, multi-participant, novice-and-expert study outlined in the Future Research Directions subsection.

**Changes in the manuscript:**  
Location: Abstract (p. 1); Section VI-A, “From Algorithm Validation to System-Prototype Evaluation” (p. 11); and Section VII, “Conclusion” (p. 12). These passages have been revised to consistently limit the contribution to technical feasibility, system integration, and preliminary formative feedback.

### Comment R1-2: Provide technical details and quantitative validation of the haptic-rendering pipeline

Please report the device force range, force-calibration method, saturation behavior, end-to-end latency, filtering or smoothing method, and synchronization between the physics update and the 1000-Hz haptic loop. Measured force-output data should be provided where possible rather than only a computed force-mapping equation.

**Response:**  
We agree. This comment overlaps with Comment R3-M3, and the corresponding revisions address both comments. We expanded the subsection “1-DOF Haptic Rendering Strategy” to report the following measurements and implementation details:

- Bench characterization with a force sensor measured a maximum static tensile force of 1.36 N.
- To characterize the frequency response, a DC current bias equal to half the peak-to-peak current was added, and the frequency was swept from 10 to 500 Hz while keeping the peak-to-peak current constant. A resonance was observed near 150 Hz; the peak-to-peak force variation reached 0.434 N at 140 Hz, approximately 3.3 times the 0.132 N measured at 10 Hz. At 500 Hz, the peak-to-peak force variation was 0.123 N, corresponding to 0.93 times the 10-Hz reference value.
- The haptic-control loop operates at 1000 Hz, and force commands are transmitted through a 2,000,000-bps serial connection in less than 1 ms per packet.
- On the default liver mesh containing 13,800 nodes and 62,897 tetrahedra, profiling over 90 frames produced a mean same-frame latency of 25.2 ms (39.6 FPS; peak latency, 30.1 ms) from the hand-input update to force-command issuance. The mean latency comprised 22.2 ms for GB-cFEM/PBD, 1.4 ms for contact processing and force mapping, and 0.3 ms for rendering.
- Device forces are low-pass filtered using a 10-ms time constant, while hand positions are smoothed using a 30-ms time constant to suppress transients. A PWM slew-rate limiter is also applied to prevent cable-snap artifacts at contact onset.

**Changes in the manuscript:**  
Location: Section III-D, “1-DOF Haptic Rendering Strategy” (p. 4). This subsection now includes the measured maximum static tensile force, the fixed peak-to-peak current and DC current bias used for the 10–500 Hz frequency-response measurement, resonance amplification relative to the 10-Hz reference, the 500-Hz relative output, software-pipeline latency, communication latency, filtering parameters, position smoothing, and PWM slew-rate limiting.

### Comment R1-3: Clarify the design of the formative evaluation

Please explain how the lesion positions were selected for the haptics-off and haptics-on trials, whether the trials were comparable in lesion size, depth, and difficulty, and how the fixed trial order may have introduced learning or familiarization effects. If possible, please report objective measures such as localization accuracy, localization time, number of presses, hand trajectory, confidence, or error distance.

**Response:**  
We have clarified both the degree of comparability between the trials and the limitations of the evaluation design. The system contains three switchable lesion presets. All three use the same local radius and the same increase in Young’s modulus. The two trial positions were therefore matched in lesion size and stiffness contrast by construction.

However, the three presets are located in different anatomical regions of the liver model. They were not independently verified to be matched in depth beneath the surface or in local geometric complexity. We therefore cannot exclude a residual difference in task difficulty between the two trials. In addition, the presets were cycled in a fixed, deterministic order rather than randomized or counterbalanced. A learning or familiarization effect between the haptics-off and haptics-on trials consequently cannot be excluded. The observed change in search strategy is now described as a plausible, suggestive qualitative observation rather than as a controlled causal effect of haptic feedback.

The present single-session diagnostic walkthrough was not instrumented to record localization accuracy, time to localization, number of exploratory presses, hand trajectory, confidence, or error distance automatically. Behavioral information was recorded qualitatively through observation and interview. We now report this explicitly as a limitation. As explained further in our response to Comment R3-M4, the existing interaction pipeline already makes per-frame fingertip proxy positions and contact states available. In future work, these data will be logged using predefined operational measures in a randomized, multi-participant haptics-on/off study.

**Changes in the manuscript:**  
Location: Section IV-B, “Formative Expert Evaluation and Observation of Lesion-Search Behavior” (pp. 7–8), and Section V-B2, “Behavioral Observation in the Lesion-Localization Task” (p. 10). These passages now explain the matched lesion parameters, unmatched anatomical factors, fixed trial order, potential learning and familiarization effects, absence of automated behavioral logging, and appropriately limited interpretation of the observed strategy change.

### Comment R1-4: Provide reproducible information for the XPBD and VegaFEM comparisons

Please specify the substeps, iterations, time step, material parameters, mesh size, loading conditions, boundary constraints, and termination or convergence criteria. Conclusions should be stated cautiously if the mesh sizes used by the methods are not fully matched.

**Response:**  
We agree. This comment overlaps with Comment R3-M6, and the same revisions address both comments. We verified the settings used for all three methods in the large-deformation accuracy comparison (Experiment 1) and the volume-preservation comparison (Experiment 2). The subsection “Application-Level Technical Validation Based on a Specific Liver Model” now includes a table entitled “Solver Settings Used in the Displacement-Accuracy and Volume-Preservation Comparisons.”

The revised text and table specify that all three methods used the same liver mesh and fixed-node definitions. They also report the common time step of 0.01 s, the material parameters ($E=7000$ Pa and $\nu=0.49$, except where Poisson’s ratio was varied in the volume-preservation experiment), 30/150 GB-cFEM coupling iterations, 5/50 XPBD substeps with `maxIterations = 1`, and 1/2 VegaFEM Newton iterations. The loading protocols, boundary conditions, and termination criteria are now described in detail.

We also expanded the Results subsection “Real-Time Performance and Scalability” to specify the mesh-resolution sweep, thread configurations (one thread and ten threads on a processor with up to 12 cores), and the numbers of warm-up and measured frames.

**Changes in the manuscript:**  
Location: Section IV-A, “Application-Level Technical Validation Based on a Specific Liver Model” (pp. 6–7), including Table 2 (p. 6), and Section V-A3, “Real-Time Performance and Scalability” (pp. 9–10). A new solver-settings table and accompanying text have been added, and the performance and scalability methodology has been expanded.

### Comment R1-5: Complete a thorough formatting and language check

Please correct spacing, hyphenation, conversion, and related formatting problems throughout the manuscript.

**Response:**  
We carefully proofread the complete manuscript and checked the LaTeX source for spacing, hyphenation, conversion, formatting, and terminology consistency. During the final pass, we reduced the column padding in Table 2 so that it fits within the column width, replaced “multi-sample user study” with “multi-participant user study,” aligned the Likert-result wording with the exact Table 8 label “system feasibility and potential,” and removed wording in Related Work that could imply that a single-expert evaluation validated usability.

**Changes in the manuscript:**  
Location: Section II-D, “Position of This Work” (p. 3); Table 2 in Section IV-A (p. 6); the paragraph following Table 8 in Section V-B3 (p. 10); and Section VI-C, “Future Research Directions” (p. 11). Formatting, terminology, and language consistency have been corrected, including the Table 2 column width.

### Additional Comment R1-6: Clarify the contribution relative to the authors’ previous work

The manuscript should clearly state that GB-cFEM and the fingertip haptic device were previously published separately, and that the contribution of the present work is their integration with hand tracking, a liver-lesion localization task, and force mapping in a functioning real-time closed-loop system.

**Response:**  
We agree. This comment overlaps with Comment R3-0. In the “Position of This Work” subsection of Related Work, we added the following statement:

> The GB-cFEM solver used here has previously been validated as a standalone real-time large-deformation algorithm (Wang et al., 2025), and the fingertip haptic device used here has previously been validated as a standalone lightweight actuator (Xu et al., 2025); neither prior study integrated the two with hand tracking into a running, task-oriented closed loop, which is what the present work adds.

This statement explicitly distinguishes the previously published standalone components from the incremental contribution of the present study: their integration with hand tracking, a lesion-localization task, and force mapping in a functioning real-time interaction loop.

**Changes in the manuscript:**  
Location: Section II-D, “Position of This Work” (p. 3). This subsection now explicitly identifies the prior standalone contributions and the new system-level contribution of the present manuscript.

### Additional Comment R1-7: Expand the related literature and comparisons

Please strengthen the literature review concerning validation methods for medical simulators, haptic-rendering evaluation, user-study design for surgical simulation, and recent liver or abdominal palpation systems. General educational literature should be used only as motivation, not as evidence that the present system has already achieved an educational effect.

**Response:**  
We expanded the literature review in several parts of the manuscript. This comment overlaps with Comment R2-4 regarding liver and abdominal palpation systems.

In “Haptic Device Degrees of Freedom and Low-Dimensional Haptic Rendering,” we added two relevant studies. Fazlollahi et al. provide a quantitative approach for evaluating the force-feedback performance of haptic devices, while Hafiz et al. provide a psychophysical basis for users’ discrimination of stiffness differences of the type represented in our system.

In “Formative Expert Evaluation and Observation of Lesion-Search Behavior,” we added three groups of methodological references. Cheng et al. and da Silva et al. support pilot testing and iterative refinement before large-scale validity studies. Cook et al. and Borgersen et al. emphasize that validity evidence must be accumulated for a specific intended use and cannot be established by the opinion of a single expert. Xu et al. show that even a reasonably designed abdominal-palpation simulation task may not automatically yield a reliable measure of competence. Together, these studies support our decision to conduct a formative single-expert evaluation first and to avoid broad validity claims at the present stage.

In the discussion of the Likert-scale results, we cite Sullivan et al. regarding the appropriate interpretation of ordinal-scale data. We now explicitly state that ratings from one participant are not statistically generalizable usability evidence and do not validate training effectiveness.

The general educational references in the Introduction, including Wisniewski et al., remain limited to motivating the potential value of the work. They are not used as evidence that the present system has demonstrated an educational benefit.

**Changes in the manuscript:**  
Location: Sections II-B and II-C (p. 3); Section IV-B (pp. 7–8); and the paragraph following Table 8 in Section V-B3 (p. 10). These passages now cover haptic characterization, psychophysics, simulation validation, pilot evaluation, abdominal palpation, cautious Likert-scale interpretation, and the motivational role of general educational literature.

---

## Response to Reviewer 2

### Comment R2-1: Expand the discussion of clinical relevance and potential educational impact without presenting potential value as a demonstrated effect

**Response:**  
We agree and rewrote the final paragraph of the Conclusion. The revised text explains that the system can be engineered to operate in real time on compact, consumer-grade hardware and that a surgical expert was able to extract physically meaningful stiffness cues during a brief walkthrough. These results support the technical feasibility and system-level connectivity of the proposed architecture. They suggest, without by themselves demonstrating, that this type of system may have future value for accessible training with information-rich feedback.

We now explicitly state that actual training effectiveness, gains in learning efficiency, and clinical-grade fidelity must be evaluated in a structured, multi-participant study involving both novices and experts. A similarly cautious statement has been added to the Abstract.

**Changes in the manuscript:**  
Location: Abstract (p. 1) and Section VII, “Conclusion” (p. 12). These passages now distinguish demonstrated technical feasibility from potential, but not yet demonstrated, clinical and educational value.

### Comment R2-2: Explain the use of a single-expert formative evaluation more fully

Please clarify that the purpose of the evaluation was to identify engineering problems rather than to validate training effectiveness.

**Response:**  
We expanded the beginning of “Formative Expert Evaluation and Observation of Lesion-Search Behavior” to explain the rationale for this study design. At the present development stage, the first objective was to determine whether the simplified anatomical constraints, current material model, and 1-DOF force channel could produce mechanically and clinically sensible feedback before undertaking a resource-intensive novice-and-expert study.

The evaluation is now explicitly defined as a formative expert evaluation intended to identify system feasibility and engineering boundaries, not to measure training effectiveness, skill acquisition, or learning outcomes. All conclusions derived from this evaluation are accordingly described as preliminary.

**Changes in the manuscript:**  
Location: Section IV-B, “Formative Expert Evaluation and Observation of Lesion-Search Behavior” (pp. 7–8). The rationale, intended purpose, and inferential limits of the single-expert evaluation have been added.

### Comment R2-3: Clarify the limitations in haptic and anatomical realism

**Response:**  
We agree. Although the original manuscript mentioned several relevant limitations, it did not distinguish them systematically. In “From Algorithm Validation to System-Prototype Evaluation,” we now organize the limitations into two dimensions: anatomical realism and haptic realism.

Anatomical realism is limited by the homogeneous, nearly incompressible linear-elastic liver model; the absence of a separate capsule layer; the simplification of ligament attachments as three fixed regions; and the use of a resting-state envelope shell to represent the abdominal environment.

Haptic realism is limited by the 1-DOF actuator, which provides only normal contact force. It cannot reproduce tangential force during sliding, the circumferential squeezing sensation that occurs when a fingertip indents tissue, friction, texture, or spatially distributed differences across multiple contact points.

The absence of material nonlinearity and explicit capsule behavior affects both dimensions: it limits the anatomical representation of the tissue and weakens the progressively increasing resistance expected during deeper indentation. These simplifications were adopted to preserve real-time performance and compatibility with consumer-grade hardware. They are now described as open engineering targets rather than as capabilities of the current system.

**Changes in the manuscript:**  
Location: Section VI-A, “From Algorithm Validation to System-Prototype Evaluation” (p. 11). The Discussion now distinguishes anatomical realism from haptic realism and explains how the current material, anatomical, and device simplifications affect each dimension.

### Comment R2-4: Provide a more detailed comparison with existing liver-palpation simulators and training systems

**Response:**  
We expanded “Liver-Related Simulation Systems and Training Tasks” with a detailed comparison of recent systems.

The robotic palpation systems reported by Fan et al. and Lee et al. primarily locate hidden lesions without image guidance by probing stiffness variations or combining contact-force and tissue-deformation information. He et al. developed a physical abdominal trainer containing adjustable-stiffness nodules and force sensors for manual palpation practice. Leong et al. used a surrogate finite-element model to visualize internal stress distributions in abdominal tissue at near-real-time rates for physical-examination training.

These studies and our work all address lesion search through stiffness contrast or mechanical response and are therefore comparable at the task level. Their technical approaches, however, differ substantially. Robotic systems emphasize robot-executed probing and tactile sensing for lesion detection or segmentation. Physical abdominal trainers rely on prefabricated models whose geometry, lesion locations, and material properties are generally difficult to alter dynamically. Surrogate finite-element approaches primarily visualize tissue stress or deformation and do not directly deliver real-time force feedback to the user.

In contrast, our system is an interactive virtual-palpation platform based on GB-cFEM. The user directly explores a deformable virtual liver and perceives normal-force feedback generated by local differences in material stiffness through a wearable 1-DOF haptic device. Lesion location, stiffness parameters, and tissue models can be modified in the virtual environment without rebuilding a physical phantom. Thus, the systems address similar lesion-search problems but differ in the interacting agent, feedback modality, model configurability, and system medium.

**Changes in the manuscript:**  
Location: Section II-B, “Liver-Related Simulation Systems and Training Tasks” (p. 3). This subsection now includes a task-level and technical comparison with recent robotic, physical-phantom, and surrogate finite-element palpation systems.

---

## Response to Reviewer 3

### General Comment R3-0: Clarify the novelty relative to the authors’ previous GB-cFEM and haptic-device studies

**Response:**  
We agree. In the Related Work subsection “Position of This Work,” we added an explicit statement that the GB-cFEM solver and fingertip haptic device were previously validated as separate, standalone contributions. Neither prior study integrated these components with hand tracking in a functioning task-oriented closed loop. The contribution of the present manuscript is the integration of these components with force mapping and a liver-lesion localization task into a real-time interactive system.

**Changes in the manuscript:**  
Location: Section II-D, “Position of This Work” (p. 3). An explicit statement distinguishing the prior component-level work from the present system-level contribution has been added.

### Major Comment R3-M1: A single-expert evaluation cannot support broad claims about training or clinical use

The findings should be limited to early formative feedback.

**Response:**  
We agree and have limited the claims throughout the manuscript. The Abstract now states that the single-participant walkthrough provides diagnostic and exploratory feedback and does not independently support claims of clinical-grade training efficacy, educational impact, or learning outcomes.

The subsection “Formative Expert Evaluation and Observation of Lesion-Search Behavior” now defines the study as a formative expert evaluation intended to identify system feasibility and engineering boundaries rather than to measure training effectiveness, skill acquisition, or learning outcomes.

The Discussion similarly states that the single-expert walkthrough is formative and diagnostic and is intended to expose engineering boundaries rather than establish clinical-grade training efficacy, educational impact, or learning outcomes.

**Changes in the manuscript:**  
Location: Abstract (p. 1); Section IV-B (pp. 7–8); Section VI-A (p. 11); and Section VII (p. 12). These passages now consistently describe the evidence as preliminary formative feedback and avoid broader training or clinical claims.

### Major Comment R3-M2: The study does not measure novice performance, skill improvement, retention, or transfer to clinical tasks

Educational and learning-effect claims should therefore be reduced, or the corresponding experiments should be added.

**Response:**  
We chose to reduce the claims rather than add a new learning-effect experiment because the present development stage is focused on establishing engineering feasibility before investing in a structured multi-participant study. The revised Discussion defines the single-expert walkthrough as formative and diagnostic, with an emphasis on identifying engineering limitations and determining whether the physics-driven feedback is mechanically and clinically plausible.

The observations are now described as preliminary qualitative evidence rather than a formal evaluation of training efficacy, educational impact, or learning outcomes. The discussion of medical education in the Introduction is used only to motivate the system’s potential value. A dedicated novice-and-expert study is now proposed in “Future Research Directions” to evaluate learning-related outcomes.

**Changes in the manuscript:**  
Location: Section I, “Introduction” (pp. 1–2); Sections VI-A and VI-C (p. 11); and Section VII, “Conclusion” (p. 12). Claims concerning education and learning have been reduced, and the required novice-and-expert validation study is identified as future work.

### Major Comment R3-M3: Strengthen the quantitative validation of haptic feedback

Please report actual force accuracy, latency, calibration, actuator limits, force resolution, saturation behavior, and interaction stability.

**Response:**  
We agree. This comment is closely related to Comment R1-2. We added measured and implementation-level information to “1-DOF Haptic Rendering Strategy,” including a measured maximum static tensile force of 1.36 N. For the frequency-response measurement, a DC current bias equal to half the peak-to-peak current was added, and the frequency was swept from 10 to 500 Hz while keeping the peak-to-peak current constant. A resonance was observed near 150 Hz; the peak-to-peak force variation reached 0.434 N at 140 Hz, approximately 3.3 times the 0.132 N measured at 10 Hz. At 500 Hz, the peak-to-peak force variation was 0.123 N, corresponding to 0.93 times the 10-Hz reference value. We also added the measured same-frame software latency.

On the default mesh, the mean latency from hand-input update to force-command issuance was 25.2 ms over 90 frames (39.6 FPS; peak, 30.1 ms). The mean comprised 22.2 ms for GB-cFEM/PBD, 1.4 ms for contact and force mapping, and 0.3 ms for rendering. Commands were sent through a 2,000,000-bps serial connection in less than 1 ms per packet, while the device control loop operated at 1000 Hz.

For interaction stability, we now report a 10-ms low-pass time constant for device forces, a 30-ms smoothing time constant for hand positions, and a PWM slew-rate limiter that suppresses cable-snap artifacts at contact onset.

**Changes in the manuscript:**  
Location: Section III-D, “1-DOF Haptic Rendering Strategy” (p. 4). Quantitative device characterization, the fixed peak-to-peak current and DC current bias used for the 10–500 Hz frequency-response measurement, resonance amplification relative to the 10-Hz reference, the 500-Hz relative output, latency profiling, filtering, smoothing, and slew-rate limiting have been added.

### Major Comment R3-M4: Provide objective results for the lesion-localization task

Examples include localization accuracy, completion time, localization error, number of presses, and a haptics-on/off comparison.

**Response:**  
We agree that these measures are important and now identify their absence as a genuine limitation of the present evaluation. We did not retrospectively fabricate or estimate data that were not recorded. Instead, “Formative Expert Evaluation and Observation of Lesion-Search Behavior” now explains three limitations.

First, the evaluation was not instrumented to record quantitative measures such as localization accuracy, time to localization, number of exploratory presses, hand trajectory, or error distance automatically.

Second, some measures require explicit operational definitions for continuous, bimanual, exploratory palpation. For example, a “press” has no unambiguous beginning and end when the user alternates among sliding, sustained indentation, and brief touches. Similarly, an error-distance measure generally assumes a predefined target point, whereas the lesion in this study is a diffuse region of increased stiffness rather than a point target.

Third, the two trials used different lesion positions in a fixed order without a randomized controlled design. Even if the measures had been recorded, the result would have been a confounded \(n=1\) comparison rather than a reliable quantitative haptics-on/off test.

The current interaction pipeline already provides the per-frame fingertip proxy positions and contact states used for haptic-force computation. In future work, we will define the measures operationally, enable logging, and evaluate them in a structured, randomized, multi-participant study with a controlled haptics-on/off comparison.

**Changes in the manuscript:**  
Location: Section IV-B, “Formative Expert Evaluation and Observation of Lesion-Search Behavior” (pp. 7–8), and Section V-B2, “Behavioral Observation in the Lesion-Localization Task” (p. 10). These passages now report the missing objective measures, explain the measurement and design constraints, and describe the logging and controlled-study plan.

### Major Comment R3-M5: Strengthen the biomechanical basis of the liver model

Please provide literature, phantom data, or clinical calibration supporting the Young’s modulus of normal tissue, the lesion-to-background stiffness ratio, lesion size, and boundary constraints.

**Response:**  
We added two paragraphs to “Liver Model and Lesion Configuration” to provide literature support for these four parameter choices.

For the background Young’s modulus and lesion stiffness ratio, ex vivo liver and liver-phantom studies report background stiffness values ranging from a few to a few tens of kilopascals, while lesions or tumor phantoms can be several times stiffer than the surrounding tissue. The approximately 4:1 lesion-to-background modulus ratio used in our study therefore lies within previously reported stiffness contrasts and was not selected solely to facilitate perception.

The preset lesion size was selected with reference to focal liver lesions that remain identifiable through manual palpation. Retrospective intraoperative studies indicate that additional nodules identified through bimanual palpation are commonly on the order of 1 cm or smaller and are located at or near the liver surface.

The fixed regions correspond to locations at which the liver is constrained by the coronary and triangular ligaments and by the inferior vena cava. Previous studies of liver boundary conditions identify these structures as dominant anatomical constraints. The three fixed regions used in our model—the hepatic hilum, the groove for the inferior vena cava, and the bare area—correspond to these principal attachment locations.

We also clarify that this remains a simplified fixed-boundary approximation rather than a compliant ligament-level model. Because ligament compliance is not modeled quantitatively, the approximation introduces boundary-condition error relative to a complete patient-specific simulation. The description of the abdominal collision shell now also states that the current model does not separately represent the anterior falciform ligament, authentic abdominal contact relationships, or ligament compliance.

**Changes in the manuscript:**  
Location: Section III-E, “Liver Model and Lesion Configuration” (pp. 4–6), including Table 1 (p. 5). The biomechanical rationale and supporting references for tissue stiffness, stiffness contrast, lesion size, and anatomical constraints have been added, together with a clearer account of the remaining boundary-condition limitations.

### Major Comment R3-M6: Provide complete implementation details for the XPBD and VegaFEM benchmarks

Please report solver settings, mesh scale, loading conditions, boundary constraints, time steps, substeps, and convergence criteria.

**Response:**  
We agree. This comment overlaps with Comment R1-4. A new table entitled “Solver Settings Used in the Displacement-Accuracy and Volume-Preservation Comparisons” has been added to “Application-Level Technical Validation Based on a Specific Liver Model.”

The table and accompanying text report the common time step of 0.01 s; material parameters of $E=7000$ Pa and $\nu=0.49$, except where Poisson’s ratio was varied; 30/150 coupling iterations for GB-cFEM; 5/50 substeps for XPBD with `maxIterations = 1`; and 1/2 Newton iterations for VegaFEM.

All three methods used the same TetGen liver tetrahedral mesh, containing 13,800 nodes and 62,897 tetrahedra, and the same fixed-node definitions. We also added the three loading magnitudes and settle/ramp/hold step counts for Experiment 1, as well as the anchored and displaced slice proportions, displacement magnitude, and two Poisson’s-ratio settings for Experiment 2.

The revised manuscript now explains that the values in Experiments 1 and 2 were obtained after fixed-step settle/ramp-or-drag/hold protocols rather than through a common residual-tolerance test. VegaFEM performs Newton iterations within each step, whereas our method and XPBD use fixed outer-iteration or substep counts. This is therefore a mixed protocol, not a comparison in which all three methods share an identical numerical convergence tolerance.

The Results subsection “Real-Time Performance and Scalability” also now reports the mesh-generation procedure for the resolution sweep, thread counts, and the numbers of warm-up and measured frames.

**Changes in the manuscript:**  
Location: Section IV-A (pp. 6–7), including Table 2 (p. 6), and Section V-A3, “Real-Time Performance and Scalability” (pp. 9–10). The solver settings, benchmark protocol, convergence and termination explanation, mesh information, loading and boundary conditions, and performance-measurement procedure have all been added or expanded.

### Major Comment R3-M7: Explain the limitations of the 1-DOF device fully

The device can represent only normal resistance and cannot reproduce tangential force, enveloping or squeezing sensations, or complex tissue-contact feedback.

**Response:**  
We agree. This comment is closely related to Comment R2-3. The Discussion now states explicitly that each fingertip actuator provides only the scalar magnitude of normal contact force. It cannot reproduce tangential forces during sliding or the squeezing sensation around the fingertip that occurs when real tissue envelops the finger. The system also does not reproduce friction, local texture, or spatially varying distributed feedback across multiple contact points.

The expert’s observation in Task C, reported in “Diagnosis of Engineering Constraints and Evaluation of Force-Rendering Quality,” is consistent with this limitation. The expert noted that the 1-DOF output conveys only one-directional normal resistance and lacks tangential force and a more complex reaction-force distribution, preventing reproduction of the enveloping squeeze sensation of real tissue. This observation provides formative support for the Discussion’s analysis of haptic realism.

**Changes in the manuscript:**  
Location: Section V-B3, “Diagnosis of Engineering Constraints and Evaluation of Force-Rendering Quality” (p. 10), and Section VI-A (p. 11). The limitations of 1-DOF feedback have been expanded to cover normal force, tangential force, enveloping squeeze, friction, texture, and distributed contact feedback.

### Major Comment R3-M8: Reduce overly strong conclusions

The study demonstrates technical feasibility and provides early expert feedback; it does not demonstrate that the system is a clinical-grade trainer or improves medical learning outcomes.

**Response:**  
We agree and rewrote the concluding summary. We removed claims that the system already meets the needs of medical education or improves learning efficiency. The revised Conclusion is limited to the findings supported by the present study: the GB-cFEM-based, physics-driven haptic simulation loop can operate in real time on compact, consumer-grade hardware, and a surgical expert was able to extract physically meaningful stiffness cues during a brief walkthrough. These findings support the technical feasibility and system-level connectivity of the architecture.

We further state that the results suggest, without directly demonstrating, that this type of system could eventually address the needs for accessible hardware and information-rich feedback in frequent medical training. Training effectiveness, learning-efficiency gains, and clinical-grade fidelity require the structured, multi-participant novice-and-expert study proposed in “Future Research Directions.” The current single-expert formative evaluation provides only preliminary evidence and engineering insight.

The Abstract and Discussion have been revised in parallel so that the scope of the conclusions is consistent throughout the manuscript.

**Changes in the manuscript:**  
Location: Abstract (p. 1); Section VI-A (p. 11); and Section VII, “Conclusion” (p. 12). Overly strong claims have been removed or qualified in these passages.

### Minor Comment R3-m1: Correct publication-date placeholders and other template or formatting inconsistencies

**Response:**  
The publication date and DOI fields currently retain the default IEEE Access submission-template placeholders because these fields are assigned by the publisher during production. We also checked the remaining template and formatting details together with the corrections made in response to Comment R1-5 and found no unresolved inconsistencies.

**Changes in the manuscript:**  
Location: publisher-assigned metadata on p. 1 and formatting throughout pp. 1–13. Formatting inconsistencies identified during proofreading were corrected; the publication-date and DOI placeholders remain pending production.

### Minor Comment R3-m2: Define and use “lesion,” “tumor,” and “hard region” consistently

**Response:**  
We agree. “Lesion” was already the predominant term in the manuscript, but two instances of “tumor” and five instances of “hard region” remained in a table and two passages. All of these terms referred to the same modeled construct: a local region with an increased Young’s modulus. We replaced the remaining occurrences with “lesion.”

We also added the following explicit definition to “Liver Model and Lesion Configuration”:

> Throughout this paper, “lesion” refers to this locally stiffness-enhanced tissue region: the term is used in a general sense to denote a mechanically distinguishable abnormal region rather than a specific pathological diagnosis such as a tumor.

We did not adopt “tumor” as the uniform term because it denotes a specific neoplastic diagnosis. Applying it to a model defined only by local modulus enhancement would introduce an unsupported clinical interpretation.

**Changes in the manuscript:**  
Location: Section III-E, “Liver Model and Lesion Configuration” (pp. 4–6), including Table 1 (p. 5) and Figure 2 (p. 6). Terminology has been standardized to “lesion,” and an explicit definition has been added.

### Minor Comment R3-m3: Improve the readability of figures and tables

Particular attention should be given to label sizes, captions, and experimental conditions in the performance figure and system screenshots.

**Response:**  
We reviewed all performance plots and system screenshots and made the following changes:

1. The performance comparison previously contained both logarithmic-scale and linear-scale panels. Because the linear-scale panel compressed three of the four curves into a narrow, indistinguishable band and added little information, we removed it and retained the logarithmic-scale plot. Abbreviated experimental conditions in the legend were expanded to full parameter descriptions, and the caption now identifies the configuration associated with each method.
2. Development-interface text and the internal label “Tumor” in Figure 2 were replaced with “Lesion.”
3. Residual debugging instructions visible in the Figure 4 screenshot were blurred.
4. The column padding in Table 2 was reduced so that the solver-settings table fits within the column width.

The axis labels and legend text in the remaining figures are legible in the two-column layout. Their sizing will be checked again at the proof stage if production scaling changes their appearance.

**Changes in the manuscript:**  
Location: Table 2 (p. 6), Figure 2 and its caption (p. 6), Figure 4 and its caption (p. 8), and Figure 7 and its caption (p. 10). The solver-settings table width, system screenshots, performance plot, legend, and captions have been revised for clarity and consistency.

### Minor Comment R3-m4: Interpret the single-participant Likert ratings cautiously

The ratings should be treated as qualitative formative feedback, not as generalizable usability evidence.

**Response:**  
We agree. The paragraph following the Likert-rating table now states that the ratings from a single surgical expert are qualitative, formative feedback used to complement the verbal walkthrough findings. They are not statistically generalizable usability evidence and should not be interpreted as validating training effectiveness. The paragraph also uses the exact Table 8 evaluation-dimension label “system feasibility and potential” for consistency.

**Changes in the manuscript:**  
Location: the paragraph immediately following Table 8 in Section V-B3 (p. 10). The interpretation of the single-participant ratings and the evaluation-dimension terminology have been revised accordingly.

### Minor Comment R3-m5: Explain the supplementary video and clarify whether the lesion was visible during the formal evaluation

**Response:**  
The approximately 54-second supplementary video has no audio and contains three sequential demonstrations:

1. A system overview showing the liver tetrahedral mesh and multiple colored wireframe fingertip proxies.
2. A real-time large-deformation demonstration under deep single-point indentation, with an overlaid deformation/contact heat map illustrating the GB-cFEM core’s real-time behavior during large deformation.
3. A bimanual demonstration in which one hand retracts tissue to improve exposure while the other performs palpation.

In the third segment, the lesion is rendered as a visible highlighted region solely to help viewers understand how retraction exposes the lesion. This visualization is for demonstration purposes only and does not represent the formal evaluation condition. During the formative evaluation, the lesion had no visual cue at any time. The expert inferred its location only from local deformation and haptic differences, consistent with the manuscript’s description of a hidden lesion. The highlighted video demonstration and the visually blinded evaluation therefore serve different purposes and are not contradictory.

**Changes in the manuscript:**  
Location: not applicable; this clarification concerns the supplementary video, which is not cited in the main manuscript. No manuscript change was required.

### Minor Comment R3-m6: Verify all reference information, particularly for the 2025 publications

**Response:**  
We checked and completed the bibliographic information for the three 2025 references:

- Wang et al., *IEEE Access*, vol. 13, pp. 179041–179056, 2025, doi: 10.1109/ACCESS.2025.3616629.
- Xu et al., *IEEE Transactions on Haptics*, vol. 18, no. 3, pp. 626–639, 2025, doi: 10.1109/TOH.2025.3581014.
- Bjelland et al., *IEEE Transactions on Haptics*, vol. 18, no. 3, pp. 569–581, 2025.

**Changes in the manuscript:**  
Location: References (pp. 12–13). The corresponding reference entries have been verified and completed.

### Minor Comment R3-m7: Shorten repetitive material and complete a careful English-language edit

**Response:**  
We edited and condensed the Introduction and Discussion. In the Introduction, repeated statements concerning frequent practice and information-rich feedback previously appeared in several paragraphs and again in the contribution list. These statements have been consolidated into one concise discussion, with brief references thereafter.

In the Discussion, the opening paragraph previously repeated the contribution already explained in “Position of This Work”; it has been replaced with a short transition. The summary of the expert’s three observations previously repeated details already reported in the Results and has been reduced to the implications of those observations for palpation-related judgment. Finally, “Application Focus and Future Evolution of the System” and “Future Research Directions” contained nearly identical lists of engineering improvements. These lists have been consolidated in “Future Research Directions.”

During the final consistency pass, “multi-sample user study” was corrected to “multi-participant user study,” and the Related Work wording was revised so that the study is described as assessing system integration and identifying engineering boundaries rather than validating usability.

**Changes in the manuscript:**  
Location: Section I, “Introduction” (pp. 1–2); Section II-D, “Position of This Work” (p. 3); and Sections VI-A–VI-C, “Discussion” (p. 11). These sections have been shortened, redundant passages have been consolidated, terminology has been corrected, and the manuscript has undergone a complete English-language edit.

---

## Closing Statement

We thank the Editor and Reviewers again for their detailed and constructive comments. The revisions have helped us define the scope of the contribution more precisely, improve the technical transparency and reproducibility of the work, strengthen the biomechanical and methodological context, and present the limitations of the current prototype more clearly. We hope that the revised manuscript satisfactorily addresses all concerns.

Sincerely,

The Authors
