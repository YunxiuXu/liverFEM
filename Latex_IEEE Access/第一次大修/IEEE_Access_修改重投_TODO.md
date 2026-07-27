# IEEE Access 修改重投 TODO

> 稿件号：Access-2026-15540  
> 结论：Reject - updates required before resubmission（编辑明确鼓励修改后重投）  
> 注意：IEEE Access 只允许一次重投机会。

## 编辑部要求

- [ ] 逐条回应所有审稿意见；如果不同意某个技术意见，需在回复信中提供论据，并在论文中作相应说明。
- [ ] 对审稿人建议的文献只在确实相关且能增强论文时引用，不必为了迎合审稿人强制添加。
- [ ] 准备 Response to Reviewers，每条都包含：(a) Reviewer's concern，(b) Authors' response，(c) Action/actual change。
- [ ] 准备所有修改均已高亮的 `Highlighted PDF`，包括语法修改。
- [ ] 准备无高亮的 clean manuscript，同时提交 LaTeX 源文件和内容完全一致的 PDF。
- [ ] 使用编辑部附带的 Resubmission Checklist 完成提交前检查。
- [ ] 如果作者名单或顺序发生变化，提交 `Request for Byline Change`，详细说明变更理由及每位作者的贡献。
- [ ] 在 IEEE Author Portal 中从原稿件选择 `Start Resubmission` 进行重投。


## Reviewer 1
Thank you for your comments, we made the following revisions.
- [x] **R1-1：重新限定论文贡献。** 将本文明确定位为“系统集成与技术可行性研究”，而不是训练有效性证据；单一外科专家只能提供形成性反馈，不能支持临床级训练、教育影响或学习效率的强结论；修改 Abstract、Discussion 和 Conclusion。
  - **审稿人回复：** 我们重新限定了本文的贡献表述。Abstract 明确将本研究定位为“系统集成与技术可行性研究”（原文：“This study is framed as a system-integration and technical-feasibility study rather than as evidence of training effectiveness”），并加入一句说明：单一外科专家的走查式评估只能提供诊断性、探索性反馈，不能单独支撑临床级训练效果、教育影响或学习效率方面的结论（原文：“this single-participant walkthrough provides diagnostic, exploratory feedback and does not by itself support claims of clinical-grade training efficacy, educational impact, or learning outcomes”）。Discussion 的 “From Algorithm Validation to System-Prototype Evaluation” 小节开头新增一句显式声明（原文：“It should be emphasized that this single-expert, walkthrough-based evaluation is formative and diagnostic in nature: it is designed to expose engineering boundaries rather than to establish clinical-grade training efficacy, educational impact, or learning outcomes, and none of the observations discussed below should be read as demonstrating such effects.”），强调该评估是形成性、诊断性的，目的是暴露工程边界而非证明训练成效。Conclusion 总结段删除了原先“matches the actual demands of modern medical education”“meets the educational standards of improving learning efficiency”等过强表述，改为（原文：“Establishing actual training effectiveness, learning-efficiency gains, or clinical-grade fidelity will require the structured, multi-participant, novice-and-expert study outlined in the Future Research Directions subsection.”）
- [x] **R1-2：补充触觉渲染管线的技术细节和定量验证。** 报告设备力范围、力标定方法、饱和、端到端延迟、滤波/平滑方法，以及物理更新与 1000 Hz 触觉循环的同步方式；尽可能提供实测力输出数据，而不只是计算力映射公式。
  - **审稿人回复：** 此条与 R3-M3 要求高度一致，共享同一处修改（详见 R3-M3 的回复）。我们在 “1-DOF Haptic Rendering Strategy” 小节补充了执行器台架实测的可感知力范围与饱和行为（The actuator was bench-characterized with a force sensor, giving a calibrated, perceptible output-force range of 0.04 N to 1.36 N (values above this range saturate at 1.36 N) and a linear vibration-frequency response over a 100--180 Hz resonance band (correlation coefficient 0.9998).）；逐用户力阈值标定方法（“A per-user calibration before each session sets the perceptible-force threshold, and any $F_n$ below this threshold is rendered at the threshold value instead of zero.）；软件管线的实测同帧延迟（The haptic control loop runs at 1000 Hz, with force commands sent over a 2,000,000 bps serial link (under 1 ms per packet). On the default liver mesh (13,800 nodes, 62,897 tetrahedra), profiling the software pipeline over 90 frames gives a mean same-frame latency of 25.2 ms (39.6 FPS, peak 30.1 ms) from hand-input update to force-command issuance, of which 22.2 ms is GB-cFEM/PBD, 1.4 ms is contact and force mapping, and 0.3 ms is rendering.）；以及滤波/平滑处理（Device forces are additionally low-pass filtered (10 ms time constant) and hand positions are smoothed (30 ms time constant) to suppress transients, and the actuator command path applies a PWM slew limiter to avoid cable-snap artifacts on contact onset.”）。
- [x] **R1-3：澄清形成性评估设计。** 说明 haptics-off 和 haptics-on 使用的病灶位置如何选择，两者在尺寸、深度和难度上是否可比；说明固定试次顺序所带来的学习/熟悉效应；如果可能，补充定位准确率、定位时间、按压次数、轨迹、置信度或误差距离等客观指标。
  - **审稿人回复：** 系统预置了 3 个可切换的病灶位置，三者共享相同的局部半径和杨氏模量增强量，因此在尺寸和刚度对比度上按构造严格匹配（原文：“all three share the same local radius and the same Young's-modulus enhancement, so the two trial positions were matched in lesion size and stiffness contrast by construction.”）；但三者位于肝脏模型的不同解剖区域，我们并未独立验证过它们在距表面深度和局部几何复杂度上的可比性，故不能排除残余的难度差异（原文：“because the three presets sit in different anatomical regions of the model, they were not independently verified to be matched in depth beneath the surface or in local geometric complexity, so a residual difference in task difficulty between trials cannot be ruled out.”）。三个预置位置按固定、确定的顺序循环切换，并未在两次试次间随机化或做顺序平衡，因此学习/熟悉效应也无法排除（原文：“The presets are also cycled in a fixed, deterministic order rather than randomized or counterbalanced across trials, so a learning or familiarization effect between the haptics-off and haptics-on trials cannot be excluded either”）；由此观察到的策略变化已改写为“提示性的定性趋势”而非“受控比较”的结论（原文：“the strategy change reported below should therefore be read as a plausible, suggestive observation rather than an isolated causal effect of haptic feedback.”）。关于客观指标：本轮单次、诊断性的形成性走查未做自动化埋点（原文：“the task was not instrumented to automatically record quantitative indicators such as localization accuracy, time-to-localization, number of exploratory presses, hand trajectory, or error distance.”），定位准确率、时间、按压次数、轨迹、误差距离等指标当时未被记录，行为数据只通过观察和访谈定性记录，文中已如实说明这一局限（详见 R3-M4 回复中的具体原因和下一步方案）。
- [x] **R1-4：补齐 XPBD/VegaFEM 对比的可复现信息。** 明确 substeps、iterations、timestep、材料参数、网格规模、载荷条件、边界约束和终止/收敛条件；对不同方法网格规模未完全匹配的情况谨慎表述结论。
  - **审稿人回复：** 此条与 R3-M6 要求完全一致，共享同一处修改（详见 R3-M6 的回复）。我们核对了三种方法在大变形精度对比（Experiment 1）和体积保持对比（Experiment 2）中实际使用的设置，并在 “Application-Level Technical Validation Based on a Specific Liver Model” 小节新增了一段说明和一张标题为 “Solver Settings Used in the Displacement-Accuracy and Volume-Preservation Comparisons” 的表，其中明确列出三种方法共用同一份网格和固定点定义，以及时间步长（0.01 s）、材料参数（$E=7000$ Pa）、GB-cFEM 耦合迭代次数（30/150）、XPBD substeps（5/50，maxIterations=1）、VegaFEM Newton 迭代次数（1/2）。同时补充了加载条件、边界条件，以及终止/收敛条件的说明。Results 部分的 “Real-Time Performance and Scalability” 小节也补充了网格规模扫描方法、线程数设置（1 线程与 10 线程，最多 12 核）、以及各方法的预热/测量帧数。
- [x] **R1-5：完成全文排版和校对。** 检查并修正空格、断字和转换问题。
  - **审稿人回复：** 已全面校对，LaTeX 源码中未发现这类问题。
- [x] **补充：说清本文相对作者既有工作的新贡献。** 明确 GB-cFEM 和指尖触觉设备已分别发表，本文的新贡献是将它们与手部追踪、肝脏病灶定位任务和力映射集成为可运行的实时闭环系统。
  - **审稿人回复：** 此条与 R3-0 要求完全一致，共享同一处改动（详见 R3-0 的回复）。我们在 “Related Work” 的 “Position of This Work” 小节新增一句（原文：“The GB-cFEM solver used here has previously been validated as a standalone real-time large-deformation algorithm {wang2025group}, and the fingertip haptic device used here has previously been validated as a standalone lightweight actuator {xu2025lightweight}; neither prior study integrated the two with hand tracking into a running, task-oriented closed loop, which is what the present work adds.”），直接点名 GB-cFEM 算法和指尖触觉设备各自已作为独立成果发表，本文的增量贡献是把二者与手部追踪一起集成进一个可运行的病灶定位任务闭环。
- [x] **R1-补充：补充相关文献与对比。** 加强医学模拟器验证方法、触觉渲染评估、手术模拟用户研究设计，以及近期肝脏/腹部触诊系统的文献与对比；通用教育文献只用于动机，不用于证明本系统已具有教育效果。
  - **审稿人回复：** 此条与 R2-4 关于肝脏/腹部触诊系统对比的要求重叠，一并回应（详见 R2-4 的回复）。此外，我们在 “Haptic Device Degrees of Freedom and Low-Dimensional Haptic Rendering” 小节补充了两项相关研究：Fazlollahi et al. 为触觉设备的力反馈性能提供了定量评测方法，Hafiz et al. 则为用户辨别本文所涉及的刚度差异提供了心理物理学依据。在 “Formative Expert Evaluation and Observation of Lesion-Search Behavior” 小节中，我们补充了三类方法学依据：Cheng et al. 和 da Silva et al. 说明模拟系统在开展大规模效度研究前应先进行试点测试和迭代改进；Cook et al. 和 Borgersen et al. 强调效度需要围绕具体用途逐步论证，不能仅依据单名专家的评价直接成立；Xu et al. 则表明，腹部触诊模拟任务即使设计合理，也未必能够形成可靠的能力评估指标。这些研究共同支撑了本文先开展单专家形成性评估、暂不作大规模效度声称的设计选择。
  在 Likert 量表结果部分，我们引用 Sullivan et al. 说明有序量表数据的适当解读方式，并明确这些结果不能作为具有统计推广性的可用性证据，也不能据此验证训练效果。
  Introduction 中 Wisniewski et al. 等通用教育研究的作用保持不变，仅用于说明研究动机，不作为本系统已经具备教育效果的证据。

## Reviewer 2
Thank you for your comments, we made the following revisions.
- [x] **R2-1：** 扩展对系统临床相关性和潜在教育影响的讨论，但不将“潜在价值”写成“已证实效果”。
  - **审稿人回复：** 我们重写了 Conclusion 最后一段，将原先对教育效果的直接断言改为更克制的表述。修改后的结论重点说明，该系统“can be engineered to run in real time on compact, consumer-grade hardware”，且“a surgical expert could extract physically meaningful stiffness cues”。现有结果仅“support the technical feasibility and system-level connectivity”，并“suggest, without by themselves demonstrating”其潜在训练价值。对于“actual training effectiveness, learning-efficiency gains, or clinical-grade fidelity”，仍需开展“structured, multi-participant, novice-and-expert study”。Abstract 中也同步加入了类似的克制表述。
- [x] **R2-2：** 更充分地说明为什么采用单一专家形成性评估，并明确其目的是识别工程问题，而不是验证训练效果。
  - **审稿人回复：** 我们在 “Formative Expert Evaluation and Observation of Lesion-Search Behavior” 小节开头补充了评估设计依据。该阶段首先确认“the simplified anatomical constraints, the current material model, and the 1-DOF force channel”能否产生“mechanically and clinically sensible feedback”，以避免过早开展“a large novice-and-expert study”。同时，我们将该评估明确界定为“a formative expert evaluation”，其目标是识别“system feasibility and engineering boundaries”，而不是测量“training effectiveness, skill acquisition, or learning outcomes”；因此，与这些结果相关的结论均应视为“preliminary”。
- [x] **R2-3：** 进一步澄清当前系统在触觉真实性和解剖真实性方面的限制。
  - **审稿人回复：** 此前相关讨论已涉及材料模型、解剖约束和 1-DOF 力反馈的局限，但未对其进行系统区分。为此，我们在 Discussion 的 “From Algorithm Validation to System-Prototype Evaluation” 小节中，将其归纳为两个独立的真实性维度：“anatomical realism”和“haptic realism”。
  
  “Anatomical realism”主要涉及均质、近不可压缩的线弹性肝脏模型、缺少独立包膜层、简化为三个固定区域的韧带连接，以及采用“resting-state envelope shell”表示腹腔环境。“Haptic realism”主要涉及 1-DOF 执行器仅能提供“normal contact force”，无法再现滑动时的“tangential force”、指尖包裹挤压感，以及摩擦、纹理和多接触点差异等“distributed feedback”。
  
  此外，“material nonlinearity and capsule behavior”的缺失会同时影响两个维度：既限制组织模型的解剖真实性，也削弱深压时逐渐增强的刚度反馈。上述简化是为了保证“real-time performance”和“compatibility with consumer-grade hardware”，目前仍属于“open engineering targets”，而不是当前系统已经具备的能力。
- [x] **R2-4：** 增加与现有肝脏触诊模拟器和训练系统的更详细对比。
  - **审稿人回复：** 此条与 R1-补充中关于“近期肝脏/腹部触诊系统对比”的要求一致，因此两条意见共享同一处修改。
  我们在 “Liver-Related Simulation Systems and Training Tasks” 小节中新增了一段对近期相关系统的对比。具体而言，Fan et al. 和 Lee et al. 提出的机器人触诊系统主要通过变刚度探测，或结合接触力与组织形变信息，在无影像引导的情况下定位隐藏病灶。He et al. 开发的实体腹部训练器则在固定的物理模型中嵌入可调刚度结节和力传感器，以支持徒手腹部触诊练习。Leong et al. 采用代理有限元模型，在接近实时的条件下可视化腹部组织内部的应力分布，用于辅助体格检查训练。
  上述研究与本文均涉及通过组织刚度差异或力学响应完成病灶搜索，因此在训练任务层面具有较强的可比性。然而，它们与本文所采用的技术路径存在明显差异。机器人触诊系统侧重由机器人执行探测并获取触觉感知信息，主要服务于病灶检测或分割；实体腹部训练器依赖预先制作的固定物理模型，其几何结构、病灶位置和材料属性通常难以动态调整；代理有限元方法主要提供组织应力或形变的可视化结果，并不直接向使用者输出实时力反馈。相比之下，本文构建的是一个基于 GB-cFEM 的交互式虚拟触诊系统。使用者能够直接探索可变形的虚拟肝脏模型，并通过可穿戴 1-DOF 触觉设备感知由局部材料刚度差异产生的法向力反馈。病灶位置、刚度参数和组织模型可以在虚拟环境中进行调整，而无须重新制作实体模型。因此，本文与上述研究针对的是相近的病灶搜索问题，但在交互主体、反馈形式、模型可配置性以及系统载体方面有所不同。

## Reviewer 3
Thank you for your comments, we made the following revisions.
### 总体意见

- [x] **R3-0：澄清新颖性。** 更明确地说明本文与作者既有 GB-cFEM 和触觉设备工作相比新在哪里。
  - **审稿人回复：** 我们在 “Related Work” 的 “Position of This Work” 小节新增一句（原文：“The GB-cFEM solver used here has previously been validated as a standalone real-time large-deformation algorithm {wang2025group}, and the fingertip haptic device used here has previously been validated as a standalone lightweight actuator {xu2025lightweight}; neither prior study integrated the two with hand tracking into a running, task-oriented closed loop, which is what the present work adds.”），直接点名 GB-cFEM 算法和指尖触觉设备各自已作为独立成果发表，本文的增量贡献是把二者与手部追踪一起集成进一个可运行的病灶定位任务闭环。

### Major Issues

- [x] **R3-M1：** 单一外科专家的评估不足以支持广泛的训练或临床使用声称；将结果限定为早期形成性反馈。
  - **审稿人回复：** 我们已在全文多处对结论范围做了限定（原文分别见 Abstract 的 “this single-participant walkthrough provides diagnostic, exploratory feedback and does not by itself support claims of clinical-grade training efficacy, educational impact, or learning outcomes”；“Formative Expert Evaluation and Observation of Lesion-Search Behavior” 小节的 “This evaluation was a formative expert evaluation. Its goal was to identify system feasibility and engineering boundaries, not to measure training effectiveness, skill acquisition, or learning outcomes”；Discussion 的 “From Algorithm Validation to System-Prototype Evaluation” 小节开头的 “this single-expert, walkthrough-based evaluation is formative and diagnostic in nature: it is designed to expose engineering boundaries rather than to establish clinical-grade training efficacy, educational impact, or learning outcomes.”），明确单一外科专家的走查评估只能提供早期、诊断性的形成性反馈，用于识别系统的工程边界，不能作为支持广泛训练效果或临床使用的证据。
- [x] **R3-M2：** 论文没有测量新手表现、技能提升、保持性或向真实临床任务的迁移；因此降低教育和学习效果声称，或补充相应实验。
  - **审稿人回复：** 我们选择“降低声称”而非“补充实验”，因为当前阶段的核心目标仍是先确认系统的工程可行性，再决定是否值得投入一次结构化的多参与者学习效果研究（详见对 R2-2 的回复）。在此基础上，我们在 Discussion 的 “From Algorithm Validation to System-Prototype Evaluation” 小节中进一步明确，该单专家走查属于“formative and diagnostic”评估，重点在于“identifying engineering limitations”，并判断系统的物理驱动反馈是否“mechanically and clinically plausible”。相关观察仅构成“preliminary qualitative evidence”，而不是对“training efficacy, educational impact, or learning outcomes”的正式评估。Introduction 中关于医学教育需求的讨论仅用于“motivate the potential value of the system”；系统的教育效果将在 Future Research Directions 所述的“dedicated novice-and-expert study”中进一步检验。
- [x] **R3-M3：** 加强触觉反馈的定量验证；补充实际力准确性、延迟、标定、执行器限制、力分辨率、饱和和交互稳定性。
  - **审稿人回复：** 此条与 R1-2 要求高度一致，共享同一处改动，具体的英文原文引用详见 R1-2 的回复：可感知力范围 0.04 N–1.36 N 及饱和行为、逐用户力阈值标定、振动频率响应线性度（100–180 Hz，相关系数 0.9998）、实测同帧延迟 25.2 ms（39.6 FPS，物理求解 22.2 ms + 接触/力映射 1.4 ms + 渲染 0.3 ms，串口 <1 ms），以及设备力/手部位姿滤波与 PWM 斜率限制。
- [x] **R3-M4：** 围绕核心的病灶定位任务提供客观结果，例如定位准确率、时间、定位误差、按压次数，以及 haptics-on/off 对比。
  - **审稿人回复：** 此条与 R1-3 后半部分要求一致，一并回应。我们没有为了回应此条意见而事后编造或估算相关数据，而是在 “Formative Expert Evaluation and Observation of Lesion-Search Behavior” 小节中明确说明了当前评估的三项限制。
  首先，本轮实验“was not instrumented to automatically record quantitative indicators”，因此未记录“localization accuracy, time-to-localization, number of exploratory presses, hand trajectory, or error distance”等指标。
  其次，部分指标在连续、双手和探索式触诊过程中缺乏明确的操作性定义。例如，当专家在滑动、持续按压和短暂轻触之间切换时，一次“press”没有清晰的起点和终点；“error distance”通常需要预先定义目标点，而本文中的病灶是“a diffuse region of increased stiffness rather than a point target”。
  再次，由于两次试次的病灶位置和实验顺序不同，且未采用随机对照设计，即使记录了这些数据，也只能形成“a confounded (n=1) comparison”，不能据此进行可靠的定量比较。
  我们将此问题明确视为“a genuine limitation of the present evaluation”。现有交互管线已经能够记录用于触觉力计算的“per-frame fingertip proxy positions and contact states”。下一步将对上述指标给出明确的操作性定义并启用日志记录，并在“a structured, randomized, multi-participant study”中设置受控的“haptics-on/off comparison”进行评估。
- [x] **R3-M5：** 加强肝脏模型的生物力学依据；为正常组织 Young's modulus、病灶刚度比、病灶尺寸以及边界约束提供文献、仿体数据或临床标定依据。
  - **审稿人回复：** 我们在 “Liver Model and Lesion Configuration” 小节新增了两段说明，为参数表中的四项设置补充文献依据。
  首先，关于正常组织 Young’s modulus 和病灶刚度比，已有离体肝脏及肝脏仿体研究报告背景组织刚度处于“a few to a few tens of kPa”的量级，病灶或肿瘤仿体则具有“several-fold higher stiffness than the surrounding tissue”。因此，本文采用的约“4:1 lesion-to-background modulus ratio”处于已有研究报告的刚度对比范围内，而非仅为便于感知而设定。
  其次，关于病灶尺寸，本文的预设尺寸参考了仍可通过手工触诊识别的局灶性肝脏病变范围。相关术中回顾性研究表明，经“bimanual palpation”进一步发现的结节多数为“on the order of 1 cm or smaller”，并位于“at or near the liver surface”。
  关于边界约束，本文设置的固定区域对应肝脏受到“the coronary and triangular ligaments”以及“the inferior vena cava”固定的位置。这些结构被既有肝脏边界条件研究认为是限制肝脏运动的“dominant anatomical constraints”。本文的三处固定区域，即肝门、下腔静脉沟和裸区，与这些主要解剖附着位置相对应。
  同时，我们明确说明，当前处理仍属于“a simplified fixed-boundary approximation”，而不是“a compliant, ligament-level model”。由于尚未对韧带柔性进行定量建模，该设置相对于完整的患者特异性仿真会产生“boundary-condition error”。相应局限性也已补充至腹腔碰撞外壳的描述中：当前模型尚未分别表示“the anterior falciform ligament”或“authentic abdominal contact relations”，也未像患者特异性边界条件模型那样表示“ligament compliance”。
- [x] **R3-M6：** 对 XPBD 和 VegaFEM benchmark 提供更完整的实现信息，包括 solver settings、mesh scale、loading conditions、boundary constraints、timesteps、substeps 和 convergence criteria，使结果可复现。
  - **审稿人回复：** 此条与 R1-4 要求完全一致，共享同一处修改，具体的英文原文引用详见 R1-4 的回复。“Application-Level Technical Validation Based on a Specific Liver Model” 小节新增的标题为 “Solver Settings Used in the Displacement-Accuracy and Volume-Preservation Comparisons” 的表列出了 timestep（0.01 s）、材料参数（$E=7000$ Pa, $\nu=0.49$）及各方法的 solver-level 迭代/子步设置（GB-cFEM 耦合迭代 30/150、XPBD substeps 5/50、VegaFEM Newton 迭代 1/2）。正文补充了 mesh scale（三方法共用同一份 TetGen 生成的肝脏四面体网格，13,800 个节点、62,897 个四面体）、loading conditions（Experiment 1 的三档拉力强度、settle/ramp/hold 步数）和 boundary constraints（Experiment 2 的锚定/拖拽切片比例、拖拽位移量、两组泊松比设置）。我们还新增一段说明 convergence/终止判据：Experiment 1、2 均按固定步数的 settle/ramp(drag)/hold 协议读取 hold 阶段稳态值，而非基于残差容差的收敛检查；VegaFEM 的 Newton 迭代逐步内收敛，本方法与 XPBD 用固定外层迭代/子步数，这是混合协议，而非三种方法共享同一数值收敛容差。Results 部分 “Real-Time Performance and Scalability” 小节也补充了网格规模生成方式、线程数和预热/测量帧数设置，使对比具备可复现所需的细节。
- [x] **R3-M7：** 充分说明 1-DOF 设备的限制：只能表达法向阻力，不能表达切向力、包裹/挤压感和复杂组织接触反馈。
  - **审稿人回复：** 此条与 R2-3 涉及的“触觉真实性”维度高度相关，共享同一处改动（详见 R2-3 的回复）。我们扩写了 Discussion 的 “From Algorithm Validation to System-Prototype Evaluation” 小节，使 1-DOF 触觉设备的限制更加明确。当前执行器在每个指尖只能提供“normal contact force”的标量大小，无法再现滑动过程中的“tangential force”，也无法呈现真实指尖压入组织时的“squeezing sensation around the fingertip”。此外，系统尚不能输出摩擦、局部纹理，以及多个接触点之间存在空间差异的“distributed feedback”。
  Results 的 “Diagnosis of Engineering Constraints and Evaluation of Force-Rendering Quality” 小节中，专家在 Task C 对同一硬件瓶颈的观察保持不变。专家指出，当前 1-DOF 输出只能传递“one-directional normal resistance”，缺少“tangential force and more complex reaction-force distribution”，因此无法再现真实手指被组织包围时的“enveloping squeeze sensation”。该观察为 Discussion 中对触觉真实性限制的归纳提供了实证支持。
- [x] **R3-M8：** 降低过强结论；当前研究只能证明技术可行性和早期专家反馈，不能证明系统已是临床级训练器或能够改善医学学习成效。
  - **审稿人回复：** 我们重写了 Conclusion 的总结段，删除了原先关于系统已满足医学教育需求或能够提升学习效率的过强表述。修改后的结论仅保留当前研究已经支持的内容：基于 GB-cFEM 的“physics-driven”触觉仿真闭环能够“run in real time on compact, consumer-grade hardware”，并且外科专家在简短走查中能够从系统中提取“physically meaningful stiffness cues”。这些结果支持所提出架构的“technical feasibility and system-level connectivity”。
  同时，我们明确指出，现有结果仅“suggest, without directly demonstrating”该类系统未来可能满足高频医学训练对“hardware accessibility”和“high-information feedback”的需求。对于“training effectiveness, learning-efficiency gains, and clinical-grade fidelity”，仍需通过 Future Research Directions 中提出的“structured, multi-participant study involving both novices and experts”进行验证。当前单专家形成性评估仅提供“preliminary evidence and useful insights”，仍需要进一步验证。
  Abstract 与 Discussion 中涉及教育效果和训练价值的相应表述也进行了同步弱化，以保持全文结论范围的一致性。

### Minor Issues

- [x] **R3-m1：** 修正出版日期等占位信息及其他模板/格式不一致问题。
  - **审稿人回复：** 稿件中的出版日期和 DOI 目前仍使用 IEEE Access 投稿模板中的默认占位符，正式出版前将由编辑部统一更新。其他格式问题已与 R1-5 一并核对，未发现残留问题。
- [x] **R3-m2：** 定义并统一使用 `lesion`、`tumor` 和 `hard region` 等术语。
  - **审稿人回复：** 全文以 `lesion` 为主导术语（约 50 处），但表格和两处正文段落中残留了 `tumor`（2 处）和 `hard region`（5 处），指的其实是同一个构造：局部杨氏模量增强区域。我们已将这些残留统一改为 `lesion`，并在 “Liver Model and Lesion Configuration” 小节新增一句显式定义（原文：“Throughout this paper, ``lesion'' refers to this locally stiffness-enhanced tissue region: the term is used in a general sense to denote a mechanically distinguishable abnormal region rather than a specific pathological diagnosis such as a tumor.”）：`lesion` 在本文中泛指力学上可区分的局部异常组织区域，不预设具体病理诊断。我们没有改用 `tumor` 作为统一术语，因为该词特指肿瘤性病变，会给当前仅做了局部模量增强、尚未验证病理类型的模型引入缺乏证据支撑的临床断言，与本轮大修“避免过强声称”的方向不一致。
- [x] **R3-m3：** 改善图和表的可读性，特别是性能图和系统截图的标签大小、caption 及实验条件说明。
  - **审稿人回复：** 我们逐一核对了性能图和系统截图并做了修改：(1) 性能对比图（Figure 7）此前包含 log-scale 和 linear-scale 两个子图，但 linear-scale 子图会把除 XPBD Fast 外的三条曲线压缩到难以区分的窄带内，未提供额外信息，我们移除该子图只保留 log-scale 图，并将图例中的实验条件从缩写展开为完整参数，caption 中补充说明各方法对应的具体配置。(2)Figure 2中此前包含开发调试用的界面文字和内部变量名 `Tumor` 标注，已替换为 `Lesion`；Figure 4屏幕上残留的调试指令列表做了模糊处理。其余图表的坐标轴、图例字号在双栏排版下清晰可读，未做进一步放大；如排版阶段仍偏小，可在定稿前根据校样调整。
- [x] **R3-m4：** 谨慎解释来自单一参与者的 Likert 评分，只将其作为定性形成性反馈，不将其解释为可推广的可用性证据。
  - **审稿人回复：** 我们在 Table 8后的段落中补充说明：这些单一外科专家评分仅作为“qualitative, formative feedback”，用于印证“verbal walkthrough findings”；它们并非“statistically generalizable usability evidence”，也不应被视为“validating training effectiveness”。
- [x] **R3-m5：** 说明补充视频展示的内容，并说清病灶是只在演示中可见，还是在正式评估中也可见。
  - **审稿人回复：** 我们在回复信中说明如下，未对论文正文做改动（正文未直接引用该补充视频）。补充视频约 54 秒、无音轨，依次展示：(1) 系统总览，肝脏四面体网格与多根彩色线框指尖代理体的整体布局；(2) 单点深压下的实时大变形演示，表面叠加变形/接触热力图，展示 GB-cFEM 核心在大变形下的实时求解能力；(3) 双手操作演示，一侧手牵拉暴露组织，另一侧手触诊，展示左手暴露、右手触诊的分工。第三段演示中为便于观众看清"牵拉后病灶被暴露"这一交互逻辑，病灶被渲染为肉眼可见的高亮区域，这一可视化处理仅用于演示目的，不代表正式评估中的实际条件——在正式的形成性评估任务中，病灶始终没有任何视觉提示，专家只能依靠局部形变和触觉差异推断位置，与正文对病灶隐蔽性的描述一致；视频高亮与评估中的盲触条件服务于不同目的，二者并不矛盾。
- [x] **R3-m6：** 核对所有参考文献信息，特别是 2025 年文献的作者、标题、卷期、页码/文章号和 DOI。
  - **审稿人回复：** 我们已核对并补全 References 中三篇 2025 年文献的书目信息：`wang2025group`（IEEE Access, vol. 13, pp. 179041–179056, DOI: 10.1109/ACCESS.2025.3616629）、`xu2025lightweight`（IEEE Transactions on Haptics, vol. 18, no. 3, pp. 626–639, DOI: 10.1109/TOH.2025.3581014），以及 `bjelland2025haptic`（IEEE Transactions on Haptics, vol. 18, no. 3, pp. 569–581）。
- [x] **R3-m7：** 压缩 Introduction 和 Discussion 中冗长、重复的内容，完成一轮仔细的英语语言编辑。
  - **审稿人回复：** 我们对 Introduction 和 Discussion 做了一轮压缩和语言编辑。Introduction 中，"高频练习""高信息反馈"等表述原本在多个段落及三条 contribution 列表中反复出现三到四次，现合并为一段更紧凑的论述，只在首次出现时完整表述，后文改为简短回指。Discussion 中，开头一段原样重复了 “Position of This Work” 小节已说明过的贡献表述，现改为一句回指性过渡句；总结专家发现的三点问题时，原文重复了 Results 部分已写过的观察细节，现精简为只保留这些观察对触诊判断逻辑的因果影响“Application Focus and Future Evolution of the System” 与 “Future Research Directions” 两小节原各自列出几乎相同的工程改进清单，现合并为一份，只保留在 “Future Research Directions” 中。

## 最后提交前

- [ ] 把审稿人回复caption 改成实际的表格number
- [x] 不用每条都感谢，太乱了
- [x] 看看有没有问号
- [ ] 按 Reviewer 1、2、3 分组写完 point-by-point response，确保上述每个 checkbox 都有对应回复和修改位置。
- [ ] 逐页检查 clean PDF 和 highlighted PDF，确认没有断字、重叠、缺图、引用错误或未高亮的修改。
- [ ] 确认 LaTeX、clean PDF、highlighted PDF、Response to Reviewers 和补充视频内容一致。
- [ ] 由所有作者审阅并同意最终重投版本。

