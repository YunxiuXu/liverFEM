# IEEE Access 第二次修订：审稿意见 TODO

稿件：Access-2026-43742。这里只记录本轮审稿人要求处理的事项。Reviewer 1 明确说不要求新增参与者实验。

## Reviewer 1

- [x] **R1-1：解释两个更新速率。** 说明完整软件流程约 39.6 FPS 时，两次物理更新之间，1000 Hz 设备控制环使用的是上一条指令、插值结果，还是其他值；写清实际力反馈可更新的频率。此项与 R2-1 重复。**已定稿并写入 Main.tex Section III-D。**
  - 论文正文（已写入，英文）：Between two consecutive physics updates (approximately 25 ms apart, at ~39.6 FPS), the 1 kHz control thread holds the most recently issued force command (zero-order hold) and applies this slew-rate limiter before writing it to the actuator, without interpolating or predicting intermediate values. This control loop runs independently of the simulation thread, so frame-to-frame variation in physics compute time does not propagate directly to the actuator: the command stream remains slew-limited and evenly timed regardless of simulation frame-time jitter. The 1 kHz rate therefore governs the low-level actuator control loop, namely the slew-limited PWM output and command timing, while the rate at which genuinely new force information becomes available remains bounded by the ~39.6 Hz simulation update rate, further shaped by the 10 ms low-pass filter noted above.
  - 中文对照：在两次相邻的物理更新之间（约每25毫秒一次，对应约39.6 FPS），1kHz控制线程持续输出最近一次收到的力指令（零阶保持），并施加斜率限制后再写入执行器，不做插值也不做预测。这个控制环路和仿真线程相互独立运行，物理仿真每帧耗时出现波动时，执行器收到的指令依然是经过斜率限制、节奏稳定的输出，不会把仿真帧率的抖动直接传递到电机端。1kHz对应的是这个底层执行器控制环路的节奏，也就是斜率限制后的PWM输出和指令发送节奏；真正新力信息的更新速率，仍然受限于约39.6Hz的仿真更新率，并进一步经过前文提到的10毫秒低通滤波整形。
  - 注：不提及振动通道。本系统在触诊任务中只输出连续法向力，未使用振动反馈（见 Section V 限制讨论），此前草稿中"短时振动由触觉线程生成"的说法与论文正文矛盾，已作废。
  - 回复信草稿（中文，稍后译为英文）：感谢指出这两个更新率容易混淆。我们已在正文中说明：仿真每约25毫秒（39.6 FPS）产生一次新的接触力，1kHz的控制线程在此期间持续输出该力值（零阶保持，不插值不预测），并施加斜率限制后写入执行器；该环路与仿真线程相互独立，因此执行器输出节奏不受仿真帧率波动影响。因此，39.6 FPS是新力信息的实际更新率，1kHz是执行器控制环路（斜率限制/PWM/指令发送）的节奏，二者含义不同。

- [x] **R1-1：纠正延迟表述。** 核对 25.2 ms 的实际计时范围；区分软件整帧耗时与手部输入到实测拉力出现的端到端延迟。**已定稿并写入 Main.tex Section III-D。**
  - 论文正文（已写入，英文）：profiling the software pipeline over 90 frames, with the tracking camera and serial link disabled and a synthetic fingertip trajectory substituted for live input, gives a mean software frame time of 25.2 ms (39.6 FPS, peak 30.1 ms) from hand-input update to force-command issuance, of which 22.2 ms is GB-cFEM/PBD, 1.4 ms is contact and force mapping, and 0.3 ms is rendering. The actuator's mechanical response was characterized separately: recording the actuator with a 1000-fps high-speed camera across five trials, timing from the onset of the serial-command indicator to the first visible onset of motor rotation, gave a mean of 14.0 ms (SD 1.0 ms, $n=5$). This measurement captures only the onset of mechanical motion; the time for the commanded force to fully develop, and for a user to perceive it, was not separately measured in this study.
  - 中文对照：软件流水线的90帧profiling（关闭追踪摄像头和串口，用合成指尖轨迹代替真实输入）测得平均软件帧耗时25.2毫秒（39.6 FPS，峰值30.1毫秒），从手部姿态输入到发出力指令，其中22.2毫秒是GB-cFEM/PBD求解，1.4毫秒是接触与力映射，0.3毫秒是渲染。执行器的机械响应单独测量：用1000fps高速相机对执行器测了5次（原始数据13、15、15、14、13毫秒，只记内部，不写入论文），记录从串口指令指示灯亮起到电机第一次出现可见转动之间的时间间隔，平均14.0毫秒（标准差1.0毫秒，n=5）。这个数值只反映机械转动开始的时刻，力完全建立以及用户实际感知到力变化所需的时间，本研究没有单独测量。
  - 回复信草稿（中文，待译英文）：感谢指出25.2毫秒的表述容易被误解为端到端延迟。我们已在正文中将其明确改称为"软件帧耗时"，并说明测量条件（关闭追踪与串口，使用合成指尖轨迹）。我们另外用1000fps高速相机对执行器做了5次独立测量，记录从串口指令指示灯亮起到电机可见转动开始的时间间隔，平均14.0毫秒（标准差1.0毫秒，范围13-15毫秒）。需要说明的是，这个硬件响应数值只覆盖机械转动开始的时刻，不包括力完全建立、以及用户实际感知到力变化所需的时间，这部分端到端数值本研究未测量。补充说明一点，该执行器电机的额定转速为6400 RPM，对应满速下单圈转动周期约9.4毫秒；这个数字说明电机本身转动很快，转动开始后的力建立过程预期是毫秒量级、而非几十毫秒量级，但由于电机从静止加速到额定转速需要时间，且绳索张力完全建立所需的具体转动角度尚未标定，我们无法仅凭额定转速严格推算出精确的力建立时间，因此仍如实说明这部分未单独测量。
  - 回复信草稿（英文，待定稿）：As additional context, the actuator motor's rated speed is 6400 RPM, corresponding to a period of about 9.4 ms per revolution at full speed. This indicates that once rotating, the motor itself turns fast enough that force buildup is expected to be on the order of milliseconds rather than tens of milliseconds; however, because the motor requires time to accelerate from standstill to rated speed, and the rotation angle needed for the string tension to fully develop has not been separately calibrated, we cannot rigorously derive a precise force-buildup time from the rated speed alone, and the manuscript therefore continues to state this as unmeasured.

- [x] **R1-1：补充力传感器表征。** 解释指令与实测力的关系、饱和如何处理，以及是否检查过重复性或滞后；如现有数据足够，加入简短校准图或表。此项与 R2-2 重复。**已定稿并写入 Main.tex Section III-D（Fig. current_force 段落之后）。**
  - 论文正文（已写入，英文，精简版）：The upper end of the commanded PWM range (180, on a 0--180 scale) coincides with the current at which the bench-measured tension plateaus (about 0.78 A, about 1.36 N). Simulation-side force is clamped to a configured range, soft-clipped near this bound, and mapped through a gamma curve ($\gamma=0.7$) onto the 181 available PWM levels. The reported current-force curve (Fig.~\ref{fig:current_force}) is a single sweep; repeatability and hysteresis were not tested. To characterize the force range used in practice, commanded output was logged as a percentage of the configured force range during an 80-second bench session in which the authors pressed both normal-tissue and lesion regions under the default configuration; outputs spanned from near zero to nearly the full configured maximum (median 60%). This session reflects the authors' own typical operation, not the reported expert evaluation, for which force logging was not enabled.
  - 中文对照：指令PWM范围的上限（180，取值范围0–180）对应台架实测拉力进入平台的电流点（约0.78A，约1.36N）。仿真侧的力值先限幅到设定范围，在接近上限处做软限幅，再通过gamma曲线（γ=0.7）映射到181级可用PWM上。报告的电流-力曲线（Fig. current_force）是单次扫描；重复性和滞后没有测试。为了刻画实际使用中的力范围，我们在一次80秒的台架测试中记录了指令输出占设定力范围的百分比，测试中作者按压了正常组织和病灶区域，输出从接近0到接近设定上限（中位数60%）。这段记录反映的是作者自己的典型操作，不是论文报告的专家评估，专家那次没有启用力值记录。
  - 回复信草稿（中文，待译英文）：感谢提出这几点。我们已补充说明：PWM指令上限与执行器实测饱和点（约0.78A/1.36N）重合；仿真侧力值经过限幅、软限幅和γ=0.7的映射曲线后转换为181级PWM；电流-力关系的重复性和滞后未测试，现有曲线是单次扫描；同时新增了一段80秒台架测试中记录的实际力输出范围（按设定范围百分比表示，从接近0到接近满量程，中位数60%），并注明这是作者自己操作的典型场景，不是专家评估那次的数据（专家那次未启用力值记录）。

- [ ] **R1-2：明确求解器设置。** Table 2 的 30/150 次耦合迭代分别用于哪个实验；说明 Young's modulus、Poisson's ratio 与 XPBD compliance 参数之间的关系。

- [x] **R1-2：定义图 5 横轴。** 解释 `Pulling Force (Accel)` 和 16、30、40 的物理量及单位；必要时同步修改轴名与正文。**已定稿，横轴改为 "Prescribed Loading Acceleration"，图已用 `out/experiment1/analyze_exp1.py` 重新生成（未重跑实验，只改标签），正文第212行同步改写。**
  - 论文正文（已写入，英文）：For the large-deformation accuracy comparison (Experiment 1), a ramped per-vertex acceleration was applied to a deterministic frontal target region, with magnitude $a=16$, $30$, or $40$ in simulation-coordinate units per second squared, scaled by a spatial falloff weight.
  - 中文对照：大变形精度对比（Experiment 1）里，对目标区域施加的是逐顶点的加速度（按空间衰减权重分配），幅值 $a$ 取16、30或40（仿真坐标单位/秒²），不是力。
  - 回复信草稿（中文，待译英文）：感谢指出这个标注不清楚。原图横轴混用了 force 和 accel 两个说法；代码里这个量确实是加速度（`pullAccel`），按每个顶点的空间衰减权重施加，不是统一的总拉力。我们已把横轴改为"Prescribed Loading Acceleration"，正文同步说明其单位为仿真坐标单位/秒²，不对应米或牛顿等真实物理单位。16/30/40 三个数值本身没有改变。

- [x] **R1-2：限定比较结论。** 说明不同求解器没有共用同一收敛容差，结果属于给定实验条件下的应用级对比。**论文正文不用改，已有的表述够了（第175行小节标题 "Application-Level Technical Validation..."，第216行 "not a common convergence tolerance across methods"）。**
  - 回复信草稿（中文，待译英文）：感谢提醒。我们已在方法与结果部分说明比较是应用级、给定实验条件下的结果，而非跨方法的统一收敛标准：该小节标题即为"Application-Level Technical Validation Based on a Specific Liver Model"，正文也明确说明本方法和XPBD使用固定步数终止，VegaFEM的隐式Newton求解器则在每步内部收敛，三者不共享统一的收敛容差，报告的对比是固定步数方案下瞬态之后的稳态值，不是跨方法统一数值收敛标准下的结果。
- [x] **R1-3：限定运行时间数据。** 如有更长日志，报告中位数、高分位及帧时间波动；否则明确 90 帧是短时性能采样。**已重新测量并写入 Main.tex Section III-D。改了 `PipelineProfiler`（`TetgenFEM/main.cpp`）加上真实的中位数/p95/p99/标准差统计（原来只有mean和一个被误标成"p99"的max），跑了6000帧+120帧热身，测量条件跟原来一致（关闭追踪和串口，合成手指输入）。**
  - 论文正文（已写入，英文）：profiling the software pipeline over 6,000 frames (following a 120-frame warmup)... gives a mean software frame time of 24.3 ms (median 24.3 ms, SD 1.2 ms, 95th/99th percentile 26.3/27.2 ms, max 43.1 ms, 41.2 FPS on the mean), of which 22.3 ms is GB-cFEM/PBD, 1.2 ms is contact and force mapping, and 0.2 ms is rendering.
  - 中文对照：软件流水线6000帧profiling（120帧热身），测量条件不变，平均软件帧耗时24.3毫秒（中位数24.3毫秒，标准差1.2毫秒，95/99百分位26.3/27.2毫秒，最大值43.1毫秒，平均41.2 FPS），其中22.3毫秒是GB-cFEM/PBD求解，1.2毫秒是接触与力映射，0.2毫秒是渲染。旧的90帧/25.2ms数据已被这组更长、更完整的数据替换（联动改了1kHz那段里引用的"~39.6 FPS"为"~41.2 FPS"）。
  - 回复信草稿（中文，待译英文）：我们已将90帧的短采样替换为6000帧（120帧热身）的完整profiling，报告中位数、标准差、95/99百分位和最大值。均值和中位数高度一致（24.3ms），但更长采样捕捉到了短采样漏掉的慢帧（最大值从30.1ms升至43.1ms），印证了审稿人的提醒。
- [x] **R1-3：限定功耗结论。** 12 W 空闲、52 W 运行只说明测试 Mac mini 的整机测量结果，不据此声称普遍低计算负担或高能效。此项与 R2 的功耗意见重复。**已写入 Main.tex 第218、285行，第440行原本表述温和不用改。**
  - 论文正文（已写入，英文）：This whole-machine measurement provides an additional dimension for hardware feasibility assessment, supporting feasibility on this tested consumer-grade device. / ...this whole-machine measurement supports feasibility on the tested device, with only several tens of watts of additional whole-machine power during interactive operation.
  - 回复信草稿（中文，待译英文）：感谢提醒。12W到52W只是这台Mac mini的整机功耗测量，不是算法计算负担的直接度量，也没有跨硬件对比。我们已删除"strong evidence of low computational burden"等断言，改为只支持这台测试设备上的可行性，不再声称普遍的低计算负担或高能效。
- [x] **R1-3：保持专家观察为定性结果。** 将 `extracted physically meaningful stiffness cues` 等可能暗示客观定位成功的表述改为专家报告/观察到的感受。此项与 R2-5 重复。**已写入 Main.tex 第442行（结论段）。**
  - 论文正文（已写入，英文）：...and that a surgical expert reported perceiving physically meaningful stiffness cues during a brief walkthrough.
  - 回复信草稿（中文，待译英文）：感谢提醒。我们已把结论段"专家提取出了有生理意义的刚度线索"改为"专家报告感知到了有生理意义的刚度线索"，避免暗示这是客观验证过的结果。
- [x] **R1-minor：核对视频。** 旧回复信写约 54 秒，审稿人看到约 76.5 秒；核对实际视频并改正时长。将视频里的 `tumor`、`target mass` 标签统一为 `lesion`。**用户确认视频实际76秒，标签已改完。论文正文没提视频时长，不用改Main.tex。**
  - 回复信草稿（中文，待译英文）：感谢指出。核对后视频实际时长约76秒，旧回复信的54秒是笔误，已更正。视频中的`tumor`、`target mass`标签已统一改为`lesion`。
- [x] **R1-minor：修改英文术语。** 将 `system connectivity`、`soft-tissue mechanical expression` 等不自然表达改为审稿人建议的更准确说法，并检查全篇一致性。**已改完，全篇5处connectivity、2处mechanical expression（含Table 8行标签）统一替换。**
  - 论文正文：`system connectivity` → `closed-loop system integration`；`soft-tissue mechanical expression` → `representation of soft-tissue mechanics`。
  - 回复信草稿（中文，待译英文）：感谢建议。已将全文的`system connectivity`统一改为`closed-loop system integration`，`soft-tissue mechanical expression`改为`representation of soft-tissue mechanics`，包括Table 8的对应行标签。

## Reviewer 2

- [x] **R2-1、R2-2：完成 R1-1 的共通问题。** 清楚说明物理更新之间的力值、有效反馈带宽，以及力指令到实测力的校准、线性、分辨率、限幅和触诊任务实际使用的力范围。对没有测过的项目如实说明。**随 R1-1a/b/c 一并写入 Main.tex，内容同上，分辨率(181级/0-180)已写入。**
- [x] **R2-3：处理缺失的各向异性结果。** 方法中写了 `Anisotropic response validation`，结果中未报告；核对现有 Experiment 3 数据，能支撑则补结果，不能则删去该验证项目。**已核实数据不可靠（10次跑里isotropic对照组X/Y比值从0.35到4.2都有，本该接近1），已删除 Main.tex 第180行该条验证项目，不留悬空引用。**
  - 回复信草稿（中文，待译英文）：感谢指出这个遗漏。我们核查了Experiment 3的数据，发现isotropic对照组本身在多次重复中未能稳定表现出各向同性（X/Y方向刚度比值波动很大），这说明实验设置存在未分离的方向性混杂因素，数据不足以支撑各向异性验证的结论。我们已从方法部分的验证协议中删除该项，不再声称做过此项验证。GB-cFEM框架对各向异性材料的一般支持能力在既有工作中已经确立，本次删除的只是针对本肝脏模型的这一项具体验证声明。
- [ ] **R2-4：把精度和速度对上配置。** 明确 Table 5 中每个位移误差、运行时间对应的耦合迭代设置；若 17.9% 与 12.0 ms 来自不同设置，不把它们写成同一个精度—速度结果。
- [ ] **R2-5：维持定位结论为定性。** 全文检查单专家病灶定位的说法；若当时确实记录了定位成功/失败，可核对记录后如实报告。
- [ ] **R2-补充：收敛功耗措辞。** 删除将 52 W 对比 12 W 称为低计算负担 `strong evidence` 的说法；说明现有整机功率测量支持的范围。
- [ ] **R2-补充：解释 `GB-cFEM/PBD`。** 明确延迟拆分中这一项包含哪些计算，以及为何出现 PBD。
- [ ] **R2-补充：修正回复信稿号，并完成最终校对。** 旧回复信出现 Access-2026-15540；新回复应准确标明本轮稿件号 Access-2026-43742。通读论文和回复，核对图表、术语、数字与表述一致性。

## Reviewer 3

Reviewer 3 认为当前技术集成和初步评估基本成立，没有提出本轮必须完成的新实验。他提到的多参与者研究以及与 SOFA/Projective Dynamics 的进一步对比，是后续研究建议。

## 杂项修正（不对应具体审稿意见）

- [x] **默认网格节点/四面体数写错。** 原文两处写13,800/62,897，实际代码生成的是4,403/20,061，已改正（Main.tex 第115、138行）。
  - 回复信草稿（中文，待译英文）：核对本轮修改时发现，正文两处默认肝脏网格的节点/四面体数误标为13,800/62,897，已更正为实际生成的4,403/20,061。这是文字性更正，不影响论文中报告的任何实验结果。
