# 评测布局维护

使用现有集合、下载与恢复方法见 [Eval_Layout/README.md](../Eval_Layout/README.md)。本页说明如何维护采样和验证工具。固定的集合 `1000` 已发布为 `eval-layout-1000-v1`；生成新数据时使用新的集合目录和版本，保留 v1 文件及其校验和。

## 实现与目录

| 位置 | 职责 |
| --- | --- |
| `src/eval_layouts/sampler.py` | 从任务 YAML、场景配置和资产元数据确定性采样；支持离散区间、关联型号、支持平面、杂物和方程任务 |
| `src/eval_layouts/storage.py` | 导出官方字段、原子写入、文件配对和哈希验证、完成清单 |
| `src/eval_layouts/validation.py` | 调用原生 `create_eval_env` / `EvalEnv.reset`，检查实例数量并从原生相机截图 |
| `src/eval_layouts/support_trajectory.py`、`teacher_solver.py` | 为排序新布局规划示范，导出原生 PKL 并进行物理回放 |
| `src/eval_layouts/lease.py`、`devices.py`、`watchdog.py` | 共享盘上的任务租约、GPU 分配、工作进程超时和故障恢复 |
| `scripts/*eval_layouts.py` | 生成、分编号并行、复查和审计入口；见 [脚本说明](../scripts/README.md#additional-evaluation-layouts) |
| `tests/test_eval_layouts.py` | 采样确定性、原生读取、元数据完成条件和排序几何约束 |

原生代码只增加两个可选数据根目录：`SeedManager` 读取 `eval_layout_root` / `ROBODOJO_EVAL_LAYOUT_ROOT`，排序任务读取 `eval_traj_root` / `ROBODOJO_EVAL_TRAJ_ROOT`。未设置时仍读取 Assets 中的官方集合。任务评分、动作接口、稳定性阈值和物理实现沿用原有代码。

采样器的输出经 `official_payload` 过滤为官方 `LayoutManager` 读取的 JSON 字段，文件内没有额外的元数据包装。实际采样种子、哈希、相机、渲染配置和回放结果存入目录外的同名 `metadata/<set>/<task>_<index>.json`。

日志、候选布局、工作进程租约和替换备份位于 `data/eval_layout_generation/<set>/`。共享盘的 `flock` 不能跨服务器保护写入，因此使用原子目录和独立进程维护租约心跳。看门狗仅监控本工具拥有的进程。

## 采样种子与复现

`--set-seed` 是整套评测集合的目录名，`--seed-start` 是候选布局的随机种子起点，`--count` 是每个任务保存的数量。v1 使用集合 `1000`、起点 `1000`、数量 `50`。

普通任务递增采样种子，放置或原生稳定性检查失败则继续采样。排序及其他分编号并行任务使用 `seed_start + index + count × k`，避免同时运行的编号使用相同候选种子。续跑读取已有记录和尝试日志；实际保留的种子见元数据，不应假设正好是 `1000..1049`。

采样时设置并恢复 NumPy 原有 RNG 状态，同时使用局部 Generator。在相同配置、资产和依赖下，同一采样种子生成相同 JSON；并行采样使用独立进程。种子不同不保证姿态不同，固定配置和有限排列仍会重复。保存的 JSON 固定实例型号和初始姿态；GPU 物理及渲染结果不保证逐位相同。

完整官方随机生成器未随仓库发布，因此不能证明自有采样与官方分布完全相同。自有集合使用完整资产目录，桌面和背景取任务场景默认值，可放不下的杂物按尽力放置处理。排序任务还会因示范规划或回放失败而拒绝候选。

方程任务的 YAML 描述完整算式，官方保存的 JSON 移除一个题块，两位数答案还增加 `mat5/t5`。采样器生成有解算式、保留原有垫子位置和标签，再随机移除一个数或运算符；原生解题判断和评分不变。

## 生成新集合

准备好完整 Assets 和仓库 `.venv`，在仓库根目录运行。例如用集合 `1001` 维护每任务 50 份布局：

```bash
.venv/bin/python scripts/generate_eval_layouts.py \
  --set-seed 1001 --seed-start 1050 --count 50 \
  --workers 4 --devices 0 1 2 3
```

用 `--tasks deposit_coin stack_bowls` 缩小任务范围。生成器自动为排序任务调用专用生成工具；已有样本经 JSON、PNG、轨迹校验和验证后续跑。全部任务完成前，整套 `manifest.json` 的 `complete` 保持 `false`。

默认在本机运行。若另一个服务器共享相同代码、环境和绝对挂载路径，显式添加 `--ssh-host <SERVER>`；工具通过 SSH 启动工作进程，文件仍保存到共享目录。`--workers` 可以超过 GPU 数量，但需考虑每个 Kit 进程的显存占用；工具会按实际剩余显存选择 GPU，并在验证记录中保存设备信息。

排序专用入口可按编号并行处理难通过的普通任务：

```bash
.venv/bin/python scripts/generate_sorting_eval_layouts.py \
  --task deposit_coin --set-seed 1001 --seed-start 1050 --count 50 \
  --workers 4 --devices 0 1 2 3
```

`--replace-indices` 和 `--replace-reason` 仅用于尚未发布集合中需替换的失败样本。旧 JSON、PNG 和元数据会保存到 `parallel/<index>/replaced/`。不要用替换或压缩工具改写已发布的 v1 数据。

## Isaac Sim 初始化与渲染

验证调用原生评测环境，包括布局恢复、原有 GPU PhysX 设置、300 步稳定性检查、相机预热和物理推进。检查布局有效标记、实际对象数量与 JSON 数量一致，随后运行原生奖励初始化。保存的 JSON 是采样时的初始姿态，没有用静止后的姿态覆盖。

模型服务器的 reset RPC 使用本地替身，验证阶段不调用模型推理。排序任务的辅助 Panda 仍执行原生物理控制和回放。

默认 `--render-profile fast` 使用 RaytracedLighting、TAA、直接光照和阴影，关闭反射、全局照明、DLSS 和 DLSSG，保留透明材质。截图只创建原生 `cam_head`，不创建两个未使用的腕部视图；头部相机位置、内参和 640×480 分辨率不变。每张 PNG 与初始化的 JSON 同名。

使用 `--render-profile official` 可恢复官方渲染及全部相机。`--warmup-spp` 仅调整 PathTracing 预热，最终 PathTracing 截图使用 16 spp。v1 的早期图片使用 RTX quality 或官方 `auto` 在 A800 上选择的 PathTracing，后续图片使用 fast；每张图的实际设置均保存在验证记录中。工具的渲染配置不写入 JSON，正式模型评测仍使用原有配置。

无关节物体任务复用原生 reset。有关节物体任务每份布局创建新世界，因为官方 CPU 包装器在第一次创建 PhysX 视图之前生成关节物体，而下一次读取布局会重新分配实例名称。验证证明 JSON 可独立初始化，不保证官方当前实现连续复用同一世界时切换关节物体正确。

Kit 工作进程定期重启以释放资源。后续截图省去不执行动作的 X5 规划器预热，保留物理机器人和原生初始关节状态；早期记录可能含原生规划器预热或相同参数的规划器复用。这些优化不更改正式模型评测。

## 排序示范

每份 `imitate_sorting_sequence` JSON 配套 `Traj/RoboDojo/imitate_sorting_sequence/<set>/<index>.pkl`。PKL 使用原生 `joint_path`、`target_label`、`aim_label` 和 `target_place_tag` 字段，示范物体与目标物体型号一致。

辅助 Panda 的原生 cuRobo 规划器在独立进程中运行，规划命令在 Isaac Sim 内执行和回放。抓取方向和篮内落点依据真实包围盒选择，检查物体可通过篮沿，允许在已入篮物体上方堆叠。导出前及原生回放后均检查五个 `aim` 物体进入 `basket1`。

为缩短示范，优先尝试导出步幅 2，失败后尝试 1.5 和原始序列。夹爪动作及接触稳定时间保留；每个版本重新读取同一布局并完整物理回放，只发布通过且不超过 16000 个物理控制步的版本。导出长度、步幅、失败尝试和回放结果均记录在元数据中。

`compact_sorting_trajectories.py` 是内部维护工具，要求当前有效的协调器租约和 token；不要独立运行或用于已发布集合。它也仅在原生回放通过后替换轨迹。

官方集合 0、1、2 各有 65 份辅助示范，控制步数中位数分别为 4567、4507、4454，全部范围为 3964–5655。v1 的最短、中位、最长分别为 4846、6970、11230；按原生 0.004 秒步长，约 19.4、27.9、44.9 秒。示范和模型动作在同一 episode 内推进，更长示范占用相同动作预算，因此自有集合并非与官方时长相同的评测条件。

## 审计与测试

完整审计需要布局、轨迹、元数据和全部配图。在恢复图片包后运行：

```bash
.venv/bin/python scripts/audit_eval_layouts.py --set-seed 1000 --count 50
```

审计逐文件检查 SHA-256、PNG RGB 维度和有效性、PKL 字段、对象对应关系、步数、关节数组形状及原生回放证据。写出 `manifest.json`、`comparison.json`、`index.html` 和 `preview.png`，不修改布局 JSON 或轨迹 PKL。官方与自有重复统计基于完整 JSON 哈希及规范化、四位小数姿态哈希，不能把轻微浮点差异都视为新的独立布局。

`verify_eval_layouts.py` 用于旧生成记录缺失原生奖励初始化证据的情况，v1 已完成这些检查：

```bash
.venv/bin/python scripts/verify_eval_layouts.py \
  --set-seed 1001 --tasks deposit_coin --workers 1 --devices 0
```

CPU 测试和代码检查不启动 Isaac Sim：

```bash
.venv/bin/python -m pytest -q \
  tests/test_eval_layouts.py tests/test_rendering.py tests/expert/test_expert.py
.venv/bin/ruff check src/eval_layouts scripts/*eval_layouts.py \
  scripts/compact_sorting_trajectories.py tests/test_eval_layouts.py
```

初始化和轨迹回放验证不等同于模型策略 rollout；模型能力仍需运行正式评测。

## 数据归档与发布

Git 跟踪工具、测试、恢复说明与 v1 校验文件；生成数据、日志、候选、压缩包、临时发布工具和认证目录均在忽略范围内。评测包包含 2700 个布局 JSON、50 个 PKL、2700 个验证记录、完成清单、差异报告和恢复说明，共 5453 个文件。图片包包含 2700 张同名 PNG、总览图和画廊，共 2702 个文件。

v1 评测包位于 `data/releases/eval-layout-1000-v1.tar.gz`，图片包位于 `data/archives/eval-layout-1000-v1-images.tar.gz`。两包在发布前已逐文件解压并比较 SHA-256；打包报告保存在 `data/releases/eval-layout-1000-v1-package-report.json`。

发布范围仅为 `origin_private` 和 `origin_xiaoyu` 的 Release，图片包保留本地。当前 GitLab 已发布核心包和校验文件，GitHub 上传状态保存在本地 `data/releases/eval-layout-1000-v1-publication-status.json`。已发布的数据使用固定版本和校验和，修改说明文档不重新打包已发布的 v1 附件。

旧自定义 YAML 和相关实现归档于 `data/archives/robodojo_custom_layouts_20261010.tar.gz`，当前集合没有沿用那些布局值。此前 notebook 验证属于临时工作，不纳入维护入口。
