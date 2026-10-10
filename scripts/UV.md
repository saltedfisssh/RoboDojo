# RoboDojo 的 uv 使用方式

在仓库根目录运行以下命令。仿真 `-uv.sh` 入口默认使用本仓库 `.venv`，
不需要先执行 `source .venv/bin/activate`。脚本也能从其他工作目录按绝对路径启动。

## 安装与资源下载

```bash
bash scripts/install-uv.sh
bash scripts/init_assets-uv.sh
bash scripts/robodojo-uv.sh doctor --skip-policy
```

资源默认从 Hugging Face 下载，也可使用 [ModelScope 镜像](https://modelscope.cn/datasets/RoboDojo-Benchmark/RoboDojo/files)：

```bash
bash scripts/init_assets-uv.sh modelscope
```

两种来源均只下载 `Assets/**`，只依赖 Git/Git LFS，无需激活 Python/Conda 环境。
下载缓存按来源分开保存；已有完整 `Assets` 时会跳过下载。
可通过 `MODELSCOPE_REPO_ID`、`MODELSCOPE_REPO_URL`、`MODELSCOPE_REVISION` 覆盖
ModelScope 仓库配置，默认分支为 `master`；`--help` 可查看完整用法。

系统依赖已安装时可加 `--skip-system`；中断后可用 `--from isaacsim` 等参数续装。
安装脚本现在也安装资源下载需要的 Git LFS。安装器保留现有 Isaac Sim 版本约束，
并显式处理 IsaacLab 与 Starlette 的版本声明冲突；`uv pip check` 可能报告该已知冲突。
本项目的 `pyproject.toml` 没有声明完整仿真依赖，请用安装脚本配置环境。

```bash
# 自定义模拟器虚拟环境；相对路径以仓库根目录为基准。
bash scripts/install-uv.sh --venv /path/to/sim/.venv --skip-system
bash scripts/robodojo-uv.sh --venv /path/to/sim/.venv doctor --skip-policy

# 也可用于其他 uv 入口。
ROBODOJO_VENV=/path/to/sim/.venv bash scripts/eval_policy-uv.sh --help

# 数据和权重下载只依赖 Git/Git LFS，不会调用 Conda。
bash scripts/RoboDojo/download_data-uv.sh huggingface demo
bash scripts/RoboDojo/download_ckpt-uv.sh huggingface ACT
```

`doctor` 会检查资源、配置、任务清单、环境路径及 Isaac 导入；
`--skip-isaac` 只跳过 Isaac 导入。缺少 Assets 时报告失败属于正常诊断结果。

## 数据生成与独立导出环境

每个人 clone 到自己的路径后创建自己的环境；不依赖开发者目录、相邻 LeRobot checkout
或手动安装的 `.cache/datagen-export-deps`。下面的命令不安装 XPolicyLab 的策略模型依赖：

```bash
# 新 clone：安装仿真与导出环境，再下载资源
bash scripts/install-uv.sh --with-export
bash scripts/init_assets-uv.sh
# 若系统依赖已具备，安装时可加 --skip-system

# 已有可运行的仿真环境：只补数据生成依赖并创建导出环境
bash scripts/install-uv.sh --from datagen_deps --with-export

# 仅处理已有 HDF5 的机器：无需 Isaac、GPU 或 Assets
bash scripts/install-lerobot-uv.sh
bash scripts/install-lerobot-uv.sh --verify-only
```

| 环境 | 版本与职责 | 自动使用它的入口 |
| --- | --- | --- |
| `.venv` | Python 3.11、NumPy 1.26.0、Isaac Sim 5.1 / cuRobo、h5py | `collect`、`batch`、`restore`、`dev run` |
| `.venv-lerobot` | Python 3.12、NumPy 2.2.6、LeRobot 0.6.1、CPU PyTorch、h5py / 视频库 | `export`、`dev qualify` 的导出/读回、旧版 `export/review` |

这是依赖兼容约束，不是 v3 文件格式要求：Isaac Sim kernel 固定 `numpy==1.26.0`，
IsaacLab 要求 `numpy<2`；LeRobot 0.4.4 的 rerun-sdk 依赖链要求 NumPy 2。
当前导出代码还使用 0.6.1 的 `RGBEncoderConfig` 和 language schema API，
[LeRobot 0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml)
自身要求 Python ≥3.12、NumPy ≥2。删除 XPolicyLab 并不能解决这些冲突。
不要在仿真环境直接安装 LeRobot，也不要用 `--no-deps` 或 PYTHONPATH overlay 绕过约束。
这里的 LeRobot **库版本 0.6.1** 输出的是 **数据格式 v3.0**。

导出依赖从 PyPI 安装 `lerobot[dataset]`；不安装训练、硬件和可视化 extras。
CPU 版 PyTorch 不需要 GPU 驱动，但完整读回验证需要系统 FFmpeg 共享库；Ubuntu 可用
`sudo apt-get install ffmpeg`。安装器会检查实际 HDF5、语言 schema、H.264 与 TorchCodec 导入，
并拒绝与仿真 venv 相同或通过软链接指向同一目录的安装目标。

```bash
# 路径可自选，相对路径始终以仓库根目录为基准。
export ROBODOJO_VENV=.venv
export LEROBOT_VENV=.venv-lerobot
bash scripts/install-lerobot-uv.sh
bash scripts/generate_data.sh export --input data/expert/raw/task --output data/expert/lerobot/task
bash scripts/run-lerobot.sh scripts/internal/validate_expert_dataset.py \
  --dataset data/expert/lerobot/task --report data/expert/validation.json
```

无需先 activate。`LEROBOT_PYTHON` 优先于 `LEROBOT_VENV`，仅用于显式复用已有兼容解释器；
安装器始终使用 `--venv` / `LEROBOT_VENV`，不会修改当前激活的环境。自定义安装目录后，
后续命令也要保留相同的 `LEROBOT_VENV`。`run-lerobot.sh` 重设 PYTHONPATH 为当前仓库，
避免旧 overlay 或其他 Python 版本的二进制包混入；外部标注 hook 应安装到导出环境中。
可选 Argus 看板需要显式设置自己的 `ARGUS_ROOT`，不属于采集/导出安装前提。

`requirements-lerobot.txt` 固定全部解析依赖，`.in` 保存直接依赖。维护者升级时在 Linux x86_64
上重新生成，然后重新执行真实导出与官方读取器验证；不要在日常运行时自动升级：

```bash
uv pip compile scripts/requirements-lerobot.in --python-version 3.12 --torch-backend cpu \
  --output-file scripts/requirements-lerobot.txt --no-annotate
```

2026-10-09 环境适配验证（Linux x86_64）：从零创建本仓库 `.venv-lerobot`，72 个包的
依赖检查、原生视频库检查与组合入口 `install-uv.sh --from verify --with-export` 均通过。
仿真环境的 315 个包版本前后完全一致；仿真侧与启动器回归 192 passed / 5 skipped，
独立 NumPy 2 环境的数据回归 80 passed / 5 skipped。
从仓库外启动，重新导出已有 `stack_blocks_by_language` 的 3 条 RGB 轨迹，共 897 帧；
官方 LeRobot reader 抽查 9 帧，三路视频共 2691 帧全部解码，全部动作/状态与原始 HDF5 一致。
产物在 `data/expert/development/stack_blocks_by_language/environment-portability/lerobot`，
安装与检查记录在 `.cache/environment-portability/`。另有不下载依赖的临时 clone 测试覆盖
空格路径、相对路径、解释器优先级、缺失环境、退出码和禁止覆写仿真环境。
此次没有重新从零安装完整 Isaac 栈，也没有重跑 GPU 采集/恢复；原任务资格记录仍对应其
原始源码快照。环境入口已纳入源码指纹，后续完整资格验收应重新采集同一冻结版本的证据。

## 策略环境与评测

模拟器和策略服务分别使用自己的环境。模拟器固定 `websockets==12.0`，
当前 XPolicyLab 策略服务需要 `websockets>=14`，因此不要把策略服务依赖直接装进模拟器环境。
各策略的模型依赖、CUDA/PyTorch 要求和检查点布局仍以该策略说明为准。

`--policy-env` 推荐传绝对路径，可以是虚拟环境目录或包含 `.venv` 的项目目录。
`uv` 则按策略的 `deploy.yml` 中 `policy_uv_env_path` 解析；未配置时使用该策略目录的 `.venv`。
原先接受 Conda 环境名的启动器仍保留相应分支。

```bash
# 单任务，启动策略服务并运行模拟器。
bash scripts/robodojo-uv.sh eval \
  --policy-dir XPolicyLab/policy/ACT --task stack_bowls \
  --ckpt YOUR_CHECKPOINT --policy-env /path/to/act/.venv --eval-num 1

# 批量任务；加 --dry-run 可先检查命令。
bash scripts/robodojo-uv.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT --ckpt YOUR_CHECKPOINT \
  --policy-env /path/to/act/.venv --only stack_bowls,push_T \
  --gpu-ids 0,1 --eval-num 1 --dry-run

# 独立策略服务：仅部署策略的机器可用 --venv 指定其运行环境。
bash scripts/robodojo-uv.sh --venv /path/to/act/.venv server \
  --policy-dir XPolicyLab/policy/ACT --task stack_bowls \
  --ckpt YOUR_CHECKPOINT --policy-env /path/to/act/.venv \
  --policy-port 9999 --bind-host 0.0.0.0

# 模拟器客户端连接已启动的服务。
bash scripts/robodojo-uv.sh client --task stack_bowls --policy-name ACT \
  --policy-host 127.0.0.1 --policy-port 9999 --eval-num 1
```

`smoke`、`benchmark`、`dimensions`、`tasks`、`summarize` 都通过同一个
`robodojo-uv.sh` 入口使用，参数与原入口一致。低层入口为
`eval_policy-uv.sh`、`internal/verify_install-uv.sh` 和 `internal/smoke_all_tasks-uv.sh`。

## 渲染兼容与降噪

公共 `BaseEnv` / `TaskEnv` 默认使用 `sim.render.profile: auto`，评测、数据生成和
预览使用同一选择逻辑，支持以下 profile：

| profile | 行为 |
| --- | --- |
| `auto`（默认） | 检查渲染 GPU。A800/A100/A30（8.0）、H100/H200/H800（9.0）等已知无 RR 设备选择低噪路径；RTX 3090/4090/5090 等保留原有配置。识别失败或架构未知时保留原配置并在日志注明。 |
| `rtx` | 沿用原有 quality / DLAA 配置及用户自定义覆盖。 |
| `low_noise` | 优先尝试已注册的 Minimal / Textured Diffuse；不可用或实际未生效则使用 PathTracing + OptiX。 |
| `minimal` | 显式请求 Minimal；同样在不可用时回退到 PathTracing + OptiX。 |
| `pathtracing` | 直接使用 PathTracing + OptiX，默认每像素 16 样本。 |

可通过环境变量对任意现有入口指定 profile，环境变量优先于 YAML：

```bash
# A800 上显式选择已验证的兼容渲染。
ROBODOJO_RENDER_PROFILE=pathtracing bash scripts/robodojo-uv.sh client \
  --task deposit_coin --policy-name ACT --policy-host 127.0.0.1 --policy-port 9999 --eval-num 1

# 支持 RR 的设备通常无需设置；也可以显式保持原有 RTX 路径。
export ROBODOJO_RENDER_PROFILE=rtx
```

也可修改 `env_cfg/sim/sim_config.yml`（该文件本身就是 `sim` 配置）：

```yaml
render:
  profile: auto
  pathtracing_spp: 16
```

低噪 profile 会关闭反射、GI、translucency、DLSS-RR、DLSS 帧生成和 DLAA，保留
直接光照与阴影。它会把 `antialiasing_mode` 设为 `null`，先等待所有相机 render product
产出数据，再切换渲染模式；reset 后重新应用设置，模式切换后的延迟 AA 覆盖也会被校正。
相机创建后立即启用 PathTracing 可能留下空 RGB 缓冲，因此初始化期间会先用实时模式
建立相机数据通路；这些预热帧不写入数据集。空缓冲会报错，避免 Warp 读取越界。
兼容路径同时清除 Replicator 的 `/rtx-transient/post/aa/limitedOps`，否则 Kit 5.1
会把关闭的 AA 重新限制为 DLSS/DLAA。
`/rtx/post/aa/op=0` 才是关闭 AA，`/rtx/post/dlss/execMode=0` 本身表示 Performance。
低噪 profile 的关键设置优先于同名自定义覆盖；需要完全自定义渲染时用 `rtx`。

每次 reset 的 `[RoboDojo render]` 日志包含请求的 profile、实际选择、GPU 和 live Carb
值。关键设置在渲染更新后仍不匹配会报错，避免静默生成噪声图像。相同信息可以通过
`env.render_metadata` 获取，并写入 expert HDF5 的 `meta.render` 和旧版采集的
`metadata.render`；物理、动作和 checkpoint 数组格式不变。

本机 Isaac Sim 5.1 没有注册 Minimal 模式，使用 PathTracing + OptiX。硬件识别依据
[NVIDIA 支持表](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_architecture_support.html)，
不把所有 Ampere 卡视为无 RR；识别检查的是渲染设备，物理可继续在 CPU 上运行。
硬件支持也需要兼容的 Kit 和驱动：例如 Linux Blackwell 的 RR 要求 Kit ≥106.5.3。
若特殊驱动、GPU 映射或自定义启动器导致自动判断不适用，请显式选择 profile。

PathTracing 可能降低动态场景的吞吐，关闭部分光照效果也会改变图像分布。应保留 profile 元数据，
在同一仿真 tick 比较原生 RGB 的连续帧 MAD、3×3 median residual 和 residual >5
像素比例，并检查三路视频尺寸、帧数和 fps。median 只用于计算诊断指标，输出图像
没有额外滤波。RTX 机器上的最终画质仍需在对应机器上验收。

2026-10-10，A800 / Isaac Sim 5.1 的 `deposit_coin` 静态场景对比：同一仿真 tick，
三路相机各采集 12 张原生 RGB（640×480、uint8）。PathTracing + OptiX 16 spp
使 MAD 从 5.87 / 2.56 / 2.73 降到 0，residual >5 像素比例降低 96.4%–97.3%，
平均 median residual 降低 83.6%–85.8%（未达到指南中该指标降低 90% 的目标）。
三路 MP4 的 640×480 / 12 帧 / 25 fps 均已完整解码验证，reset 后关键设置仍有效。
默认 `auto` 的全新进程启动及执行动作后的三路相机刷新也已通过实机检查。
静态 PathTracing 达到样本上限后会停止累积，因此 MAD=0 仅代表该静态测试。
RTX 5090 的配置保留由回归测试覆盖，本机没有 5090，未做该型号的实机画质验证。
额外测试在同一 stage 内直接销毁并重建 render products 时，Kit 5.1 在 annotator
注册阶段报 `AnnotatorRegistryError`，该通路尚未通过实机验证；上面通过的是全新
进程初始化、动作后采集及常规 reset 复用相机。

## 无 GPU 的 demo 调试闭环

下面使用独立的轻量策略环境；无需模型权重，不启动 Isaac Sim，也不依赖 Assets：

```bash
uv venv --python 3.11 XPolicyLab/policy/demo_policy/.venv
uv pip install --python XPolicyLab/policy/demo_policy/.venv/bin/python -e XPolicyLab

EVAL_ENV_TYPE=debug bash scripts/robodojo-uv.sh eval \
  --policy-dir XPolicyLab/policy/demo_policy --task stack_bowls \
  --ckpt demo --policy-env "$PWD/XPolicyLab/policy/demo_policy/.venv" \
  --action-type joint --eval-num 1
```

调试客户端的回合数由 XPolicyLab 调试脚本自身控制。测试应到达
`[MAIN] eval finished`，并清理策略服务进程。要验证图像编解码通路，可再加
`DEBUG_OBS_ENCODED=1`。这不能代替实际 GPU 渲染验证。

这次适配包含 XPolicyLab 子模块内的公共环境辅助脚本、客户端和策略启动脚本，
保存或提交改动时需要一并保留；不会自动安装任何真实策略的模型依赖。
Docker 入口仍沿用原有镜像流程。
