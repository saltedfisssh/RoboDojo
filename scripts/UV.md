# RoboDojo 的 uv 使用方式

在仓库根目录运行以下命令。所有 `-uv.sh` 入口默认使用本仓库 `.venv`，
不需要先执行 `source .venv/bin/activate`。脚本也能从其他工作目录按绝对路径启动。

## 安装与资源下载

```bash
bash scripts/install-uv.sh
bash scripts/init_assets-uv.sh
bash scripts/robodojo-uv.sh doctor --skip-policy
```

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
