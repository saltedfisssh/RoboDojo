# 数据生成扩展：代码结构与维护

本扩展位于`src/datagen`，复用RoboDojo任务、双X5机器人和cuRobo v2。运行命令见[README](README.md)，实际可用范围见[CAPABILITIES](CAPABILITIES.md)，当前暂停的实验见[PAUSED_STATE](PAUSED_STATE.md)。

## 模块分工

| 层次 | 入口/模块 | 责任 |
| --- | --- | --- |
| 命令入口 | `scripts/datagen-uv.sh` | 采集/恢复使用RoboDojo环境，导出/检查页使用独立LeRobot环境 |
| 单条采集 | `collect.py`、`runtime.py` | 初始化真实环境、动作适配、运行专家、记录终态与失败原因 |
| 布局 | `layouts.py`、`curriculum.py`、`*_layouts.py` | 独立训练分布、稳定label、资产/工装的成组变化；不读取Eval_Layout |
| 专家 | `planner.py`、`ik.py`、`programs/` | 原语、连续IK、任务顺序、抓持反馈及原终态检查 |
| 规划世界 | `collision_world.py`、`contact_diagnostics.py` | 显式静态障碍快照与可选真实接触诊断；不是完整动态碰撞系统 |
| 执行隔离 | `lease.py`、`frozen.py`、`provenance.py` | GPU排他队列、启动时源码冻结、版本/资产/源文件哈希 |
| 批处理 | `batch.py`、`campaign.py`、`dispatch.py` | 单进程单episode监督、无图像预验、跨分片补采与seed去重 |
| 原始记录 | `recording.py` | HDF5、异步RGB写入、逐帧检查点、partial文件与原子发布 |
| 状态 | `state.py`、`articulations.py`、`secondary_state.py`、`particle_transforms.py` | 机器人、外部刚体/关节、粒子、控制器与任务历史保存和恢复 |
| 交互/显示适配 | `interaction.py`、`support_demo.py`、`asset_adapters.py`、`visual_state.py`、`render_guard.py` | 对手游标、支持臂示范、场景适配版本、USD/物理显示同步与渲染保护 |
| 导出/标注 | `export.py`、`annotations.py`、`atomic.py` | LeRobot v3、语言schema、原子切片、来源映射和原子目标度量 |
| 恢复/模型接入 | `replay.py`、`policies.py` | 按帧/时间/subtask恢复，原动作或VLA继续执行 |
| 诊断 | `planner_probe.py`、`probe_programs.py`、`state_probe.py`、`camera_probe.py`、`planning_diagnostics.py` | 有界检查点后缀、状态/显示与规划证据；不作为完整训练episode |
| 补图/检查 | `render_episode.py`、`promote.py`、`review.py`、`review_page.py` | 成功源轨迹补图、有效数据汇总、视频/subtask检查页 |
| 任务定义 | `tasks35.json`、`inventory.py`、`reconstructed.py` | 固定35类名单、上游工程库存、明确标记的重建任务 |

主流程是：启动并冻结源码 → 采样训练布局 → 获取GPU锁并初始化环境 → 真实专家执行 → HDF5成功过滤 → LeRobot/语言标注/状态sidecar。恢复直接读取所选帧的检查点；补图模式与独立物理重放分别记录，不能混为新增独立示范。

`env/planner_manager/curobo_planner.py`的本地调整仅包括：遵循运行配置中的CUDA graph开关；为motion与IK世界在warmup前预留32个cuboid槽位，以支持当前cuRobo dev100的障碍更新。没有替换其v2 API，也未修改官方奖励条件。

## 数据契约

- 25 Hz、14维绝对关节动作；第i行是执行`action[i]`前的图像和状态，`terminal`是最后动作执行后状态。
- episode先写partial，完整结束后发布；`complete`与`success`分开，进程退出码不代替文件判据。无RGB、失败和诊断后缀不能进入完整成功训练集。
- subtask边界是半开区间`[start_frame,end_frame)`；原子切片重置局部时间并保留源帧偏移。
- 任务条件中的tuple/list有不同逻辑，使用显式JSON编码保留；检查点不使用pickle。
- 来源数据、源码版本、资产与场景适配信息必须保留。每个等待worker已冻结源码，后续编辑不会改变它实际执行的程序。
- 布料恢复要求动态mesh变换与粒子数组同时存在；v1旧布料检查点不能作为可信中途恢复。恢复小误差不保证后续物理逐bit相同。

字段、标注hook和VLA接口的完整说明见[使用文档](README.md)。

## 修改任务程序

1. 在能力表定位正式sampler和`programs/`中的实际入口，先确认对象label、模型和当前已验证初态。
2. 诊断程序可放在`probe_programs.py`，但必须说明适用任务及前置检查点；禁止把已经抓稳/闭盖后的程序直接当成完整初态专家。
3. `planner_probe`最多600帧，按实际保存结果检查`restore_passed`、`program_completed`及`task_after_suffix_passed`。后缀即使通过整任务终态，也仍是诊断数据。
4. 将通过的动作组合接入正式程序后，以新版本目录重新验证完整初态；再采新seed、生成RGB、导出和读回。不要改变原奖励以消除失败。
5. 修改布局范围时同步[能力表](CAPABILITIES.md)与版本记录；修改状态/标注契约时添加有实际语义的回归测试。

`probe_programs.py`保留了历史候选，并不表示其中每个函数都有效或已经采用。生产入口使用`programs/`，实验采用状态以暂停现场和验证报告为准。

## 检查命令

以下命令在仓库根目录执行。这些检查不启动Isaac仿真，不恢复暂停的采集队列。分别在两个已有环境执行，是因为LeRobot与RoboDojo依赖不同。

```bash
# RoboDojo：记录、状态契约、原子官方判据、队列和监督等
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=. .venv/bin/python -m pytest tests/datagen -q -ra

# LeRobot：包含真实LeRobot writer/reader与视频导出往返
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash scripts/run-lerobot.sh -m pytest tests/datagen -q -ra

.venv/bin/python -m ruff check src/datagen tests/datagen env/planner_manager/curobo_planner.py
.venv/bin/python -m ruff format --check src/datagen tests/datagen
bash -n scripts/datagen-uv.sh
git diff --check
```

导出环境通过 `bash scripts/install-lerobot-uv.sh` 创建，详见README。普通Python进程默认找不到Sim自带USD时，USD相关测试会跳过；用当前安装的Isaac Sim 5.1 USD库独立补验：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess

root = Path.cwd()
sim = root / '.venv/lib/python3.11/site-packages/isaacsim'
usd = next((sim / 'extscache').glob('omni.usd.libs*'))
env = dict(os.environ)
env['PYTHONPATH'] = f'{usd}:{root}'
libraries = [usd / 'bin', sim / 'kit', sim / 'kit/kernel/plugins', sim / 'kit/libs']
env['LD_LIBRARY_PATH'] = ':'.join(map(str, libraries)) + (
    ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''
)
result = subprocess.run([
    '.venv/bin/python', '-m', 'pytest', '-q', '-ra',
    'tests/datagen/test_articulations.py',
    'tests/datagen/test_particle_transforms.py',
    'tests/datagen/test_visual_state.py',
], env=env)
raise SystemExit(result.returncode)
PY
```

这是USD库专项，未创建AppLauncher。仿真层面的成功与恢复续跑需要单独实测；CPU测试通过不替代它们。

## Git与本地运行产物

Git包含`src/datagen`、`tests/datagen`、启动脚本、文档和cuRobo适配修改。`data/`、`Assets/`、`.venv/`、`.cache/`以及Python缓存由已有ignore规则排除；HDF5、视频、LeRobot目录和失败日志保留在本机共享盘，不随代码提交。

文档中指向`data/datagen_review35`的链接是本机证据入口，克隆Git本身不会得到这些文件。临时实验编排脚本已另行归档到该数据目录；正常collect/export/replay模块不依赖`/tmp`脚本。暂停PID记录只适用于本次运行，换主机或进程重启后必须重新核对。
