# 自有 RoboDojo 评测布局

集合 `1000`（v1）覆盖 54 个可运行任务，每个任务 50 份布局，共 2700 个官方格式 JSON，以及 50 份与排序模仿布局匹配的原生格式 PKL。全部布局通过 Isaac Sim 原生初始化和文件审计；排序示范另通过完整物理回放。2700 张同名 PNG 是实际初始化后的头部相机图片。

## 下载与恢复

从 [origin_xiaoyu 的数据 Release](https://gitlab.xiaoyu.ai/librain/robosim/RoboDojo/-/releases/eval-layout-1000-v1) 下载以下两个附件，需要该私有仓库的访问权限：

- `eval-layout-1000-v1.tar.gz`：布局、轨迹、验证元数据、完成清单、官方差异报告和恢复说明，约 10.77 MiB。
- `eval-layout-1000-v1.sha256`：与 Git 中的 [校验文件](eval-layout-1000-v1.sha256) 相同。

在下载目录校验，然后在仓库根目录解压：

```bash
sha256sum -c eval-layout-1000-v1.sha256
tar -xzf eval-layout-1000-v1.tar.gz -C /path/to/RoboDojo
```

压缩包保留 `Eval_Layout/` 前缀。恢复后直接读取固定的 JSON 和 PKL，无需重新采样。仍需原有 Assets 和 Isaac Sim 环境。

```text
Eval_Layout/
  RoboDojo/arx_x5/1000/<task>_<index>.json
  Traj/RoboDojo/imitate_sorting_sequence/1000/<index>.pkl
  metadata/1000/<task>_<index>.json
  manifest.json
  comparison.json
```

编号为 `0..49`；`manifest.json` 的 `complete` 为 `true`。`metadata` 记录实际采样种子、文件哈希和验证结果，`comparison.json` 记录官方差异和重复统计。

## 用于模型评测

在仓库根目录设置数据根目录，并在原有评测命令中使用 `--seed 1000`：

```bash
export ROBODOJO_EVAL_LAYOUT_ROOT="$PWD/Eval_Layout"
export ROBODOJO_EVAL_TRAJ_ROOT="$PWD/Eval_Layout/Traj"
bash scripts/robodojo-uv.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --task deposit_coin --seed 1000 \
  --ckpt <CKPT> --policy-env <ENV>
```

Conda 环境可使用 `scripts/robodojo.sh`。配置字段 `eval_layout_root`、`eval_traj_root` 优先于上述环境变量；未设置时读取 `Assets/Eval_Layout` 和 `Assets/Traj` 中的官方数据。

对于未增加自定义路径选项的官方代码，将集合复制到原有 Assets 路径即可使用相同 JSON/PKL：

```bash
mkdir -p Assets/Eval_Layout/RoboDojo/arx_x5
mkdir -p Assets/Traj/RoboDojo/imitate_sorting_sequence
cp -a Eval_Layout/RoboDojo/arx_x5/1000 Assets/Eval_Layout/RoboDojo/arx_x5/
cp -a Eval_Layout/Traj/RoboDojo/imitate_sorting_sequence/1000 Assets/Traj/RoboDojo/imitate_sorting_sequence/
# 使用原有评测命令的 --seed 1000
```

目录 `1000` 选择整套评测集合；它不等于每份文件的采样种子。任务原生评测数量限制仍适用，部分任务上限为 25，默认不会自动跑完维护的全部 50 份布局。

## 图片与 Git 管理

图片包 `eval-layout-1000-v1-images.tar.gz` 约 1.22 GiB，仅暂存于本地 `data/archives/`，没有上传 Release。配图用于查看初始化效果和完整文件审计，模型评测不需要它。按需在仓库根目录恢复：

```bash
tar -xzf data/archives/eval-layout-1000-v1-images.tar.gz
```

恢复后可打开 [图片画廊](index.html) 或 [54 个任务的渲染总览](preview.png)。每张 PNG 位于布局 JSON 旁边，并与其同名。完整审计要求恢复图片包：

```bash
.venv/bin/python scripts/audit_eval_layouts.py
```

布局 JSON、轨迹 PKL、验证元数据、PNG 和预览文件由 `.gitignore` 排除；Git 只保留本目录的说明与 v1 校验文件。压缩包位于被忽略的 `data/releases/` 和 `data/archives/`，评测包通过 Release 分发。已发布的 v1 压缩包及其校验和固定不变，后续数据变更使用新版本。

## 与官方的差异及评测边界

| 项目 | 官方集合 | 自有集合 `1000` |
| --- | --- | --- |
| 格式与目录 | `RoboDojo/arx_x5/{0,1,2}/<task>_<index>.json`，排序另有 PKL | 同一格式，集合目录为 `1000` |
| 数据来源 | 官方发布的固定文件，完整生成器未公开 | 从现有任务配置和完整资产目录重新采样，筛选稳定候选 |
| 数量 | 每套 54 个任务、2690 个 JSON，各任务份数不同 | 每任务 50 份，共 2700 个 JSON |
| 初始化、物理、评分 | 原生评测流程 | 原生评测流程；额外逐文件验证和截图 |
| 初始化截图 | 官方 `auto` 在 A800 上选择 PathTracing 16 spp | 以较快的 RaytracedLighting + TAA 为主；正式模型评测使用原有渲染配置 |
| 排序辅助示范 | 控制步数 3964–5655 | 4846–11230，中位数 6970；格式和原生回放逻辑相同 |

自有集合成绩应单独报告。格式和初始化流程一致，不代表采样分布经过统计证明与官方相同。更长的排序示范会占用同一 episode 的动作预算；预算未增加。

固定或有限排列任务保留重复布局。官方集合 0 的 `cover_blocks` 有 55 份 JSON、6 种独立布局，`play_tic_tac_toe` 有 75 份 JSON、1 种物体姿态。自有集合沿用任务随机范围，不强制去重。

有关节物体的布局逐份使用新世界验证，不保证官方当前实现连续复用同一世界时切换关节物体正确。全部布局的初始化验证已完成，本次没有运行模型策略 rollout。

同一采样种子在相同配置、资产和依赖版本下生成相同 JSON。保存的 JSON 固定型号和初始姿态；GPU 物理与渲染结果可能随环境变化。生成流程、逐图渲染配置、排序轨迹和验证边界见 [维护文档](../docs/EVAL_LAYOUTS.md)。
