# HPS3DGS 版本管理与日常操作

HPS3DGS 的正式入口位于 `/disk3/ydz/code/HPS3DGS`。主仓库保留原 PAHC 的 `src/`、`scripts/`、`configs/`、`tests/` 结构，通过三个子模块固定研究实现和后端。整合目的是让代码来源、验证证据和恢复方式清楚，不代表算法已经改进。

本文中的命令在远端 Linux Bash 执行。它们是维护流程，不表示这些分支、标签或验收结果已经存在。实际状态以 Git 和本次验证报告为准。运行前先替换示例分支名、BUG 编号和提交号；命令失败即停止，不接着提交、合并或打标签。

## 1. 一次发布包含哪些版本

| 仓库 | 职责 | 版本如何固定 |
|---|---|---|
| HPS3DGS | PAHC、HAC/HAC++ 适配、统一启动、配置和验证 | 主仓库提交及发布标签 |
| `third_party/SegAnyGAussians` | Geo33/存储消融、独立 Fig. 7 和各自解码器 | 主仓库 gitlink 固定完整提交号 |
| `third_party/HAC` | 原生 HAC 训练及编解码 | 主仓库 gitlink 固定完整提交号 |
| `third_party/HAC-plus` | 原生 HAC++ 训练及编解码 | 主仓库 gitlink 固定完整提交号 |

不要只保存主仓库 HEAD。每次发布还要保留子模块提交对象、递归依赖版本、环境记录和输入/参考结果清单。`release_manifest.json` 记录被纳入清单的文件 SHA256；修改这些文件后必须使用 `scripts/release_manifest.py refresh` 刷新并核对清单，再一起提交，不要靠关闭哈希检查获得通过。数据、checkpoint、压缩包和大结果留在独立目录，用路径、大小和 SHA256 关联；不把它们整体加入普通源码 Git。

本次整合的 Git 来源采用服务器本地 bare 仓库，保存在 `/disk3/ydz/code/HPS3DGS-repositories`。这些是本机可恢复来源，外部 GitHub/GitLab 平台尚未配置；本机 clone/fetch 成功不等于已经有异机备份。SegAny 内的 kmeans、segment-anything、CLIP 也须按递归子模块保存。

本次整合的来源基线为：PAHC `e9ebeafc1f09920e6393e757f08a1eecb4d291db` 加已有 HAC 适配增量；SegAny `669fe1987b88b940daf22f0638c1c2384403cce7` 加 Geo33 存储消融、独立 Fig. 7 和相关测试；HAC `dfc2821bc04ac5c4d1d6b9b5b4fba52b94cd958b` 加三处现有补丁；HAC++ `36088aa43d00b62d479c4b99ca3f7f174cef7ec7`。这些是来源，不是整合后的发布提交号。

## 2. 分支与标签

| 名称 | 用途 | 可否作为正式基线 |
|---|---|---|
| `main` | 已完成相应验证并检查过 diff 的版本 | 可以，仍需读该版本验证范围 |
| `integrate/<主题>` | 初次整合或跨组件迁移 | 验收完成前不可以 |
| `fix/<编号>-<简述>` | 局部 bug 修复 | 合入后可以 |
| `feature/<简述>` | 功能、性能或算法开发 | 原型完成不等于可以 |
| `repro/<编号>-<简述>` | 复现、诊断及最小输入 | 默认不合入临时输出 |
| `release/<版本>` | 冻结候选内容并完成发布验收 | 通过后推进 `main` |
| `snapshot/<来源>-<日期>` | 保存历史源码及本地增量 | 仅证明可恢复，不证明可用 |

用 `v0.1.0-rc.1` 一类标签表示初次候选版本；只有发布范围全部通过 `docs/VALIDATION_POLICY.md` 的要求后，才创建 `v0.1.0`。不要在证据不足时先给正式标签，再把验证放到以后。

后续约定：修复兼容行为用补丁版本（如 `v0.1.1`）；新增兼容路线或功能用次版本（如 `v0.2.0`）；破坏已有公共接口或包格式时单独说明兼容策略，必要时升级主版本。码流自己的格式版本必须显式记录，不能只靠项目版本推断。已发布标签保持不变；发现问题发布新版本，保留旧版本证据。

不要求长期维护一个不断变化的 `develop`。每个任务从已验证基线建立短期分支，以最终验证过的提交快进到 `main`，避免未经检查的合并结果进入主线。

## 3. 开始一次 bug 修复

先登记 `docs/templates/BUG_REPORT.md` 中的信息。症状、输入和预期不明确时，先做只读检查和复现，不凭最终指标猜测原因。

```bash
set -euo pipefail
ROOT=/disk3/ydz/code/HPS3DGS
git -C "$ROOT" status --short --branch
git -C "$ROOT" rev-parse HEAD
git -C "$ROOT" submodule status --recursive
git -C "$ROOT" worktree list
```

如当前目录有未提交修改，记录它们，不自动 stash、提交或丢弃。确认 `main` 是已通过验收的基线；初次整合尚未通过时，从明确记录的整合提交开始，不把它称为正式基线。

```bash
BRANCH=fix/BUG-001-decoder-example
WT=/disk3/ydz/code-worktrees/HPS3DGS/BUG-001-decoder-example
test ! -e "$WT"
git -C "$ROOT" worktree add -b "$BRANCH" "$WT" main
git -C "$WT" -c protocol.file.allow=always submodule update --init --recursive
git -C "$WT" status --short --branch
git -C "$WT" submodule status --recursive
```

停止条件：目标目录已经存在；分支名冲突；子模块取不到固定提交；子模块状态以 `-`、`+` 或 `U` 开头；共享运行环境无法定位；实际路径与计划不一致。先解决具体原因，不改为从任意最新提交运行。本仓库依赖使用受控服务器本地 bare 路径，恢复命令使用单次 `-c protocol.file.allow=always`；不要设为全局配置。若将来明确改为外部 HTTPS 依赖，可省略该参数。若当前 Git/子模块组合不能可靠支持工作树，在新目录建立独立 clone，固定同一基线及递归子模块，不能共享可写的子模块目录来绕过隔离。

在新工作树复现故障，把命令、退出码、日志及最小必要输入保存在独立的新实验目录。先做能区分修复前后的验证，再修改相关代码。检查 `git diff`，按明确文件名暂存，不用未经核对的全目录 `git add -A`。

```bash
git -C "$WT" diff --check
git -C "$WT" diff --stat
git -C "$WT" diff
# 以下路径替换为本次实际修改的文件。
git -C "$WT" add -- src/example.py tests/test_example.py
python "$WT/scripts/release_manifest.py" refresh --root "$WT" --output release_manifest.json
git -C "$WT" add -- release_manifest.json
git -C "$WT" diff --cached
git -C "$WT" commit -m "fix(codec): correct BUG-001 decoder behavior"
```

提交消息说明具体问题与结果。一次 bug 修复不要夹带无关格式整理、实验参数变更和算法重写。提交前后的试运行可以辅助定位，合入依据应覆盖最终提交。

## 4. 修改子模块时先子后父

只改主仓库适配器时不需要改原生后端。确实需要改 SegAny/HAC/HAC++ 时，在任务工作树里的对应子模块新建分支，保存原 HEAD，完成局部修复和该组件的验证后提交。下面以 HAC 为例。

```bash
DEP="$WT/third_party/HAC"
git -C "$DEP" status --short --branch
git -C "$DEP" rev-parse HEAD
git -C "$DEP" switch -c fix/BUG-001-hac-example
# 修改并运行该组件的针对性验证后：
git -C "$DEP" diff --check
git -C "$DEP" add -- scene/dataset_readers.py
git -C "$DEP" diff --cached
git -C "$DEP" commit -m "fix(data): correct BUG-001 camera handling"
git -C "$DEP" status --short
git -C "$DEP" rev-parse HEAD
```

必须保证该提交可从可访问的 Git 来源或经过恢复检查的 Git bundle 找回。不要记录一个仅存在于即将丢弃工作树的提交。没有远端发布授权时，先在新的归档目录保存依赖 bundle；主仓库也需单独归档，父 bundle 不包含子模块对象。

```bash
BACKUP=/disk3/ydz/archives/HPS3DGS/BUG-001-example
test ! -e "$BACKUP"
mkdir -p "$BACKUP"
git -C "$DEP" bundle create "$BACKUP/HAC.bundle" HEAD --branches --tags
git -C "$DEP" bundle verify "$BACKUP/HAC.bundle"
git -C "$DEP" rev-parse HEAD > "$BACKUP/HAC.commit.txt"
```

随后更新主仓库子模块指针和对应版本清单，提交适配器改动及验证说明，再运行跨组件验证。

```bash
git -C "$WT" diff --submodule=log
git -C "$WT" add -- third_party/HAC
# 另行 add 本次确实修改的适配器、版本清单及文档。
python "$WT/scripts/release_manifest.py" refresh --root "$WT" --output release_manifest.json
git -C "$WT" add -- release_manifest.json
git -C "$WT" diff --cached --submodule=log
git -C "$WT" commit -m "fix(hac): pin validated BUG-001 dependency"
```

不要在父仓库已提交后继续修改子模块却不更新 gitlink。子模块有脏文件、提交缺失或与锁定信息冲突时，主仓库验证不完整，不能合入。

## 5. 对最终候选提交验证

每次改动后先按明确路径 `git add` 本次源码、配置和文档（包括新增文件与删除），然后刷新清单、暂存清单并提交。刷新器不替你暂存或提交；它允许主仓库已跟踪文件有改动，但拒绝可能被遗漏的未忽略未跟踪文件，要求全部递归子模块干净并与父 index 的 gitlink 一致。被 Git 忽略但运行必需的源码也必须显式纳入版本管理，不能依靠忽略规则隐藏。

```bash
cd "$WT"
# 此前已 git add 本次明确的源码/配置/文档路径。
python scripts/release_manifest.py refresh --output release_manifest.json
git add -- release_manifest.json
git diff --cached --stat
git diff --cached --submodule=log
# 核对后提交本次变更，再对提交后的干净工作树验收。
```

清单只记录 Git index 选择的文件和工作区实际字节，排除清单自身，不写当前主仓库 HEAD 或时间戳，避免提交后产生自引用。LFS 大文件的实际权重不计入源码哈希；其 index 指针的 SHA256、对象 ID 和大小单独记录在 `lfs_pointers`。清单不证明权重已经下载或可用，运行所需权重仍须在对应验证中检查。

统一验证器的约定接口如下。`--output` 每次指定一个不存在的目录；具体检查项目、环境和产物以验证器的 `--help` 与结果报告为准。下列三条命令应逐条执行并检查结果，不能只看最后一条的退出码。

```bash
cd "$WT"
python scripts/validate_release.py --help
python scripts/validate_release.py --profile source --runtime configs/runtime.4090.json --output /disk3/ydz/experiments/hps3dgs_validation/BUG-001-source-01
python scripts/validate_release.py --profile cpu --runtime configs/runtime.4090.json --fixtures configs/validation.4090.json --output /disk3/ydz/experiments/hps3dgs_validation/BUG-001-cpu-01
python scripts/validate_release.py --profile gpu --runtime configs/runtime.4090.json --fixtures configs/validation.4090.json --output /disk3/ydz/experiments/hps3dgs_validation/BUG-001-gpu-01
```

主验证器使用配置中记录的各路线环境；不要为了命令启动方便擅自合并环境。上面的 `python` 指可运行验证器的 Python，实际路由运行环境由 `configs/runtime.4090.json` 及工具帮助说明。CPU/GPU fixture 配置为 `configs/validation.4090.json`，上述命令显式传入 `--fixtures`；若省略，先通过验证器帮助核对其默认位置。若接口或配置尚未实现，记录为未完成，不能把本节命令当成已通过的证据。

每次输出的 `report.json` 至少记录 `status`、`profile`、`root_head`、`repositories`、测试结果、命令和退出码。验证器退出 0 表示配置的检查通过，非 0 表示失败或未完成；必须再核对报告覆盖了本次必需项目。缺 fixture 不能算通过。`source`、`cpu`、`gpu` 三个 profile 均要求候选主仓库和递归依赖干净、子模块固定提交一致、清单文件哈希与覆盖范围一致，并在运行后再次核对版本未变化。清单刷新与定位问题可以在开发状态进行，不能代替提交后的正式验收。

CPU profile 使用现有 `unittest` 入口：主仓库测试由 `unittest discover -s tests -v` 收集，SegAny 运行 Geo33 几何和 Fig. 7 packet 测试；Fig. 7 canonical 检查读取 `--fixtures` 文件中的 `canonical_point`。当前 `sacgs` 环境不依赖 pytest。实际执行数量、跳过情况和结果写入本次验证/发布报告，不能只凭进程退出 0 把零用例或跳过当作通过。

按照修改影响选择必要项目；纯文档变更不需要重跑 GPU，修改训练、码流或解码逻辑则需要对应验证。初次整合的正式发布必须覆盖所有保留路线。核对报告是否包含必需项目、是否跳过、输入协议是否一致、指标容差是否预先确定。详见 `docs/VALIDATION_POLICY.md`。

若验收后再改源码、依赖指针或运行参数，要重新验证受影响范围。报告要保存被验收提交号；文档补充提交可以复用前一代码提交的证据，但必须明确最后差异仅为文档，并做来源检查。

## 6. 合并到 main

先用只读门禁核对报告对应当前候选提交和清单。它会检查主仓库及递归子模块干净、报告 HEAD/清单 SHA 一致、依赖提交完整匹配、必需检查通过，不会替你 merge 或 push。

```bash
python "$WT/scripts/check_merge.py" --release-root "$WT" \
  --reports /disk3/ydz/experiments/hps3dgs_validation/BUG-001-source-01/report.json \
            /disk3/ydz/experiments/hps3dgs_validation/BUG-001-cpu-01/report.json \
            /disk3/ydz/experiments/hps3dgs_validation/BUG-001-gpu-01/report.json \
  --require source cpu gpu
```

纯文档修改可按验证策略明确使用 `--require source`，同时在变更报告说明 CPU/GPU 不适用。入口、模型、码流和依赖变更不得用缩小必需 profile 的方式绕过必要验证。门禁非 0 即停止；它检查证据一致性，实际所需路线、格式和指标协议仍按验证策略核对。

合并前检查主线是否已前进。如果分支未包含当前 `main`，先在任务分支整合主线并解决冲突，再对最终结果重新验证受影响范围。

```bash
git -C "$WT" fetch origin
git -C "$WT" log --oneline --decorate -8
git -C "$WT" diff main...HEAD --stat
git -C "$WT" diff main...HEAD --submodule=log
git -C "$WT" status --short
git -C "$WT" merge-base --is-ancestor main HEAD
```

`fetch origin` 仅在仓库已有且可访问的 `origin` 时使用；没有远端时跳过并记录，不能随意添加外部地址。远端存在时还需核对 `origin/main` 是否有本地未知提交，先明确主线关系。`merge-base` 非零表示还不能快进；在任务分支执行 `git merge main`，冲突解决和验证完成后再继续。不要在正式目录里先合并再补测试。

检查变更报告和必要验证通过后，在正式目录快进到刚刚验收的提交：

```bash
git -C "$ROOT" status --short --branch
test "$(git -C "$ROOT" branch --show-current)" = main
test -z "$(git -C "$ROOT" status --porcelain)"
git -C "$ROOT" merge --ff-only "$BRANCH"
git -C "$ROOT" -c protocol.file.allow=always submodule update --init --recursive
git -C "$ROOT" status --short --branch
git -C "$ROOT" submodule status --recursive
git -C "$ROOT" rev-parse HEAD
```

停止条件：正式目录脏、当前分支不对、不能快进、依赖恢复失败、待合入 HEAD 与报告不一致、必需验证不通过。解决前不继续发布。快进保证 `main` 指向已验证的提交；它不取代 diff 检查和验证。推送是独立动作，按现有授权执行，不由上述流程自动触发。

## 7. 发布、备份和恢复核对

先核对发布范围、主仓库与子模块状态、实际报告、引用输入及运行环境。每种独立格式保留代表压缩包的来源和哈希。正式发布还需在新目录恢复源码和所有递归子模块，执行来源检查及至少一次代表包独立解码，排除对旧工作树遗漏源码的隐式依赖。

对每个含本地提交的仓库分别保存 bundle，并保存父仓库。`git bundle verify` 只能验证 bundle 对象/依赖关系，不能代替全新目录中的实际恢复。恢复时如子模块使用本地文件路径，仅对可信本机源按次使用 `git -c protocol.file.allow=always submodule update --init --recursive`，不修改全局 Git 安全设置。

创建标签前确认目标名称不存在并写好包含验证范围和限制的发布说明。以下是命令格式示例；仅在本节条件满足时执行。

```bash
git -C "$ROOT" tag --list 'v0.1.0*'
git -C "$ROOT" tag -a v0.1.0-rc.1 -m "HPS3DGS integration candidate; see recorded validation scope"
# 正式发布条件全部满足后，在已验证的目标提交创建新的正式标签：
# git -C "$ROOT" tag -a v0.1.0 -m "HPS3DGS 0.1.0; validated scope recorded in release report"
git -C "$ROOT" show --stat v0.1.0-rc.1
```

若标签已存在，不重复执行创建，也不使用 `-f` 移动。标签应指向报告对应的提交。候选标签不自动证明所有路线已通过，正式发布说明仍要列明算法结论边界。

历史目录继续保留为只读来源。只有后续明确要清理、源码和递归子模块可恢复、无运行进程且没有被新配置引用时，才制定逐项归档/删除清单；本方案不自动清理。

## 8. 回退与修正

运行恢复和修复主线分开处理。需要立即恢复已验证版本时，在新目录部署已验证标签，不覆盖正在运行或有修改的目录：

```bash
ROLLBACK=/disk3/ydz/code-worktrees/HPS3DGS/rollback-verified-release
test ! -e "$ROLLBACK"
git -C "$ROOT" worktree add --detach "$ROLLBACK" <已验证标签>
git -C "$ROLLBACK" -c protocol.file.allow=always submodule update --init --recursive
git -C "$ROLLBACK" submodule status --recursive
```

核对环境和输入后，使用该目录的入口运行。新输出仍写入新目录，不覆盖旧结果；是否切换已有服务或作业由当前任务授权决定。

修正主线时，从当前主线建立 `fix/` 工作树，用 `git revert <问题提交>` 生成可追溯的撤销提交，或提交针对性修复。若撤销的是父仓库依赖指针，之后在该工作树运行 `git -c protocol.file.allow=always submodule update --init --recursive` 恢复相应版本。按同一验证、检查 diff、快进合并流程发布补丁版本。不要使用 `reset --hard` 改写主线，不删除问题版本标签，不为了恢复旧代码改写历史实验结果。

多个提交共同引入问题时，先确定依赖顺序和撤销范围，不能只撤一半适配器或只回退一半码流格式。无法确定时，先保留复现分支并定位，稳定运行使用已验证版本。
