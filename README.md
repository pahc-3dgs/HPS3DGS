# HPS3DGS

本项目汇集已存在的 PAHC、Geo33、论文 Fig. 7 复现，以及 HAC/HAC++ 便携编解码接口。主体使用 PAHC 的 `src/`、`scripts/`、`configs/`、`tests/` 结构；SegAny、HAC、HAC++ 以固定提交的子模块提供。原算法入口、数据格式和实验协议分别保留。

远端工作目录：`/disk3/ydz/code/HPS3DGS`。完整版本以主仓库提交、递归子模块提交、`release_manifest.json` 及对应验证报告共同确定。原 PAHC 使用说明保存在 `docs/PAHC_ORIGINAL_README.md`。

## 运行入口

`scripts/hps3dgs.py` 使用独立子进程、工作目录和 Python 环境调度已有入口，不混用各仓库的同名模块。

| 路线 | 实现及用途 |
|---|---|
| `pahc` | 原 PAHC 训练、mask/feature 与压缩流程。 |
| `geo33` | 几何准备、四组 QAT、原生完整包解码和参考评价。 |
| `fig7` | Geo32 原始优化、五组 PTQ、独立阈值运行与对应包解码。 |
| `hac` / `hacpp` | 各自的原生便携 codec，范围为 `codec_only`。 |
| `hac-train` / `hacpp-train` | 分别运行已固定的原生训练入口。 |

在服务器上运行，所有数据、模型和输出参数建议使用绝对路径：

```bash
cd /disk3/ydz/code/HPS3DGS
python3 scripts/hps3dgs.py --runtime configs/runtime.4090.json --dry-run geo33 -- --help
python3 scripts/hps3dgs.py --runtime configs/runtime.4090.json geo33 -- --help
python3 scripts/hps3dgs.py --runtime configs/runtime.4090.json fig7 -- --help
python3 scripts/hps3dgs.py --runtime configs/runtime.4090.json hac -- --help
```

`--dry-run` 只显示解析后的命令、环境与缺失路径，不证明导入或模型兼容。首次在其他机器运行时，复制 `configs/runtime.example.json` 并配置实际环境；不要把同一个 Python 环境强用于所有路线。4090 的已安装环境和扩展来源在 `configs/runtime.4090.json`、`provenance/runtime_assets.json` 中记录。

Fig. 7 使用 `--source-manifest configs/fig7_sources.4090.json`（透传参数请用绝对路径）。训练另需显式候选缓存及 SHA256，或 `configs/fig7_candidates.4090.json`。完整说明在 `third_party/SegAnyGAussians/HPS3DGS_FIG7_ENTRYPOINTS.md`。Geo33 紧凑存储包使用 `run_storage_ablation.py decode`，不要交给原生 Geo33 包解码器。

从实验目录纳入的正式脚本位于 `scripts/experiments/`，包括五场景 QAT、Acrimsat 输入对齐、HAC++ 独立解码渲染和包验证。命令及来源见 `docs/EXPERIMENT_ENTRYPOINTS.md`。

## 验证当前提交

先完成明确文件的暂存、清单更新和提交，再验证。每次输出目录必须不存在；原数据和参考结果不会被覆盖。

```bash
git status --short --branch
git submodule status --recursive
python3 scripts/validate_release.py --profile source --output /disk3/ydz/experiments/hps3dgs_validation/source_001
python3 scripts/validate_release.py --profile cpu --runtime configs/runtime.4090.json --fixtures configs/validation.4090.json --output /disk3/ydz/experiments/hps3dgs_validation/cpu_001
python3 scripts/validate_release.py --profile gpu --gpu 0 --runtime configs/runtime.4090.json --fixtures configs/validation.4090.json --output /disk3/ydz/experiments/hps3dgs_validation/gpu_001
```

`source` 检查干净提交、递归子模块与清单完整性；`cpu` 执行已有几何/codec 契约和 Fig. 7 测试；`gpu` 将已有包复制到不含训练模型及旁置记录的新目录，独立解码并比较指标、图像数量和物理字节，再运行非空实例合成测试。默认选择的 8 个包覆盖 Geo33 原生/存储、Fig. 7、HAC、HAC++ 及 UE/COLMAP 输入，合计 1200 个同训练视角重建帧；合成实例测试另外记录，不当作真实几何压缩效果。

结果写入各输出目录的 `report.json`，退出码非零、缺 fixture 或未执行均不能视为通过。GPU 检查使用共享 GPU 锁及空闲检查，不自动占用已有任务的 GPU。

## 修复、合并和发布

完整流程见 `docs/VERSIONING.md` 与 `docs/VALIDATION_POLICY.md`。日常使用：从已验证基线建立 `fix/`、`feature/` 或 `repro/` 隔离工作树；子模块先提交；父仓库固定依赖；刷新清单；验证最终候选；通过合并检查后快进 `main`。

```bash
# 在候选工作树内，只暂存本次明确修改的文件与依赖指针。
git add -- scripts/changed_file.py third_party/SegAnyGAussians
python3 scripts/release_manifest.py refresh --output release_manifest.json
git add -- release_manifest.json
git commit -m "Fix the concrete issue and preserve existing codec behavior"

# 使用同一候选提交的三份报告；本命令只检查，不执行 merge。
python3 scripts/check_merge.py --reports /path/source/report.json /path/cpu/report.json /path/gpu/report.json --require source cpu gpu
```

上述 `changed_file.py` 和报告路径是示例，须替换为实际内容。发布标签不可移动；回退通过旧标签的新工作树或经验证的 `git revert` 完成。不要直接重置共享主线或清理未知工作树。

## 来源、运行资源与功能边界

- `provenance/` 记录导入前的 Git 状态、逐文件来源/哈希、依赖来源及运行资源。
- `/disk3/ydz/code/HPS3DGS-repositories` 保存服务器本地 bare Git 仓库，提供可恢复的提交对象；这些目录不是额外的实验工作副本。原始来源工作树继续保留。
- SAM/CLIP 权重与训练扩展的稳定副本在 `/disk3/ydz/pahc_hacpp_runtime/hps3dgs_weights` 和 `hps3dgs_extensions`。权重、环境、数据和实验结果属于外部运行输入，普通 Git clone 不复制它们。Git LFS 模型 metadata 随子模块固定；恢复完整训练环境时还须按 `provenance/runtime_assets.json` 核对或恢复模型载荷。
- Geo33 已有真实五场景实例共享结果为 0；合成实例测试验证软件通路。HAC/HAC++ 新接口属于 `codec_only`，没有完成训练期间 owner 传播。整合不改变这些结论。
- 评价为 150 train / 同 150 reconstruction views，不是 held-out。不得把包可移植性解释为算法泛化或模板共享收益。
