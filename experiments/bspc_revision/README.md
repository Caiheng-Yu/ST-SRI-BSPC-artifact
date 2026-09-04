# BSPC 修订实验入口

本目录实现与旧实验隔离的 BSPC 修订流程。它不会修改 `data/` 中的 `.mat` 或 `.npy`，也不会覆盖 `checkpoints_2000hz/`。

## 当前入口

`leakage_free_db2.py` 完成以下工作：

- 按每类 6 次动作重复，在窗口化前执行训练 4、验证 1、审计 1 的固定种子划分；
- 在重复段边界设置原始采样隔离区；
- 仅使用训练重复段计算归一化均值和标准差；
- 检查训练、验证和审计窗口是否共享原始采样；
- 保存重复段清单、窗口清单、类别统计、新检查点和独立审计指标。
- 使用逐受试者结果原子写入、检查点哈希和 `--resume` 支持可靠续跑。

`onset_protocol.py` 和 `audit_onset_protocol.py` 实现修正后的 `restimulus` 标签起点协议。零点不是 sEMG 生理起点或机械起点；严格 onset 审计和普通分类训练使用两个独立门控。

## R001：只生成 40 人数据清单

```powershell
python experiments/bspc_revision/leakage_free_db2.py `
  --all-subjects `
  --manifest-only `
  --output-dir results/bspc_revision_v2/r001_split_manifest
```

## R002：S1 冒烟训练

```powershell
python experiments/bspc_revision/leakage_free_db2.py `
  --subjects 1 `
  --epochs 3 `
  --patience 3 `
  --batch-size 64 `
  --max-train-windows 2048 `
  --max-validation-windows 512 `
  --max-audit-windows 512 `
  --output-dir results/bspc_revision_v2/r002_smoke_s1 `
  --checkpoint-dir checkpoints_bspc_v2/r002_smoke_s1
```

冒烟结果只验证流程和 GPU 训练链，不作为论文结果。

## R004 门控结论

- 普通分类训练门：PASS，可启动 R005。
- 严格 onset ST-SRI 主结论门：FAIL，0/50/100/150 ms 只可作为诊断。
- 模型 `argmax` 表示实际决策；固定动作标签的目标概率只表示信息可用性，不能代替实际决策。

## R005：40 人持久训练

R005 已完成 40/40 人训练，但群体门未通过：平均 accuracy 为 0.6444，macro-F1 为 0.4851，仅 15/40 人达到预设的 0.65/0.50 门槛。S10、S15、S17、S27 出现以静息预测为主的训练塌缩，因此 R005 检查点只作为未平衡基线保留。

```powershell
wsl.exe -d Ubuntu -- tmux capture-pane -pt st-sri-r005:train.0 -S -80
Get-Content -Tail 80 logs/bspc_revision_v2/r005_full_s1_s40.log
Get-Content -Raw results/bspc_revision_v2/r005_full_s1_s40/run_status.json
```

正式输出写入：

- `results/bspc_revision_v2/r005_full_s1_s40/`
- `checkpoints_bspc_v2/r005_full_s1_s40/`
- `logs/bspc_revision_v2/r005_full_s1_s40.log`

## R005a：多种子平衡训练诊断与自动门控

R005a 不修改数据划分和原始样本，只调整训练策略：

- 每名受试者使用由训练种子和受试者编号共同确定的独立种子，确保断点续跑不改变初始化；
- 使用类别频数的平方根逆权重，降低静息类对损失的支配；
- 至少训练 30 轮，避免多数类基线在第 1 轮触发早停；
- 使用验证集 macro-F1 选择检查点，同时记录 accuracy 和 balanced accuracy；
- 对 S1、S10、S15、S17、S27、S40 运行 3 个训练种子。

`run_r005a_supervisor.ps1` 在诊断完成后调用 `evaluate_r005a_gate.py`。只有四名问题受试者恢复、种子范围受控且 S1/S40 不明显退化，才自动启动 R005b 全 40 人训练。

```powershell
wsl.exe -d Ubuntu -- tmux capture-pane -pt st-sri-r005a:train.0 -S -80
Get-Content -Tail 80 logs/bspc_revision_v2/r005a_supervisor.log
Get-Content -Raw results/bspc_revision_v2/r005a_balanced_diagnostic/supervisor_status.json
```

诊断和门控输出：

- `results/bspc_revision_v2/r005a_balanced_diagnostic/`
- `checkpoints_bspc_v2/r005a_balanced_diagnostic/`
- `logs/bspc_revision_v2/r005a_balanced_seed_*.log`

门控通过后的全量输出：

- `results/bspc_revision_v2/r005b_balanced_full_seed20260815/`
- `checkpoints_bspc_v2/r005b_balanced_full_seed20260815/`

## R009：统一相位窗口清单（纯 CPU，已生成）

`transition_windows.py` 把三类时间对照窗口统一为 `PhaseWindowRecord`：

- `onset`：静息到运动转换（复用已审计的 onset 协议），相位 0/50/100/150 ms；
- `offset`：运动到静息反向转换，参考点为活动段最后一个采样点，相位与 onset 镜像；
- `steady`：稳态运动窗口，参考点在活动段内部固定相对位置（默认中点）。

40 人 × 17 个审计重复 × 9 个窗口 = 6120 条记录已全部通过完整性校验（坐标唯一、因果窗 600 点、当前块 10 点、不跨重复段支持、offset 参考点确为 active→0 转换、steady 参考点在活动段内部 953–3291 ms）。

`collect_st_sri_curves.py` 的 `--records-csv` 模式可直接按该清单采集曲线（`--window-kinds` 筛选类型）；`stratified_st_sri_report.py` 在检测器判决产出后按窗口类型×相位、真实类别、预测类别、正确/错误和目标概率四分位分层汇总支持率。`train_label_shuffle.py` 是标签打乱阴性对照训练脚本（协议与 R005a/R005b 一致，仅置乱训练窗标签），需 GPU，排在 R005b 之后。

```powershell
python experiments/bspc_revision/stratified_st_sri_report.py `
  --curves results/bspc_revision_v2/r008_curves/S01_trained.npz `
  --report results/bspc_revision_v2/r008_detector_report.json `
  --output results/bspc_revision_v2/r009_stratified_report.json
```

```powershell
python experiments/bspc_revision/transition_windows.py
```

输出：

- `results/bspc_revision_v2/r009_window_manifest/phase_windows.csv`
- `results/bspc_revision_v2/r009_window_manifest/window_manifest_integrity.json`

## R006/R007：零分布校准与可拒绝峰值检测器（纯 CPU）

`st_sri_detector.py` 实现检测链，只依赖 numpy/scipy，不占用 GPU：

- 先移除 |lag| 小于块宽（10 点 = 5 ms）的重叠滞后分箱，再对保留分箱平滑，修复旧 `detect_st_sri_peak_ms` 先平滑后排除的顺序问题；
- 在有资格区域计算连续正质量、有符号质量、峰值显著度和峰值位置；
- 支持阈值只由零分布的 (1-alpha) 分位数标定：零模型曲线（`--null-curves`，R006 主来源）或与真实处理链一致的置乱代理（`--surrogate-count`，解析对照）；
- 门控统计量未超过阈值时明确报告“无支持峰”，并汇总支持率和无支持峰比例。

输入为滞后剖面 npz（`lags_ms` + `curves` [+ `ids`]）。曲线采集由 `collect_st_sri_curves.py` 完成：对 onset 协议审计窗口扫描训练检查点、完全重初始化和张量内参数置乱三类变体，默认强制 CPU 并限线程，可与 GPU 训练并行；标签打乱训练变体需单独 GPU 训练任务。单元测试见 `test_st_sri_detector.py`（20 项）和 `test_collect_st_sri_curves.py`（11 项）。

```powershell
python experiments/bspc_revision/collect_st_sri_curves.py `
  --subjects 1 `
  --checkpoint-dir checkpoints_bspc_v2/r005b_balanced_full_seed20260815 `
  --output-dir results/bspc_revision_v2/r006_curves/r005b `
  --device cpu --threads 2
```

```powershell
python experiments/bspc_revision/st_sri_detector.py `
  --curves results/bspc_revision_v2/r008_curves/trained.npz `
  --null-curves results/bspc_revision_v2/r006_curves/null_models.npz `
  --output results/bspc_revision_v2/r006_detector_report.json
```

## R016：合成交互验证（已完成）

`a1_synthetic_detector_validation.py` 在当前 10-sample block、独立负对照零分布和
可拒绝峰值检测链上验证七种已知机制。每种机制 40 个窗口；独立 160 个加性阴性
窗口用于假阳性检查。结果为 120/120 个交互窗口在 ±5 ms 内恢复，160/160 个
阴性窗口无支持峰，假阳性率为 0%。

```powershell
python experiments/bspc_revision/a1_synthetic_detector_validation.py `
  --output-dir results/bspc_revision_v2/a1_synthetic_detector_validation
```

## R017：DB2 当前检查点回放与条件谱原点（已完成）

`r017_db2_fixed_replay.py` 读取 R005b 的 40 个固定检查点已经生成的 onset 曲线，
验证当前 4/1/1 审计集、训练-only 归一化、10-sample block 和四个当前位置，
并汇总 phase-0 与四相位协议的受试者级谱。当前重建不是原稿 839 窗口的复制：
它使用 40 人 × 17 个审计重复 × 4 相位 = 2720 条曲线。

条件谱原点诊断把每个受试者均值谱旋转 300 个保存的 lag 原点并包含 identity，
再使用相同的 5 ms 下界和首个最大值规则。该结果明确标记为条件旋转诊断，
不能解释为信号生成零分布、总体推断或生理起点证据。

```powershell
python experiments/bspc_revision/r017_db2_fixed_replay.py
```

输出：

- `results/bspc_revision_v2/r017_db2_fixed_replay/summary.json`
- `results/bspc_revision_v2/r017_db2_fixed_replay/subject_protocol_summary.csv`
- `results/bspc_revision_v2/r017_db2_fixed_replay/conditional_origin_detail.csv`

## R018：真实方法比较（代码就绪，未完成全量运行）

`comparison_methods_scan.py` 在当前检查点、当前训练分区静息背景和统一审计清单上
比较直接析因遮挡与 TimeSHAP；Dynamask 仅在依赖可用时运行，未安装时保留明确的
`not_installed` 状态。原稿对应的是 40 个共同支持窗口，因此当前重建使用每名
受试者一个 onset phase-0 窗口，并保留正确/错误决策、AOPC、共同时间支持和运行时。

```powershell
python experiments/bspc_revision/comparison_methods_scan.py `
  --all-subjects `
  --data-root data/DB2 `
  --records-csv results/bspc_revision_v2/r009_window_manifest/phase_windows.csv `
  --window-kinds onset `
  --max-records 1 `
  --methods occlusion timeshap `
  --device cuda `
  --output-dir results/bspc_revision_v2/r018_comparison_methods
```

## R019：参数置乱、检测器敏感性与分层（预声明三种子完成）

R019 必须使用当前 onset 曲线、当前零分布阈值和 participant-cluster 汇总，
覆盖 within-tensor permutation 的预声明 seeds `101/202/303`、峰搜索下界/平滑
敏感性，以及真实类别、预测类别、正确性和置信度分层。历史原稿的 40/40 置乱
短峰和旧 26/40 计数不属于 R019 输出。预声明的 `101/202/303` 均已完成：
每个种子覆盖 40 名受试者、2720 条曲线，共 8160 条参数置乱曲线。5 ms 下界、
sigma=2 时，训练曲线为 36/40 短滞后、22/40 恰在下界，置乱三个种子均为
40/40 短滞后且 40/40 恰在下界；这显示原始峰位置会被搜索下界吸引，不能作为
生理时序、模型特异性或检测器支持证据。相应的置乱正谱质量中位数为
0.003645 +/- 0.000316，而训练曲线为 1.90675，约低 523 倍；仅此量级差异可作
描述性伪证诊断。`summary.json` 记录了全部种子、完整检测器网格及分层结果。
