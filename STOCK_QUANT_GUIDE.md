# TSLib 股票量化评估指南

本指南说明如何在本仓库中完成股票数据适配、模型训练/回测与超参数调优。流程包含两个脚本：  
- `scripts/stock/stock_benchmark.py`：训练 + 回测  
- `scripts/stock/stock_tune.py`：超参数调优（Optuna/TPE）

两者可独立运行、互不影响。

## 数据要求

输入文件：`stock_data.parquet`（放在仓库根目录或指定 `--root_path/--data_path`）

必须字段：
- `date` 或 `data`：交易日（格式 `YYYYMMDD`）
- `code`：股票代码（如 `000543.SZ`）
- `open`：开盘价
- 其他可用字段（如存在会自动作为特征）：  
  `high` `low` `close` `volume` `amount` `settle` `openInterest` `preClose` `suspendFlag`

可选字段：
- `time`：时间字段（若存在且非全 0，会与 `date` 合并为时间戳）

## 目标定义（避免泄露）

目标是预测“下下个交易日开盘价相比下个交易日开盘价上涨比例”：

```
return_t = open_{t+1} / open_t - 1
```

训练使用 `pred_len=2`，并取预测序列的第 2 步（对应 `open_{t+2} / open_{t+1} - 1`）。  
输入仅使用历史窗口，且缩放仅拟合训练年份，避免未来泄露。

## 数据切分

按交易日年份切分样本（以“交易日”为准）：
- 训练：2014–2023
- 测试：2024
- 验证：2025

注意：2025 样本允许使用 2024 历史作为输入，这是合理的“历史可用信息”，不视为泄露。

## 训练与回测（对比评估）

默认模型列表：
```
TimeXer, iTransformer, PatchTST, TSMixer, DLinear, TimeMixer, FEDformer, Chronos2, TiRex, TimeMoE
```

运行示例：
```bash
python scripts/stock/stock_benchmark.py   --use_wandb --wandb_project tslib --resume
```

指定模型：
```bash
python scripts/stock/stock_benchmark.py \
  --models TimeXer,iTransformer,PatchTST
```

可覆盖超参（示例）：
```bash
python scripts/stock/stock_benchmark.py \
  --model TimeXer --d_model 256 --n_heads 8 --e_layers 3 \
  --d_ff 512 --dropout 0.1 --learning_rate 0.0001 \
  --patch_len 16 --stride 8
```

输出目录结构：
```
stock_results/<Model>/<split>/
  predictions.csv
  equity_curve.csv
  metrics.json
```

## 回测规则

- 每日按预测收益排序，取 `topk=1`
- 交易顺序：开盘先卖后买
- 交易成本：佣金 0.03%，印花税 0.1%
- 指标：最终资金、累计收益率、年化收益率、最大回撤、夏普、胜率、盈亏比、总交易次数、交易天数

## 超参数调优（Optuna / TPE）

调优脚本采用 TPE 采样，不使用网格搜索。  
目标默认最大化 `annualized_return_pct`（可选 `cumulative_return_pct/sharpe/final_capital`）。

运行示例：
```bash
python scripts/stock/stock_tune.py \
  --max_trials 30 --tune_epochs 5 --objective annualized_return_pct
```

仅调优可训练模型（默认不含 zero-shot）：
```bash
python scripts/stock/stock_tune.py --models TimeXer,PatchTST,DLinear
```

包含 zero-shot 模型：
```bash
python scripts/stock/stock_tune.py --include_zero_shot
```

输出目录结构：
```
stock_tuning/<Model>/
  best_params.json
  study.csv
```

## 一键调优后回测流水线

如果你希望“调优完成后直接做完整回测”，使用：

```bash
python scripts/stock/stock_tune.py \
  --run_benchmark_after_tune --benchmark_tag tuned
```

该流程会在每个模型调优结束后，自动读取最佳参数并重新训练（使用 `train_epochs`）然后分别回测 2024/2025：

```
stock_results/<Model>/tuned/<split>/
  predictions.csv
  equity_curve.csv
  metrics.json
```

### 默认搜索范围（摘要）

- TimeXer: `d_model{128,256,512}`, `n_heads{4,8}`, `e_layers{2,3}`, `d_ff{256,512,1024}`,
  `dropout[0.05,0.3]`, `lr[1e-5,5e-4]`, `patch_len{8,16,32}`, `use_norm{0,1}`
- iTransformer: `d_model{128,256,512}`, `n_heads{4,8}`, `e_layers{2,3}`, `d_ff{256,512,1024}`,
  `dropout[0.05,0.3]`, `lr[1e-5,5e-4]`
- PatchTST: `d_model{128,256,512}`, `n_heads{4,8}`, `e_layers{2,3}`, `d_ff{256,512,1024}`,
  `dropout[0.05,0.3]`, `lr[1e-5,5e-4]`, `patch_len{8,16,32}`, `stride{4,8,16}`(<=patch_len)
- TSMixer: `d_model{64,128,256}`, `e_layers{2,4,6}`, `dropout[0.05,0.3]`, `lr[1e-4,5e-3]`
- DLinear: `moving_avg{7,25,49}`, `individual{False,True}`, `lr[1e-4,5e-3]`
- TimeMixer: `d_model{128,256,512}`, `n_heads{4,8}`, `e_layers{2,3}`, `d_ff{256,512,1024}`,
  `dropout[0.05,0.3]`, `lr[1e-5,5e-4]`, `moving_avg{7,25}`, `down_sampling_layers{0,1,2}`
- FEDformer: `d_model{128,256}`, `n_heads{4,8}`, `e_layers{2,3}`, `d_layers{1,2}`,
  `d_ff{256,512}`, `dropout[0.05,0.3]`, `lr[1e-5,5e-4]`, `moving_avg{7,25}`
- Zero-shot（Chronos2/TiRex/TimeMoE）:
  `seq_len{32,64,96,128}`, `batch_size{8,16,32}`

## wandb 追踪

开启：
```bash
--use_wandb --wandb_project <project> --wandb_entity <entity>
```

记录内容：
- 训练：`train/loss`, `val/loss`, `test/loss`
- 调优：`tune/<objective>`, `tune/trial`
- 回测：`test/*`、`val/*` 指标

## 依赖

```
pip install -r requirements.txt
```

新增依赖包含：
- `pyarrow`（parquet 读取）
- `wandb`（日志追踪）
- `optuna`（超参数调优）

## 常见注意事项

- `pred_len` 必须 >= 2，否则无法构造“下下个交易日/下个交易日”目标。
- `suspendFlag=1` 的样本会被过滤，不参与回测交易。
- 若你希望使用调优结果跑全量回测，请将 `best_params.json` 中的参数手动填入 benchmark 脚本参数。

## 预处理缓存（加速）

股票数据预处理会自动缓存到 `./cache/`，包含已处理的序列、索引和缩放器。  
同一数据文件和配置再次运行会直接命中缓存，显著加速。

关闭缓存：
```bash
python scripts/stock/stock_benchmark.py --disable_stock_cache
```

缓存位置可配置：
```bash
python scripts/stock/stock_benchmark.py --stock_cache_dir /path/to/cache
```
