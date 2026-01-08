# 股票滚动训练 + 回测全流程（`stock_rolling_backtest.py`）

> 适用范围：本仓库的滚动回测脚本 `scripts/stock/stock_rolling_backtest.py`。  
> 目标：用**尽量通俗**的方式，把「从数据 → 构造样本 → 训练 → 预测 → 回测 → 输出结果」的整条链路讲清楚，并标出常见“数据泄露/收益偏高”的坑。

---

## 0. 一张图看懂流水线

```
stock_data.parquet
   │
   ├─(Dataset_Stock / Dataset_StockPacked)  读取 + 目标计算 + 归一化 + 生成样本索引
   │        │
   │        ├─ train split  →  Exp_Long_Term_Forecast.train() 训练（早停看 val）
   │        └─ test/val     →  predict_dataset() 逐样本产出 pred_return/true_return
   │
   └─ backtest_topk_detailed()  按天选 TopK → 计算净值曲线与指标 → 写入结果文件
```

---

## 1. 你到底在运行什么？

核心脚本：
- 滚动回测：`scripts/stock/stock_rolling_backtest.py`

关键依赖代码：
- 数据集与预处理：`data_provider/data_loader.py`（`Dataset_Stock`, `Dataset_StockPacked`）
- 数据入口：`data_provider/data_factory.py`
- 训练逻辑：`exp/exp_long_term_forecasting.py`（`Exp_Long_Term_Forecast`）
- 回测逻辑：`scripts/stock/stock_rolling_backtest.py`（`backtest_topk_detailed`）

---

## 2. 数据文件：`stock_data.parquet`

### 2.1 必须字段（以代码为准）

`Dataset_Stock` / `Dataset_StockPacked` 在构建 base cache 时，**严格要求**（见 `data_provider/data_loader.py` 的 `_build_base_cache`）：

- `code`：股票代码（字符串）
- `time`：时间戳（**毫秒**，epoch ms；脚本用 `pd.to_datetime(..., unit="ms")` 解析）
- `open`：开盘价（数值）

如果 `--target` 选择的是默认的 `lag_return_cs_rank` / `lag_return_rank` / `lag_return`，还需要：
- `close`
- `preClose`

建议字段（存在就会作为特征使用的一部分）：
- `high`, `low`, `volume`, `amount`, `settle`, `openInterest`, `suspendFlag`（停牌标记）

### 2.2 快速自检：你的 parquet 有没有这些列？

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("stock_data.parquet", engine="pyarrow")
print("columns:", sorted(df.columns))
print(df.head(3))
PY
```

### 2.3 `time` 的含义（非常重要）

- 这里的 `time` 不是 `YYYYMMDD`，而是 **ms 时间戳**。
- 你的数据可能在每天收盘时间（例如 16:00）落一个 timestamp，这没问题；**只要同一天每只股票一致**即可。
- 所有“按天”的逻辑（滚动切分/选股/回测）最终都依赖这个 `time` 转成的 `datetime`。

---

## 3. 目标（label）到底是什么？`--target` 怎么影响训练/回测？

默认参数里：`--target lag_return_cs_rank`，并且 `--features MS`。

### 3.1 `lag_return` 的计算方式（“链式”开盘收益）

当 target 属于 `lag_return / lag_return_rank / lag_return_cs_rank` 时，代码会先计算当日 `lag_return`（见 `data_provider/data_loader.py`）：

设：
- `prev_open` = 前一交易日开盘价
- `prev_close` = 前一交易日收盘价
- `open` = 当日开盘价
- `preClose` = 当日昨收（通常等于前收盘，可能含复权/调整差异）

则：

```
intraday  = prev_close / prev_open
overnight = open / preClose
lag_return = intraday * overnight - 1
```

并且会做质量过滤（示意）：
- `suspendFlag==0` 才算有效
- `abs(lag_return) <= STOCK_RETURN_LIMIT` 才算有效（默认 `STOCK_RETURN_LIMIT=0.30`）

### 3.2 `lag_return_cs_rank` / `lag_return_rank`

当 target 不是 `lag_return` 时，代码把当日 `lag_return` 做**横截面 rank（百分位）**：

```
rank_t(code) = rank_pct( lag_return_t(code) | 同一天所有 code )
```

这意味着：
- **训练学的是“未来某天的 rank”**（更偏排序学习）
- 但回测最终需要一个“真实收益率 true_return”去算资金曲线

### 3.3 为什么回测的 `true_return` 可能不是训练的 label？

在 `scripts/stock/stock_rolling_backtest.py` 里：
- 如果 `target ∈ {lag_return_rank, lag_return_cs_rank}`，回测用的 `true_return` 会**强制用开盘价计算**：

```
true_return = open_exit / open_trade - 1
```

这属于“用 rank 训练、用真实开盘收益回测”的设计：模型输出的 `pred_return` 其实更像“打分/排序分数”。

---

## 4. 样本是怎么构造的？（最核心：避免把未来塞进特征）

几个关键参数：
- `seq_len`：输入历史窗口长度
- `label_len`：decoder 的起始 token 长度（很多模型需要）
- `pred_len`：预测未来多少步（本脚本要求 `pred_len>=2`）
- `trade_horizon`：回测用第几步做交易（默认 2）

### 4.1 单股票模式：`Dataset_Stock`

对每只股票，数据是一条按时间排序的序列。一个样本大致是：

```
end_idx = t
seq_x   = [t-seq_len+1 ... t]          # 只到 t（历史）
seq_y   = [t-label_len+1 ... t+pred_len]  # 包含未来 pred_len（用于训练监督）

trade_date = t+1
pred_date  = t+pred_len
```

**关键点**：特征 `seq_x` 只到 `end_idx`，不会包含 `trade_date` 之后的数据；因此不存在“把目标日的真实值喂进特征”这种硬泄露。

### 4.2 打包模式（默认）：`Dataset_StockPacked`

默认 `--stock_pack` 为 True，使用 `Dataset_StockPacked`：

- 它把多个股票拼成一个“大张量”，让模型学横截面信息。
- 输入形状（概念上）：
  - `data[t, feature_group, code]` 被拉平成 `data[t, feature_group*code]`
  - 最后一组 feature_group 固定是 target，因此 `target_slice` 可直接切出来作为监督/预测对象

样本索引也是同样逻辑：

```
end_idx = t
seq_x   = [t-seq_len+1 ... t]
trade_date = t+1
exit_date  = t+trade_horizon
```

#### 4.2.1 `pack` 的完整构建过程（逐步拆解）

`Dataset_StockPacked` 的核心工作在 `data_provider/data_loader.py` 的 `Dataset_StockPacked.__read_data__()`：它会把 `N` 只股票在 `[pack_start, pack_end]` 期间的数据对齐到同一套交易日历，并拼成一个巨大的矩阵。

下面按真实代码的顺序解释（省略了一些异常处理/缓存细节）：

**Step 0：读取 / 复用 base cache（按股票分组的原始序列）**

- 首次运行会从 `stock_data.parquet` 读出 `code/time/open/.../target`，并按 `code` 分组，得到：
  - `payload['data']`: 每只股票二维数组 `[T_code, n_base_columns]`
  - `payload['dates']`: 对应的时间戳（已转成 datetime）
  - `payload['suspend']`: 对应的 `suspendFlag`
- 后续会命中 `./cache/stock_base_*.pkl` 加速。

**Step 1：确定 pack 的特征组（feature groups）**

由参数 `--features` 决定（注意：这里的 features 是 Time-Series-Library 的通用参数，沿用到股票场景）：

- `features='S'`：只 pack `target`（1 组）
- `features!='S'`：pack 「基础特征 + target」
  - `pack_feature_cols = [open, high, low, close, volume, amount, settle, openInterest, preClose, suspendFlag, ..., target]`
  - 并且代码保证 **target 永远在最后一组**，因此可以得到：

```
target_slice = ((n_groups - 1) * n_codes, n_groups * n_codes)
```

这个 `target_slice` 会被训练与预测专门拿来切出目标部分（见 `exp/exp_long_term_forecasting.py` 的 `_slice_for_stock_pack`）。

**Step 2：确定 `pack_start` / `pack_end`（pack 的整体日历范围）**

`Dataset_StockPacked` 自身要求你提供：
- `--stock_pack_start`
- `--stock_pack_end`

但在滚动回测脚本里（`scripts/stock/stock_rolling_backtest.py`），如果你不显式传，会在每个窗口自动推导：

- `pack_start = train_start` 往前再挪 `seq_len` 个交易日（为了构造最早的输入窗口）
- `pack_end = max(test_end, val_end)` 往后挪 `pred_len-1 + extra_td` 个交易日（确保未来开盘价/标签可用）

**Step 3：选股票池（universe selection）**

pack 模式的关键假设：**股票池是固定的**（整个 pack 范围内都用同一批 code）。

筛选逻辑（概念上）：

1. code 必须在 `pack_start` 之前就已经有数据（保证最早日期也能对齐）
2. code 必须至少覆盖到 `coverage_end`（避免“只选未来还活着的股票”）

其中 `coverage_end` 由 `--stock_pack_select_end` 决定：

- `train_end`（默认，推荐）：`coverage_end = train_end`，避免用未来信息选股票池
- `pack_end`（不推荐）：`coverage_end = pack_end`，会引入存活者偏差（代码会 warn）
- `none`：不要求覆盖到某个结束日期（允许中途退市/缺失；退市后在张量中会被当作缺失/不可交易）
- `YYYY-MM-DD`：用你给的日期作为覆盖终点

`--stock_universe_size` 可以限制最多选多少只股票（0 表示全选满足条件的 code）。

**Step 4：拟合缩放器（StandardScaler），只用训练期**

如果 `--scale True`（默认）：
- 会对每只股票抽取 `train_start..train_end` 的数据（且只取 pack_feature_cols 这些列）
- 用这些训练数据 `partial_fit` 出一个 `StandardScaler`
- **重要**：scaler 的均值/方差是按“特征组”统计的，长度是 `n_groups`，不是 `n_groups*n_codes`
  - 这意味着：同一特征组（比如 open）对所有股票用同一套均值方差做标准化

**Step 5：构建“union 交易日历”（关键设计点）**

pack 不能用“交集日历”（intersection），因为只要某只股票缺一天，交集就会缩得很短，样本会变少。

所以代码用 **并集日历（union）**：
- 对每只选中的股票，取它在 `[pack_start, pack_end]` 的 dates
- 把这些 dates 做 `union1d`，得到 `common_dates`

结果：
- `common_dates` 会包含“有些股票缺失”的交易日
- 缺失位置后面会用 `fill_value`（默认 0.0）填充，并把 `suspend_mat` 留成 1（不可交易）

**Step 6：逐股票对齐并写入大矩阵**

最终生成的核心张量（以 float32 为主）：

- `data_mat`: shape = `[t_len, n_groups * n_codes]`
- `open_mat`: shape = `[t_len, n_codes]`（用于回测计算真实开盘收益）
- `suspend_mat`: shape = `[t_len, n_codes]`（用于回测过滤停牌/缺失）

其中 `data_mat` 的布局是“先按特征组，再按股票”：

```
# 第 f 组特征（例如 open）对应的列区间：
cols[f] = [f*n_codes, (f+1)*n_codes)

data_mat[t, f*n_codes + j] = feature_value(t, code_j, feature_group_f)
```

当某只股票在某天没有数据时：
- 对应 `data_mat` 会保持 `fill_value`
- 对应 `suspend_mat` 会保持 1（默认初始化就是 1）
- 对应 `open_mat` 会保持 0

**Step 7：生成时间特征 `stamp`**

与普通数据集一致：
- `timeenc=0`：`month/day/weekday`
- `timeenc=1`：`time_features(...)`

得到：
- `stamp`: shape = `[t_len, stamp_dim]`

**Step 8：生成样本索引（用于 DataLoader 取样）**

pack 的一个样本是“某个 end_idx 对应的全市场截面输入”，因此索引只需要 `end_idx`（不带 code）：

```
end_idx ∈ [seq_len-1, t_len-pred_len-1]
trade_idx = end_idx + 1         # trade_date
pred_idx  = end_idx + pred_len  # pred_date（严格模式用它限制 split）
```

并按 split（train/test/val）过滤：
- 先过滤 `trade_date` 落在该 split 的日期范围内
- 若 `stock_strict_pred_end=True`，再要求 `pred_date` 也落在 split 内（防止跨 split 标签泄露）

最后 `__getitem__` 返回：
- `seq_x = data_mat[s_begin:s_end]`
- `seq_y = data_mat[r_begin:r_end]`
- `seq_x_mark/seq_y_mark = stamp[...]`

#### 4.2.2 pack 模式下“训练 / 预测 / 回测”是怎么衔接的？

**训练（loss 只看 target block）**

因为 `data_mat` 里包含了多组特征 + target，所以模型输出维度也是 `c_out = n_groups*n_codes`。

训练时不会对所有输出做监督，而是用 `target_slice` 只切出最后一块（target）算 loss：
- 这就是 `Exp_Long_Term_Forecast._slice_for_stock_pack()` 做的事。

**预测（得到每个 trade_date 的全股票打分）**

预测时同样先切 `target_slice`，得到：
- `pred_step`: shape `[B, n_codes]`（每个样本、每只股票一个预测分数）
- `true_return`: 取决于 `--target` 是否是 rank 类：
  - rank 类：用 `open_mat` 计算开盘收益（更贴近交易）
  - 非 rank：直接用 label 中的 target 值

**回测（按天选 TopK）**

回测阶段按 `trade_date` 分组，对每一天：
- 按 `pred_return` 选 TopK 股票
- 用 `true_return` 算资金曲线

### 4.3 `stock_strict_pred_end`：阻止跨 split 的标签泄露

`stock_strict_pred_end=True`（默认）时，数据集会要求：
- **trade_date 在 split 内**
- 并且 **pred_date（也就是 t+pred_len）也在 split 内**

这能避免一种常见泄露：训练集的样本 trade_date 还在训练期，但 `pred_date` 已经跑到验证/测试期。

---

## 5. 滚动窗口怎么切？（Rolling windows）

滚动窗口由 `scripts/stock/stock_rolling_backtest.py::_build_windows()` 生成：
- `start_date` / `end_date`：整个回测的覆盖范围
- `train_years`：训练窗长度（年）
- `val_months` / `test_months`：验证/测试窗长度（月）
- `step_months`：每次向前滚动多少（月）
- `window_order`：
  - 推荐 `train_val_test`（默认）：先训练 → 再验证（早停）→ 再测试
  - `train_test_val`：**不推荐**（早停会用到 test 后面的 val，属于“看未来”）

每个窗口都会产出一个 `window_tag`，并把结果落盘到：

```
stock_results_rolling/<Model>/<window_tag>/{test,val}/
```

---

## 6. 训练阶段：`Exp_Long_Term_Forecast.train()`

训练入口：
- `scripts/stock/stock_rolling_backtest.py` 会实例化 `Exp_Long_Term_Forecast(args)` 并调用 `exp.train(setting)`

训练过程（高层理解）：
1. `data_provider(args, 'train')` 取训练数据（train loader 会 shuffle）
2. `data_provider(args, 'val')` 取验证数据（val loader 不 shuffle）
3. 每个 epoch：
   - 前向预测 `pred_len` 步
   - 计算 loss（默认 `HYBRID_WIC`：IC 类 loss + CCC 类 loss 的混合）
   - 用验证集 loss 做 early stopping

> 注意：`Exp_Long_Term_Forecast.train()` 里每个 epoch 还会额外计算一次 `test_loss` 并打印出来。  
> 这本身不一定造成“代码级泄露”（因为 early stopping 用的是 val），但会造成**人为调参泄露**：如果你盯着 test_loss 调参数，相当于把测试集信息用进了策略/超参选择。

---

## 7. 预测阶段：`predict_dataset()`

预测入口：
- `scripts/stock/stock_rolling_backtest.py::predict_dataset(exp, data_set, data_loader, args)`

它会对每个样本取：
- `step_idx = trade_horizon - 1`
- `pred_return = outputs[step_idx]`
- `true_return`：
  - 如果 `target` 是 rank 类：用开盘价 `open_exit/open_trade - 1`
  - 否则：直接用 label 的对应步 `batch_y[step_idx]`

并落成一张表（重要字段）：
- `code`
- `trade_date`（交易日）
- `exit_date`（实际持有到哪天的开盘；rank 模式一定有）
- `pred_return`
- `true_return`
- `suspendFlag`

这张表随后进入回测函数。

---

## 8. 回测阶段：`backtest_topk_detailed()`

回测逻辑（按天）：
1. 按 `trade_date` 分组
2. 每天按 `pred_return` 排序选 TopK（默认 `topk=1`）
3. 计算每只股票净因子（考虑手续费）：

```
net_factor = (1-commission) * (1+true_return) * (1-commission-stamp)
```

4. 当天组合净因子取 TopK 平均
5. 将每天净因子连乘得到资金曲线

### 8.1 一个会让收益“虚高”的坑：NaN `true_return` 的处理

在股票数据里，`true_return` 可能因为各种原因变 NaN：
- `open<=0` 或缺失
- 被 `STOCK_RETURN_LIMIT / stock_true_return_limit` 过滤成 NaN
- packed 填充值/对齐导致的缺失

如果回测把这些 NaN 行直接丢掉，就等价于：
> “提前知道哪些股票未来没法算收益/可能极端波动，于是把它们从候选中踢掉”

这会**偏乐观**。

因此脚本新增了参数（见 `scripts/stock/stock_rolling_backtest.py`）：
- `--backtest_nan_true_return drop`：丢掉 NaN（旧逻辑，可能偏乐观）
- `--backtest_nan_true_return zero`：保留行，把 NaN 当 0 收益（更保守）

> 建议你排查收益异常偏高时，至少跑一版 `--backtest_nan_true_return zero` 看结果是否显著回落。

### 8.2 `trade_horizon!=2` 的风险提示

脚本当前回测是“按 trade_date 逐日复利”的形态。如果你把 `trade_horizon` 设成 3/4/...，代表多日持有，但回测仍可能按日复利滚动，结果容易不严谨（脚本会给出 warn）。

---

## 9. 输出文件解读

每个窗口、每个 split（`test` / `val`）都会输出：

```
predictions.csv     # 原始逐股票预测与真实收益（用于排查）
daily_picks.csv     # 每天选中的 TopK 股票与当日组合收益
daily_metrics.csv   # 每天的资金、回撤、夏普等序列
equity_curve.csv    # 净值曲线
metrics.json        # 汇总指标（年化、回撤、胜率、盈亏比等）
```

如果你怀疑“收益偏高”，第一优先建议你看：
- `daily_picks.csv`：是否大量出现极端 `true_return` 被过滤/缺失导致的异常
- `predictions.csv`：是否某些日期/股票 `true_return` 大量 NaN

---

## 10. 推荐的“更不容易踩坑”的参数组合（排查用）

（1）避免结构性泄露/偏差：
- `--window_order train_val_test`
- `--stock_strict_pred_end True`（默认就是 True）
- `--stock_pack_select_end train_end`（默认就是 train_end；避免 pack_end 的存活者偏差）

（2）回测更保守：
- `--backtest_nan_true_return zero`

（3）不要随意改动：
- `trade_horizon`：默认 2 逻辑最一致；改了要确认持有期与复利逻辑匹配

---

## 11. 运行示例（建议从小跑一窗开始）

### 11.1 快速 sanity：只跑 1 个窗口

```bash
python scripts/stock/stock_rolling_backtest.py \
  --model iTransformer \
  --max_windows 1 \
  --train_epochs 1 \
  --batch_size 256 \
  --backtest_nan_true_return zero
```

### 11.2 正式滚动回测（示例）

```bash
python scripts/stock/stock_rolling_backtest.py \
  --model iTransformer \
  --train_epochs 10 \
  --batch_size 1024 \
  --backtest_nan_true_return zero \
  --resume True
```

> `--resume True` 会跳过“已有 checkpoint 且已有结果”的窗口；如果你改了回测逻辑或参数，需要 `--resume False` 才会重算。

---

## 12. 排查“收益偏高”的 checklist（强烈建议逐项对照）

1. 你是否用了 `--window_order train_test_val`？（早停看未来）
2. 你是否把 `stock_strict_pred_end` 关掉了？（跨 split 标签泄露）
3. 你是否把 `stock_pack_select_end` 设成 `pack_end`？（存活者偏差/看未来选股池）
4. 你的回测是否因为 NaN `true_return` 丢掉了大量行？（隐式“未来可得性过滤”）
   - 用 `--backtest_nan_true_return zero` 对比一遍
5. 你是否改了 `trade_horizon`？（持有期与复利逻辑可能不匹配）
6. 交易成本、滑点、涨跌停、停牌等是否与真实市场一致？（脚本只处理了 `suspendFlag` 和固定费率）

---

## 免责声明

本文档用于解释代码流程与实验复现，不构成任何投资建议。股票回测结果受数据质量、交易规则假设、费用与可交易性约束强烈影响。
