# Alignment Probing on COCO-2K —— 70 tokenizer × 3 LLM 对照实验

> 精简归档说明：逐 `(encoder, LLM, budget)` 的原始 JSON、embedding 缓存和日志
> 已删除；逐 encoder 分数、相关性和 FLOPs 已保留在 `results/` 的汇总表中。

作为 SAIL（CVPR'25, `/cache/wangky/SAIL`）的对照：冻结视觉 tokenizer 与
LLM，只训练一层轻量 **alignment layer**，然后做图文 retrieval，用检索分数
衡量每个 (tokenizer, LLM) 组合的跨模态可对齐性。

* **tokenizer**：`/cache/wangky/tokenizer.txt` 里的 70 个（65 个 continuous +
  5 个 discrete），配置直接复用 `/cache/wangky/ocr_exp/VTB/configs/`，视觉塔
  的构建方式与 VTB 的 LLaVA 训练管线完全一致。
* **LLM**：`qwen25` (Qwen2.5-1.5B-Instruct)、`qwen3` (Qwen3-1.7B)、
  `smollm2` (SmolLM2-1.7B-Instruct)。
* **数据**：COCO-2K —— 从 MSCOCO Karpathy test（5000 图 × 5 caption）用
  seed=42 采样 2000 张图，**1600 训练 / 400 测试**。每张图保留 5 条 caption。
* **对齐层**（照搬 SAIL `model/sail_model.py::AlignmentLayer`）：
  两侧各一个 `LayerNorm + 映射网络`（默认 `linear`，可选 `mlp` / `star`）投到
  共享维度，配可学习的 `logit_scale` / `logit_bias`，用 **SigLIP** 损失训练。
* **打分**：测试集上 i2t / t2i 的 R@1/5/10 的均值（mean recall）。
  同时给两种协议：`1cap`（400×400 一图一句的配对矩阵）和
  `5cap`（标准 COCO 协议，400 图 × 2000 句）。

## 流水线

和 SAIL 一样分两段，**先把 embedding 预编码好再训对齐层**，所以 70 个视觉
特征 + 3 个文本特征就能拼出全部 210 个组合，不用重复跑 backbone。

```
build_data.py     COCO-2K 划分           -> data/coco2k.json
encode_text.py    冻结 LLM 编码 caption   -> cache/text/<llm>/emb_mean.npy   [2000,5,D]
encode_vision.py  冻结 tokenizer 编码图像 -> cache/vision/<slug>/emb.npy     [2000,Dv]
train_align.py    训练对齐层 + 检索评测   -> results/align/<llm>/<slug>.json
summarize.py      汇总                   -> results/summary_*.{csv,md}
```

## 一键运行（8 卡）

```bash
cd /cache/wangky/align_probe
nohup bash run_all.sh > logs/run_all.log 2>&1 &
tail -f logs/run_all.log
```

可调环境变量：

```bash
GPUS="0,1,2,3"        # 用哪些卡（默认 0-7）
STAGES="vision align" # 只跑某几步：data text vision align summary
VISION_BATCH=32       # 视觉编码 batch（显存不够会自动对半降）
WORKERS=6             # dataloader 线程
ASHARDS=2             # 每个 LLM 并行几个对齐训练进程
EXTRA_ALIGN="--linear-type mlp --steps 3000"   # 透传给 train_align.py
EXTRA_VISION="--overwrite"                      # 透传给 encode_vision.py
```

所有阶段都有 tqdm 进度条：每个 shard 在自己的 `logs/*.log` 里有细粒度进度，
`run_all.sh` 前台另有一个总进度条（`watch_progress.py`，按已完成的缓存/结果
文件计数，vision 阶段 x/70，align 阶段 x/210）。

> 日志里 tqdm 用 `\r` 刷新，直接 `cat` 会挤成一行，看的时候用
> `tr '\r' '\n' < logs/vision_shard0.log | tail -40`。

## 分步 / 单独运行

```bash
# 0) 数据划分（秒级）
python build_data.py

# 1) 文本预编码（每个 LLM 约 1 分钟）
CUDA_VISIBLE_DEVICES=0 python encode_text.py --llm qwen3 --device cuda:0

# 2) 视觉预编码；分片跑，第 s 片取 slug[s::N]
CUDA_VISIBLE_DEVICES=3 python encode_vision.py --shard 0 --num-shards 8 \
    --device cuda:0 --batch 32
# 只跑指定 tokenizer
CUDA_VISIBLE_DEVICES=3 python encode_vision.py --slugs dinov2_base,toklip_s_256 --device cuda:0

# 3) 训练对齐层 + 检索（每个组合几十秒）
CUDA_VISIBLE_DEVICES=4 python train_align.py --llm qwen3 --device cuda:0
CUDA_VISIBLE_DEVICES=4 python train_align.py --llm qwen25 --slugs dinov2_base --device cuda:0

# 4) 汇总
python summarize.py --protocol 1cap
python summarize.py --protocol 5cap
```

三个脚本都做**断点续跑**：已经存在的 `emb.npy` / `<slug>.json` 会跳过，加
`--overwrite` 强制重算。单个 tokenizer/组合失败不会中断整个 sweep，失败记录在
`results/vision_failures_shard*.json` 和日志里的 `[FAIL]` 行。

## 对齐层超参（`train_align.py`）

默认对齐 SAIL `scripts/alignment_probing.sh`：`linear` 映射、
`logit_scale=20`、`logit_bias=-10`、SigLIP 损失、AdamW(β=0.9/0.99)。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--linear-type` | `linear` | `linear` / `mlp` / `star`（SAIL 的 StarMLP） |
| `--target-dimension` | 1024 | 共享空间维度 |
| `--loss` | `siglip` | 或 `clip`（InfoNCE） |
| `--steps` | 1500 | 训练步数（cosine 衰减） |
| `--batch-size` | 512 | 对比学习 batch |
| `--lr-grid` / `--wd-grid` | `1e-3` / `1e-4,1e-2,1e-1` | 网格搜索 |
| `--train-captions` | 1 | 每图每步采样几条 caption（1~5） |
| `--val-size` | 160 | 从 1600 训练图里留出来选超参/选步数的验证集 |

**测试集只用于最后打分**：超参和 early-stop 步数都在 160 张验证图上选，选完
把最优 checkpoint 拿到 400 张测试图上评一次。1600 = 1440 train + 160 val。

## 产出

* `results/align/<llm>/<slug>.json` —— 单个组合的完整指标（两种协议的
  i2t/t2i R@1/5/10、选中的超参、验证分、耗时）。
* `results/summary_1cap_long.csv` —— 210 行长表，每行一个组合。
* `results/summary_1cap_wide.csv` —— 70×3 宽表，行 tokenizer、列 LLM。
* `results/summary_1cap.md` —— 按三个 LLM 平均分排序的 markdown 表。
* `5cap` 同名文件是标准 COCO 5-caption 协议下的同一套表。

---

# 变体二：SAIL 协议（CC3M-2K 训练 → COCO-2K 评测）

第一版是"COCO 内部划分"，训练和评测同域。这一版改成 SAIL 论文的真实协议：
**在 CC3M 上训对齐层，在 COCO 上做零样本检索**，并把整个 COCO-2K（2000 张）
全部当评测 gallery。算力卡在 **~1.77 PFLOPs/tokenizer**。

## 数据

* **训练**：CC3M-2K —— 从 `/cache/wangky/DownloadCC3M`（已下载 21 万对）
  seed=42 采 2000 对。每张图逐一解码验证，损坏的跳过（实际跳了 27 张）。
  * 主 caption = `raw_caption`，额外正样本 = `longSV_captions`
    （即 SAIL `scripts/alignment_probing.sh` 里的 `dreamclipcc3m_raw` +
    `dreamclipcc3m_longSV`）
  * `data/cc3m2k.json` 是数据集本身；
    **`data/cc3m2k_selected.csv` 是单独抽出来的选用清单**（idx、csv 行号、
    相对路径、原始 URL、两条 caption），方便核对和复现。
* **评测**：COCO-2K 全部 2000 张（不再切 1600/400），5 caption 标准协议。
  gallery 从 400 涨到 2000，所以绝对分数和第一版不可比。

## 对齐层配置：SAIL 原样

照抄 `scripts/alignment_probing.sh` + `train/params.py` 的默认值，**没有任何
调参**：

| 项 | 值 | 出处 |
|---|---|---|
| 映射网络 | `linear` | `linear_type="linear"` |
| 共享维度 | 2048 | `d=2048` |
| 损失 | SigLIP + longSV 额外正样本 | `--siglip` + `extra_text_embedding_list` |
| logit_scale / bias | 20 / −10 | `logit_scale=20 logit_bias=-10` |
| 优化器 | **Lion** | `params.py` 的 `--optimizer` default |
| lr / wd / betas | 1e-5 / 1e-7 / (0.9, 0.99) | `lr=1e-5 --wd 1e-07 --beta1 0.9 --beta2 0.99` |
| 调度 | cosine + `ceil(0.1×total)` warmup | `main.py` |
| 梯度裁剪 | 无 | `--grad-clip-norm` default `None` |
| batch | 全批（SAIL 的 32768 > 2000） | `bs=32768` |
| 超参搜索 / 早停 / 验证集 | 都没有，取最终 checkpoint | SAIL 三者皆无 |

Lion 和 cosine 调度从 `/cache/wangky/SAIL` 原样 vendored 到
[`sail_optim.py`](sail_optim.py)（Apache-2.0，附出处），可逐行比对。

**唯一抄不了的是训练时长。** SAIL 的 `epochs=100` 是在 23M 对上跑的 ≈ 7 万次
参数更新；同样 100 epoch 放在 2000 对全批上只有 100 次更新，实测落在随机水平
（COCO mean recall 0.34–0.93，随机基线 0.27%）。所以步数改由**收敛**决定：在
SAIL 自己的 Lion + lr=1e-5 下，训练损失在 2000 步左右进入平台期
（`logs/lion_convergence_probe.txt`，全程未使用 COCO 测试集）。**取 2000 步。**

这不是为了刷分 —— 步数由训练损失定，不由测试分数定；所有 70 个 tokenizer 用
同一个步数，和 SAIL 一样只有一套配置。已知副作用：收敛慢的大编码器在 2000 步
仍在爬坡（`pe_core_g14_448` 在 2000 步 60.3、4000 步 73.7）。按 tokenizer 分别
选步数就是在给 baseline 调参，故意不做。

## 运行

```bash
cd /cache/wangky/align_probe
nohup bash run_cc3m.sh > logs/run_cc3m.log 2>&1 &
tail -f logs/run_cc3m.log
```

环境变量：`GPUS`、`STAGES`（`data text vision align summary`）、`TAG`、
`ASHARDS`、`ALIGN_ARGS`（整段透传给 `train_align.py`）。

结果落在 `results/sweeps/cc3m2k/<llm>/<slug>.json`，汇总表和与主表的相关性
分别由 `summarize.py --tag cc3m2k` 和 `correlate.py --tag cc3m2k` 产出。

## 算力核算

```bash
python budget.py --extra-positive --grid 1 --steps 2000 --target-dimension 2048 --val-frac 0
```

| 组成 | 平均/tokenizer |
|---|---|
| 视觉编码（4000 张 = 2000 CC3M + 2000 COCO） | 1.52 P |
| 对齐层训练+评测（3 LLM × 2000 步） | 1.30 P |
| 文本编码（摊到 70） | 81 T |
| **合计** | **2.90 P** |

对比：第一版 COCO 内部划分是 0.94 P/tokenizer。这一版更贵，因为图多了一倍
且训到了收敛。

口径是 2×MAC（乘加分开），和 `flops.py` 一致。

## 注意

* 第一版（COCO 内部划分）的结果仍在 `results/align/`，用
  `--loss-reduction sum_over_n` 可逐位复现 —— 那批跑的时候损失用的是
  `-sum/N`，这一版改成 SAIL 的 `-mean`。AdamW 近似梯度尺度不变，实测差
  0.4–1.0 分。
* 两版的绝对分数不可比（gallery 400 vs 2000，且有域偏移），只能比排名和
  与主表的 Spearman。
