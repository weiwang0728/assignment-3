# 作业 3：使用强化学习训练智能体

11-768 AI Agents 课程，卡内基梅隆大学（CMU），2026 年秋季。

## 概述

在本次作业中，你将协助构建 `scottyrl`——一个使用强化学习训练智能体的极简训练框架。具体而言，你将：

- 构建轨迹处理与优势计算流水线。
- 实现同步的 REINFORCE 与 GRPO 训练。
- 实现带过期 rollout 处理的异步 GRPO。
- 通过受控消融实验比较不同算法与系统设计。
- 在 TextCraft-Synth 上训练并评估一个智能体。

**尽早开始！** 本次作业任务量更大，无论是实现方面，还是你将要启动的各类训练运行的等待时间方面，预计都会比之前的作业花费更长时间。

## Tinker API 兼容性

`scottyrl` 构建在 Tinker API 之上并与之完全兼容。我们建议通读其文档，并在完成作业的过程中随时参考，以熟悉 Tinker SDK 暴露的训练与采样原语：https://tinker-docs.thinkingmachines.ai/tinker/quickstart/。

以 Tinker API 为目标带来了极大的灵活性：它把 RL 框架的高层部分（训练循环、智能体框架、环境、数据处理、奖励设计、优势计算等）与底层分布式 GPU 执行及状态解耦——后者被封装进一个实现了范围受限 API（`forward_backward`、`optim_step`、`sample` 等）的服务中。

重要的是，这使得框架模块化且高度灵活：`scottyrl` 可以与任何实现了类 Tinker 服务的训练后端配合工作。例如：

托管服务（需要 API 额度）：
- [Thinking Machines 官方的 Tinker 服务](https://tinker-docs.thinkingmachines.ai/tinker/)
- [Fireworks 的 Tinker 兼容训练服务](https://docs.fireworks.ai/fine-tuning/training-api/introduction)

自托管服务（自带算力，在自己的 GPU 节点上托管一个 Tinker 兼容的后端服务）：
- [SkyRL 的 Tinker 兼容训练服务](https://docs.skyrl.ai/docs/tinker/overview)（支持全量微调和 LoRA）
- [Miles 的 Tinker 兼容训练服务](https://github.com/radixark/miles/issues/3284)；实验性——目前仅支持 LoRA

对于作业 3，你将使用 SkyRL 的开源 Tinker 服务实现和 [Modal](https://modal.com/) 算力进行自托管。

如果你想在期末项目中使用或在此框架之上进行构建，你可以根据自己的算力或 API 额度自由尝试其他服务选项。

## 使用 SkyRL 和 Modal 自托管

关于如何为各题目设置和配置后端训练服务，请参阅[此处](skyrl_tinker_modal/README.md)的说明。

在开始各题目的训练之前，请务必先阅读该文档，因为我们针对你将执行的不同训练运行推荐了不同的配置。

## 环境配置

使用 [uv](https://docs.astral.sh/uv/) 和 Python 3.12 安装项目：

```bash
uv python install 3.12
uv sync
uv run pytest
```

在起始代码中，25 个测试中有 12 个通过，其余的因 `NotImplementedError` 而失败。随着你填入实现，它们会逐渐通过。所有命令都要在仓库根目录下通过 `uv run` 执行。测试和整个数据流水线都在本地运行，不需要 GPU 或后端服务。

当你准备好启动训练运行时，请为相应部分部署 SkyRL 服务，并按照[自托管指南](skyrl_tinker_modal/README.md)的说明设置 `TINKER_BASE_URL`。共有三个启动脚本，各自按其服务的运行规模设计，下面每个部分都会告诉你使用哪一个。每个服务器一次只接受一个运行，但这三个服务器相互独立，因此你可以[同时](skyrl_tinker_modal/README.md#running-parts-in-parallel)运行不同的部分。在训练之前先让代码通过测试：服务器从运行连接的那一刻起就开始为 GPU 计费，因此一个本可以在本地发现的崩溃仍然会烧钱。

本作业中的所有训练运行都使用 `Qwen/Qwen3.5-4B` 并配合 LoRA（rank 32）。每个必需的运行在 [configs/](configs/) 中都有现成的配置。

## 环境

你将在两个环境上训练智能体。在这两个环境中，智能体每回合通过发出一个工具调用来行动，写作 `<tool>name(args)</tool>`，并收到一条文本观察作为回应。回合（episode）在任务被解决、智能体调用 `finish()`（TextCraft-Synth）或步数预算耗尽时结束。奖励是二值的：成功为 1，否则为 0。

**Number Search（数字搜索）** 是一个规模小、成本低的环境，本作业的大部分内容都将使用它。智能体需要用唯一的工具 `guess(n)` 在 `[low, high]` 中找到一个隐藏数字，该工具会回答"太小"、"太大"或"正确"。它共有 $\lceil \log_2(\text{high}-\text{low}+2) \rceil$ 次猜测机会：这正是最坏情况下二分搜索的预算（包括最后一次猜中的那次），没有任何富余。共有 50,000 个训练任务和 1,000 个验证任务。所有部分都使用同一个 `xml_toolcall` 智能体。推理由 YAML 聊天模板设置控制：Number Search 使用 `model.enable_thinking: false`，TextCraft 使用 `true`。两者都设置 `model.preserve_thinking_history: true`，以保留先前的思考块（包括空的思考块）。

**TextCraft-Synth** 是一个更难的合成（crafting）环境，你将在最后一部分使用它。智能体一开始拥有一个物品清单（inventory）和一份目标物品列表，并有四个工具：`craft`、`get_info`、`view_inventory` 和 `finish`。必须严格遵循配方，原材料只能来自初始物品清单。只有当智能体合成了所有目标物品 *并且* 在步数耗尽前调用 `finish()` 时，它才能获得奖励。共有 852 个训练任务、213 个中等难度的验证任务，以及 136 个我们在第 5 部分用于测试泛化能力的困难验证任务。该环境是论文 [Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) 中首次提出的版本的变体。

这两个环境的训练运行有一个重大差异：我们会对 Number Search 禁用思维链（CoT），但对 TextCraft-Synth 启用它，这使得 TextCraft 的运行时间长得多。

数据集位于 [data/](data/)（见 [`data/README.md`](data/README.md)）。请不要编辑或重新生成它们。

## 每次运行的预期成本与时间

这两个环境的预算差别很大，请围绕它们来规划你的运行：

| 环境 | 每次运行约成本 | 每次运行约时间 | 预期用途 |
|---|---|---|---|
| Number Search | ~$10 | ~40 分钟 | 便宜且快速：用于调试、测试你的实现以及消融实验。 |
| TextCraft-Synth | ~$100 | 6-8 小时 | 昂贵且耗时：第 5 部分的一次性 **hero run（关键运行）**。 |

**把 TextCraft-Synth 当作一次 hero run（关键运行）。** 你应该力争只启动它一次，并且是在你已用 Number Search 对训练器进行了端到端调试和验证之后。不要用 TextCraft-Synth 进行调试或反复尝试。运行开始后，密切关注其指标和转录文本（参见[标准诊断图](#标准诊断图)和[健全性警告](#健全性警告)），以便尽早发现错误，在糟糕的运行烧完预算之前将其停止。

## 代码库概览

一个很好的阅读起点是 `scottyrl/types.py`，它定义了在框架中流转的各类对象：

1. 智能体（`scottyrl/agents/xml_tool_call.py`）与环境交互完成一个回合（`scottyrl/rollout.py`）。这会产生一个 `Trajectory`（轨迹）：回合中每一次 LLM 调用（以 token 空间记录），外加最终奖励。
2. 数据流水线（`scottyrl/advantages.py`、`scottyrl/datagen.py`）把轨迹转换为 `TokenSequence`：带有损失掩码、采样 logprob 和优势的训练序列。
3. 训练器（`scottyrl/train/`）收集 rollout、构建批次，并通过封装了 Tinker 训练 API 的 `TinkerSession`（`scottyrl/tinker_adapter.py`）将它们发送到后端。

[README](README.md) 中有仓库的完整导览。你将按以下顺序实现下列六个文件：

| 文件 | 你要实现的内容 | 部分 |
|---|---|---|
| `scottyrl/advantages.py` | `compute_group_advantages` | 1 |
| `scottyrl/datagen.py` | `extension_suffix`、`stitch_trajectory`、`trajectory_to_training_sequences` | 1 |
| `scottyrl/train/common.py` | `collect_group`、`groups_to_batch` | 2 |
| `scottyrl/tinker_adapter.py` | `TinkerSession.submit_update`（文件其余部分已给出） | 2 |
| `scottyrl/train/sync.py` | `SyncTrainer`：`collect_batch`、`build_batch`、`update`、`evaluate_now`、`train` | 2 |
| `scottyrl/train/async_grpo.py` | `AsyncGRPOTrainer`：`generate_group`、`rollout_worker`、`collect_fresh_batch`、`update`、`evaluate_now`、`train` | 3 |

其余所有内容都是给定代码，我们会在评分前将其还原，因此只有你对这六个文件的修改才计入成绩。请把每个桩函数（stub）的 docstring 当作规格说明。保持函数签名不变；你可以自由添加私有辅助函数。

`tests/` 中的公开测试检查你的代码能否正确接入框架。通过它们并不意味着你的代码完全正确：我们将使用更全面的隐藏测试套件进行评分。

### 聊天模板与序列扩展

提示词（prompt）由一个改编自 Qwen3.5 自带模板的聊天模板（`scottyrl/chat_templates/qwen3_5.jinja`）渲染。一个回合涉及多次 LLM 调用，我们希望把整个回合作为 *一个* 序列来训练，而不是每次调用一个序列。当每次调用的 prompt 恰好是上一个 prompt 及其补全（completion）的精确扩展时，这是可行的（参见[序列扩展](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/sequence-extension/)）。智能体在 token 空间中构建其上下文以保持这种性质。

两个配置字段控制 prompt 的渲染方式：

- `model.enable_thinking`：是否启用 CoT。Number Search 关闭，TextCraft-Synth 开启。
- `model.preserve_thinking_history`：先前回合的 CoT 是否保留在 prompt 中。当为 `true`（默认值）时，序列扩展通常成立，一个回合会成为单个训练序列（偶尔仍可能中断，例如模型以 `<|endoftext|>` 而不是 `<|im_end|>` 停止时）。当为 `false` 时，先前回合的 `<think>` 块会被剥离以节省上下文。这会破坏序列扩展，因此每个回合都会成为各自的训练序列。你的数据流水线需要同时处理这两种情况。

### 运行实验

每次运行由一个包含 `model:` 和 `train:` 两节的 YAML 配置描述。你可以在命令行通过添加 `--set section.field=value` 覆盖任意字段，例如 `--set train.lr=2e-5`（完整命令参见[先做一次短探针](#先做一次短探针)）。

一次运行由 `train.run_name` 标识，并将所有内容写入 `runs/`：

| 产物 | 路径 |
|---|---|
| 每步指标 | `runs/<run_name>.jsonl` |
| rollout 转录文本 | `runs/<run_name>_trajectories/step_XXXXXX/` |
| 保存的检查点 | `runs/<run_name>.checkpoints.json` |

每份日志的第 0 步包含对未训练模型的基线评估。提供的配置每五步保存一次转录文本。请阅读它们！这是了解你的智能体实际在做什么的最快方式。

必需的运行请保留给定的运行名称，因为评分会查找它们。已有输出的运行名称会被拒绝，以免你意外覆盖结果。新实验请取一个新名称，或传入 `--set train.resume=true` 以从上次的检查点继续一次崩溃的运行。

### 记录到 Weights & Biases（可选）

每次运行都会把指标写入 JSONL 日志，我们评分依据的正是这份日志。我们仍然建议同时记录到 [Weights & Biases](https://wandb.ai)：它能在运行进行时为每个指标提供实时图表，使你更容易盯着长时间运行、并排比较多个运行。要启用它，先登录一次，并在训练命令中添加 `--set train.wandb_project=<project>`：

```bash
uv run wandb login
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
    --set train.wandb_project=11768-a3
```

每次运行以其 `run_name` 显示，恢复的运行会继续同一个 W&B run。如果 W&B 无法启动（例如你未登录），运行会打印 `[metrics] wandb disabled` 并仅使用 JSONL 日志继续，所以请在启动时留意这一行。

### 标准诊断图

每个实验报告部分（第 2 部分、第 3 部分、第 4 部分的每个小节，以及第 5 部分）都应针对其所讨论的每次训练运行，给出以下指标随训练步数变化的图：

- 奖励质量：`train/success_rate`、`eval/success_rate`，以及所有记录在案的 `train/pass@*` 和 `eval/pass@*`；
- 优化：`train/grad_norm`；
- 策略漂移：`train/importance_weight_mean`、`train/importance_weight_max` 和 `train/importance_weight_std`；
- 数据健康度：`train/frac_degenerate_groups` 和 `train/entropy_proxy`。

你可以把相关的指标和运行合并成清晰易读的多面板图。请为每条 pass@ 曲线标注其确切的键名：训练和评估通常使用不同的 $k$ 值，因此它们是不同的量。

### 时间指标

每条训练记录还带有自上一条训练记录以来这段区间的计时指标（单位为秒）。它们由给定代码记录，因此你无需实现任何内容，它们也不属于标准诊断图。请利用它们了解一次运行把时间花在了哪里，并尽早发现变慢的迹象。

| 指标 | 含义 |
|---|---|
| `train/step_time_s` | 该步的实际耗时（wall-clock）。 |
| `train/update_time_s` | 优化器更新所花的时间，包括周期性的可恢复检查点保存。 |
| `train/learner_wait_s` | 步耗时减去更新和采样器快照保存的时间：学习器花在其他所有事情上的时间。 |
| `train/learner_wait_frac` | `learner_wait_s` 占 `step_time_s` 的比例。 |
| `train/rollout_time_s` | 从本步内最早结束的训练回合的开始，到最晚结束的训练回合的结束。 |
| `train/episode_time_mean_s` | 这些回合的平均时长。 |

## 背景知识

### 从回合到梯度

设拼接后的回合包含 token $t_0,\ldots,t_{n-1}$，并设 $m_j \in \{0,1\}$ 标记目标 token $t_{j+1}$ 是否由策略采样（而非来自 prompt 或环境）。给定回合级优势 $A$，策略梯度为

$$\nabla_\theta J(\theta) = \mathbb{E}\Big[\, A \sum_{j:m_j=1} \nabla_\theta \log \pi_\theta(t_{j+1}\mid t_{\le j}) \Big].$$

只有策略采样出的 token 才应进入损失。如果掩码包含观察 token，模型将学会预测环境，而不是学会行动。

### REINFORCE 与 GRPO

REINFORCE 使用 $A = r$：成功回合的每个被采样 token 都得到强化，失败的回合不贡献任何东西。该估计是无偏但有噪声的，在稀疏奖励下尤其如此。

GRPO 为同一任务采样一组 $G$ 个 rollout，并使用该组的平均奖励作为基线：

$$A_i = r_i - \frac{1}{G}\sum_{k=1}^{G} r_k.$$

减去基线消除了任务难度的影响，使得只有当某个 rollout 表现好于策略在同一任务上的典型尝试时才会被强化。如果一组内所有奖励都相同，所有优势都为零，该组就不携带任何学习信号。我们称这样的组为 *退化组（degenerate）* 并跳过它。遵循 [Dr.GRPO](https://arxiv.org/abs/2503.20783)，我们不除以组内奖励的标准差，也不添加对参考策略的 KL 惩罚。

在配置中，`train.algo` 选择优势规则，`train.loss_fn` 则单独选择下面描述的 token 级损失。

### 异步训练与重要性加权

同步训练在收集 rollout 和更新模型之间交替进行，因此每个批次都是 on-policy 的，但学习器在 rollout 期间闲置，而 rollout 工作进程在更新期间闲置。异步训练将两者重叠。这更快，但现在一个组可能是由落后了若干次更新的策略采样得到的。我们用两种方式处理这种 *过期（staleness）*：比 `train.max_staleness` 个版本更旧的组被丢弃，其余的则用重要性加权进行修正。

对于一个从 $\pi_{\text{old}}$ 采样、在 $\pi_\theta$ 下训练的 token，重要性比率为

$$\rho_j = \exp\big(\log \pi_\theta(t_{j+1}\mid t_{\le j}) - \log \pi_{\text{old}}(t_{j+1}\mid t_{\le j})\big).$$

框架支持三种损失：

- `ppo`，裁剪代理损失：$-\sum_{j:m_j=1} \min(\rho_j A_j,\ \mathop{\text{clip}}(\rho_j, 1-\epsilon, 1+\epsilon)\,A_j)$
- `importance_sampling`，相同但不裁剪：$-\sum_{j:m_j=1} \rho_j A_j$
- `cross_entropy`，优势加权对数似然，完全不使用比率：$-\sum_{j:m_j=1} A_j \log \pi_\theta(t_{j+1}\mid t_{\le j})$

GRPO 运行使用 `ppo`，REINFORCE 运行使用 `importance_sampling`，而 `cross_entropy` 仅用于第 4 部分的"无修正"消融实验。

$\rho$ 的统计量会在每次更新时记录。在同步运行中，它们应始终非常接近 1。如果不是，你的采样 logprob 很可能挂在了错误的 token 上。

## 第 0 部分：热身（0 分）

在编写任何代码之前，请先熟悉智能体、rollout 和环境的实现：`scottyrl/agents/xml_tool_call.py`、`scottyrl/rollout.py` 和 `scottyrl/envs/`。

## 第 1 部分：数据流水线（20 分）

在这一部分，你将把智能体轨迹转换为训练数据。

**实现** `scottyrl/advantages.py` 中的 `compute_group_advantages` 以及 `scottyrl/datagen.py` 中的三个函数，遵循它们的 docstring。

请小心处理下一 token 对齐：一个 $n$ 个 token 的序列有 $n-1$ 个预测目标，因此 `mask`、`logprobs` 和 `advantages` 的长度都是 $n-1$，且位置 `j` 描述的是目标 `tokens[j+1]`。如果长度为 $c$ 的补全跟在长度为 $L$ 的上下文之后，其位置是 `L-1 : L-1+c`，而不是 `L : L+c`。

拼接时，只要下一次调用的 prompt 是你已累积内容的扩展，就继续扩展当前序列；否则开始一个新序列。只有补全 token 的掩码为 1，并同时带上它们的采样 logprob 和该回合的优势。

**测试**：运行 `uv run pytest tests/test_interfaces.py -k "advantages or datagen"`（该文件中的其他测试属于第 2 部分）。`tests/test_envs.py` 中的 `ThinkStrippingPolicy` 会给你一个破坏序列扩展的回合，便于在本地测试这种情况。

## 第 2 部分：同步 REINFORCE 与 GRPO（25 分）

接下来，你将构建同步训练循环，并用它训练你的第一批智能体。

**实现**以下内容：

- `scottyrl/train/common.py` 中的 `collect_group` 和 `groups_to_batch`。前者并发运行一个组的 rollout，后者计算优势并把各组合并成一个训练批次。
- `scottyrl/tinker_adapter.py` 中的 `TinkerSession.submit_update`。请参考 tinker 文档了解其 API 的用法。
- `scottyrl/train/sync.py` 中的 `SyncTrainer`：四个小辅助方法和把它们串起来的 `train` 循环。确保每次更新后刷新采样策略，使下一个批次是 on-policy 的。

**测试**：运行 `uv run pytest tests/test_interfaces.py tests/test_contracts.py -k "not async"`。其余的 `test_async_*` 测试将在你完成第 3 部分后通过。

**运行**两个 Number Search 实验，使用提供的配置不做修改，对接 `serve_number_search_sync.py` 部署。我们建议在做完整运行之前先进行一次[短探针](#先做一次短探针)作为健全性检查。

```bash
uv run python scripts/train_sync.py --config configs/p2a_number_search_reinforce.yaml
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml
```

`p2a` 是每个任务单个 rollout 的 REINFORCE，`p2b` 是每个任务八个 rollout 的 GRPO。两者都训练 50 步、每步八个任务，并且每 10 步在 50 个验证任务上评估、每个任务四个 rollout。

**报告：**

- 附上两次运行的[标准诊断图](#标准诊断图)。
- 用 3 到 5 句话从学习速度的角度比较 REINFORCE 与 GRPO，并解释这种差异是否在意料之中以及为什么。正常运行的奖励应在 50 步内明显升至其第 0 步基线之上。如果你的曲线是平的，请在撰写分析之前先调试你的实现。

## 第 3 部分：异步 GRPO（25 分）

在这一部分，你将通过生产者-消费者设计减少训练中的空闲时间。rollout 工作进程持续使用最新发布的策略生成组，并把它们推入一个有界队列。学习器从队列中取出组、在其上训练，并发布新的策略版本。

**实现** `scottyrl/train/async_grpo.py` 中 `AsyncGRPOTrainer` 的六个学生方法，遵循它们的 docstring。需要注意以下几点：

- 每个组都应使用事先捕获的同一个 `(policy, version)` 对来生成，这样即使中途发布了新策略，它所携带的版本仍然是正确的。过期检查由学习器在消费该组时进行，而不是由工作进程进行。
- 队列是有意设计为有界的。让 [`asyncio.Queue`](https://docs.python.org/3/library/asyncio-queue.html) 通过 `await queue.put(...)` 和 `await queue.get()` 提供背压（backpressure），而不是轮询 `full()` 或 `empty()`。
- 评估和关闭都不应能卡住训练。把每次评估作为后台任务在捕获的 `(policy, version)` 上运行（与训练重叠）。训练结束后，[取消](https://docs.python.org/3/library/asyncio-task.html#task-cancellation)并等待工作进程结束；如果有任何地方出错，取消其他所有任务并让原始错误传播出去，而不是卡死。

**测试**：运行 `uv run pytest`：整个公开测试套件此时应全部通过。`tests/test_contracts.py` 中的 `test_async_*` 测试用假的 rollout 和更新驱动你的训练器，因此你可以在不消耗任何算力的情况下把并发逻辑调对。

**运行**提供的配置不做修改，对接 `serve_number_search_async.py` 部署（异步训练需要独立的 GPU 分别用于训练和采样，因此第 2 部分的服务器不行）：

```bash
uv run python scripts/train_async.py --config configs/p3_number_search_async.yaml
```

`p3` 与 `p2b` 相当（每步八组、每组八个 rollout，共 50 步），但由八个 rollout 工作进程生成，并允许最大为 2 的过期度。运行结束后，它会打印 `train()` 返回的总量，并把它们写入 `runs/p3_number_search_async.jsonl` 的最后一条记录，字段为 `summary/train_steps`、`summary/groups_consumed` 和 `summary/groups_dropped_stale`。第 4.2 部分会用到 `summary/groups_dropped_stale`。

**报告：**

- 附上 `p3` 的[标准诊断图](#标准诊断图)。在分析中，请比较重要性权重的均值、最小值、最大值和标准差；接近 1 的均值可能掩盖了厚重的尾部。
- 使用 `train/step_time_s`（或相邻训练记录的 `time` 字段之差）比较每步的实际耗时与 `p2b`，并量化加速比。
- 从转录文本中挑选一个早期、一个中期和一个晚期的训练 rollout（给出每条的步数、任务 id 以及文件头中的 `policy_version`）。描述至少两种智能体行为发生改变的具体方式，例如它如何利用反馈、它是二分还是线性扫描，或者它如何从无效调用中恢复。

## 第 4 部分：Number Search 上的消融实验（20 分）

现在用你的训练器进行三项小研究。每一项都要在报告中写一个简短的小节，包含要求的图和一段分析。使用提供的配置不做修改。

### 4.1 组大小

比较 `p2a`（$G=1$，REINFORCE）、`p41`（$G=2$，GRPO）和 `p2b`（$G=8$，GRPO）。唯一的新运行是 `p41`，它使用与第 2 部分相同的部署：

```bash
uv run python scripts/train_sync.py --config configs/p41_number_search_grpo_g2.yaml
```

**报告：** 附上所有三次运行的[标准诊断图](#标准诊断图)。用 `eval/success_rate` 和 `train/frac_degenerate_groups` 解释更小的组对优势方差以及每步有用序列数量的影响。REINFORCE 的退化比例请报告为 N/A：它没有组基线，因此其记录的 0 值没有意义。请记住 `p2a` 在优势规则和损失上也与 GRPO 运行不同，因此这并不是纯粹的组大小消融。

### 4.2 同步与异步训练

**报告：** 附上 `p2b` 和 `p3` 的[标准诊断图](#标准诊断图)。复用你在第 3 部分中的实际耗时分析，并加上每次运行最终的 `eval/success_rate`（一个小表格即可）。解释加速来自哪里，以及在过期数据上训练是否损害了样本质量。用重要性权重统计量以及 `p3` 日志最后一条记录中的 `summary/groups_dropped_stale` 计数作为证据。

### 4.3 更高过期度下的重要性加权

运行两个仅在损失上不同的异步实验，使用与第 3 部分相同的部署：

```bash
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_iw.yaml
uv run python scripts/train_async.py --config configs/p43_number_search_async_stale4_noiw.yaml
```

两者都允许最大为 4 的过期度，以使效果可见。`iw` 运行使用 PPO，而 `noiw` 使用优势加权交叉熵，完全忽略重要性比率。

**报告：** 附上两次运行的[标准诊断图](#标准诊断图)。

## 第 5 部分：TextCraft-Synth（10 分）

在最后一部分，你将用你的异步 GRPO 训练器在 TextCraft-Synth 上以提供的配置进行单次运行，训练一个智能体，然后评估它在中等和困难验证集上的表现。

> **这是一次 hero run（关键运行，约 $100，6-8 小时）。** 只有在你的异步训练器在 Number Search 上正常工作之后才启动它，并监控指标以便在出现问题时尽早停止。参见[每次运行的预期成本与时间](#每次运行的预期成本与时间)。

**运行**提供的配置不做修改。它启用了思考（thinking），每回合最多 512 个 token。它的 50 步中，每步在 16 组、每组八个 rollout 上训练，由 32 个 rollout 工作进程生成，允许最大为 4 的过期度，并每 10 步评估一次。TextCraft 回合比 Number Search 回行长得多，且大部分时间花在采样上，因此这一部分有自己的、更大的部署（`serve_textcraft.py`）：

```bash
uv run python scripts/train_async.py --config configs/p5_textcraft_grpo.yaml
```

**评估**最终适配器（adapter）和基座模型在中等和困难验证集上的表现，每个任务四个 rollout。从 `runs/p5_textcraft_grpo.checkpoints.json` 读取 `final_sampler_path` 以找到模型 ID 和快照名称（例如 `tinker://model_abc123/v50`）。下载该采样器检查点并部署：

```bash
# 在仓库根目录下执行（替换成你的模型 ID 和快照名称）：
mkdir -p runs/p5_adapter
uv run --extra modal modal volume get scottyrl-textcraft-state \
    checkpoints/<model-id>/sampler_weights/<snapshot>.tar.gz runs/p5_adapter.tar.gz
tar -xf runs/p5_adapter.tar.gz -C runs/p5_adapter

cd skyrl_tinker_modal
LORA_ADAPTER=../runs/p5_adapter \
    uv run --extra modal modal deploy serve_evaluation.py
cd ..

# 复制 deploy 打印的 URL，例如 https://your-workspace--scottyrl-evaluation-serve.us-east.modal.run：
export EVAL_SERVER=<printed_url>

# 微调模型：先 medium，再 hard
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model a3 --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model a3 --k 4 --tasks data/textcraft_val_hard.jsonl

# 基座模型：先 medium，再 hard
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model Qwen/Qwen3.5-4B --k 4
uv run python scripts/evaluate_checkpoint.py --config configs/p5_textcraft_grpo.yaml \
    --server "$EVAL_SERVER" --model Qwen/Qwen3.5-4B --k 4 --tasks data/textcraft_val_hard.jsonl

# 完成后停止推理部署：
uv run --extra modal modal app stop scottyrl-evaluation
```

其他服务选项参见[服务指南](skyrl_tinker_modal/README.md#standalone-evaluation-with-vllm)。

**导出**你的最终适配器到 Hugging Face，以便我们可以自行评估。该脚本通过你的 `serve_textcraft.py` 部署下载检查点，因此该部署必须处于运行状态。如果你已经停止了它，请用同样的命令重新部署，并保持 `TINKER_BASE_URL` 指向它；你的检查点持久化保存在它的卷（volume）上。训练结束后立即导出可以省去这次额外的启动。第一条命令是一次性的登录：

```bash
uv run hf auth login
uv run python scripts/export_checkpoint.py --run p5_textcraft_grpo
```

该脚本将 LoRA 适配器上传到一个名为 `<hf-username>/scottyrl-a3-p5_textcraft_grpo` 的公开模型仓库，并将其作为 `hf_model_id` 记录在检查点文件中。添加 `--exist-ok` 可更新已有的导出，请在成绩公布之前保持模型公开。

**报告：**
- 附上 `p5` 训练运行的[标准诊断图](#标准诊断图)。
- 一个包含四行的表格（微调模型和基座模型，各自在两个验证集上），展示 `success_rate` 和 `pass@4`。
- 你对这次运行的分析。用几句话描述你的结果。

## 提示与调试

### 先做一次短探针

在投入一次长时间运行之前，先用一个一次性名称试一次短运行：

```bash
uv run python scripts/train_sync.py --config configs/p2b_number_search_grpo.yaml \
    --set train.num_steps=10 --set train.eval_every=2 --set train.eval_tasks=20 \
    --set train.run_name=probe_p2b
```

注意：如果你只想确认训练能跑起来而不运行评估，可以添加 `--set train.eval_every=0`（或等价的 `--set train.eval_tasks=0`）来完全禁用评估，包括第 0 步的基线。此时日志将只包含训练记录，因此请跳过下文检查项 1 中涉及评估的部分。

然后检查：

1. 日志在第 0 步有基线评估，每步一条训练记录，且每两步一次评估。
2. `step_000005/` 下的转录文本读起来应像真实回合：一个补全、一个解析出的工具调用以及环境的回复。`null` 工具调用意味着补全解析失败。唯一的例外是带有 `stop_reason` 而没有补全的最后一个回合：该回合丢失了一次采样调用（例如因为它超出了服务器的上下文窗口），并计为失败。
3. `train/importance_weight_mean` 保持接近 1，同步运行本应如此。
4. 你能解释终端输出中的每一行 `[sanity]`。

注意，探针只检查一切是否连接正确。十步太少了，无法判断智能体是否在学习。

### 健全性警告

训练期间，当某些情况看起来不对时，框架会打印 `[sanity]` 警告：熵坍缩、重要性权重远离 1、梯度尖峰，或几乎所有组都退化（确切阈值见 `scottyrl/metrics.py`）。

## 报告

提交一个 `report.pdf`，放在仓库根目录。每部分下列出了应包含的内容。请让你的图清晰易读（坐标轴、图例、标题），并讨论你展示的每条曲线。

## AI 使用情况

提交一个 `AI_USAGE.md` 文件，详细说明你在本次作业中对任何 AI 技术的使用。列出你使用的所有工具，并清楚描述你如何使用每一个工具。如果你没有使用任何 AI 辅助，请在该文件中声明。

将该文件放在仓库根目录，与 `report.pdf` 并列。打包脚本会把它包含进你的 `submission.zip`，没有它将拒绝打包。

## 提交

当所有七次必需运行和第 5 部分导出完成后，打包你的提交：

```bash
uv run python scripts/package_submission.py
```

该脚本会打印一个清单并生成 `submission.zip`，其中包含你的六个源文件、每次必需运行的配置、日志、转录文本和检查点文件、你的 `report.pdf`、你的 `AI_USAGE.md`，以及一个记录了第 5 部分适配器 `hf_model_id` 的清单文件。它拒绝打包不完整的提交，并告诉你缺了什么。

将 `submission.zip` 上传到 Canvas。迟交政策遵循课程大纲。

## 评分

| 部分 | 分数 |
|---|---|
| 1 | 20 |
| 2 | 25 |
| 3 | 25 |
| 4 | 20 |
| 5 | 10 |
| | **100** |
