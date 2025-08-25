---
title: Transformer LMs
slug: transformer-lms-1bb4jj
url: /post/transformer-lms-1bb4jj.html
date: '2025-08-23 19:07:23+08:00'
lastmod: '2025-08-25 23:50:11+08:00'
tags:
  - '10423'
keywords: '10423'
toc: true
isCJKLanguage: true
---



# Transformer LMs

‍

​#10423#​

‍

# GQA

[GQA](https://arxiv.org/pdf/2305.13245)

#### Key Insight(Green)

‍

#### Analysis（Red）

‍

#### Key Design（Blue）

‍

#### System Overview（Blue）

‍

#### Key Algorithm（Blue）

‍

#### Experiment（Purple）

‍

#### Other

‍

#### 评估：

- 摘要和引言中没有清晰总结你的方法和成果，引入时候至少应该先阐明研究目标和约束条件。若能结合具体应用场景详细说明则更佳，同时还需解释现有方案为何失效——简而言之，要讲清楚研究动机
- 核心想法是什么，以及这些想法如何突破你具体实现的局限（泛化）
- 有没有简明、清晰的术语
- 大量计算机科学研究都是先有解决方案，再反向推导问题（我自己也常犯这种错）。这个文章是不是需要人为构造一个虚假问题
- 图表表达性如何？理想情况下，图表应当自成体系：说明文字既要概括图表内容，也要阐明所呈现数据的意义

---

‍

‍

‍

‍

‍

‍

# RoPE

[RoPE](https://arxiv.org/pdf/2104.09864)

[苏神原文](https://spaces.ac.cn/archives/8265)​

on GitHub: https://github.com/ZhuiyiTechnology/roformer.

‍

|章节 / 部分|核心论点|论证细节（支撑材料）|
| -------------| ------------------------------------------------------------------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**摘要**|提出 RoPE，构建 RoFormer，在中文长文本任务中表现优越|一句话概括背景（Transformer 位置编码缺陷）、方法（RoPE）、效果（CAIL2019-SCM 提升）、贡献（理论 + 模型）|
|**1. 引言**|Transformer 需位置编码，现有方案存在痛点，RoPE 可解决这些痛点|1. 简述 Transformer 与自注意力的位置无关性；<br />2. 对比绝对 / 相对编码的缺陷（引用 Shaw 2018、Transformer-XL 2019）；<br />3. 点明 RoPE 的核心优势（相对位置、长文本、兼容线性注意力）<br />|
|**2. 相关工作**|系统梳理位置编码的三类方案，凸显 RoPE 的创新性|1. 绝对位置编码：BERT 可训练嵌入、Transformer 正弦嵌入（公式 4）；<br />2. 相对位置编码：Shaw 方案（公式 6-8）、Transformer-XL（公式 9-10）；<br />3. 其他方案：动态位置编码、注意力权重修正；4. 对比指出：现有方案无法兼顾 “相对位置 + 长文本 + 高效注意力”<br />|
|**3. （RoPE）**|RoPE 通过旋转向量实现位置编码，满足 “内积依赖相对位置” 的设计目标|1. 设计目标：公式 11 定义 “内积仅依赖相对位置”；<br />2. 2D RoPE：推导旋转矩阵（公式 13）、内积与相对位置的关系（公式 16）；<br />3. 高维 RoPE：分块对角矩阵（公式 14-15）、参数$\theta_i$设计；<br />4. RoPE 特性：长距离衰减（附录 C 图 2）、长度灵活、兼容线性注意力（公式 18-19）、稳定性（正交矩阵）<br />|
|**4. RoFormer 模型**|用 RoPE 替换 WoBERT 的绝对位置嵌入，构建增强型模型|1. 模型结构：基于 WoBERT（词级中文 BERT），仅替换位置嵌入；2. 对比表格：明确 RoFormer 与 BERT、WoBERT、NEZHA 的差异（分词级别、位置嵌入类型）|
|**5. 实验**|RoFormer 在预训练和下游长文本任务中表现优于传统模型|1. 预训练设置：34GB 中文语料、动态长度训练（128→1536）、批次大小调整；<br />2. 预训练结果：长度越长准确率越高（表 2）；<br />3. 下游任务（CAIL2019-SCM）：数据集介绍（8138 个三元组，长文本为主）、对比模型（BERT-512 等）、结果表格（表 4，RoFormer-1024 最优）<br />|
|**6. 结论与展望**|RoPE 解决了位置编码痛点，RoFormer 验证有效性；未来可扩展多语言、跨模态|1. 核心贡献：RoPE 的理论设计、RoFormer 的工程实现、长文本任务验证；<br />2. 局限：英文任务、跨模态未验证；3. 展望：多语言、跨模态、更长序列应用<br />|
|**附录 A-C**|补充 RoPE 的数学推导与特性证明，增强理论可信度|1. 附录 A：2D RoPE 的完整推导；2. 附录 B：高维 RoPE 的高效计算（元素级运算实现）；3. 附录 C：长距离依赖衰减的数学证明与可视化（图 2）|

#### Key Insight(Green)

Transformer 模型的自注意力机制天然 “位置无关”，必须依赖额外的**位置编码**注入序列顺序信息，但现有方案无法同时满足 “捕捉相对位置依赖、适配长文本、兼容高效注意力” 三大需求

#### Analysis（Red）

历史渊源 [苏神文章](https://spaces.ac.cn/archives/8130)​

- 目的都是用某种方法，让self-attn过程中采用到位置信息($\textbf{q}_m^T \textbf{k}_n$)
- 加法的思路：

  - 绝对的位置编码

    $q_m^tk_n = x_m^tW_q^TW_kx_n + x_m^TW_q^TW_kp_n + p_m^TW_q^TW_kx_n +P_m^TW_q^TW_kp_n$  

    - 绝对编码的位置标识与**固定序列长度强绑定**，无法自然适配训练时未见过的 “超长序列”：

      - 若为**可学习绝对位置编码**（如 BERT、GPT 早期版本）：训练时序列长度固定为 L（如 512），则仅学习了 “位置 1\~L” 的嵌入向量；测试时若遇到长度 \> L 的序列（如 1024），“位置 L+1\~1024” 无对应的预训练嵌入，只能通过截断、 padding 或随机初始化处理，导致位置信息失效，模型性能骤降。
      - 若为**固定公式生成的绝对编码**（如 Transformer 原论文的正弦余弦编码）：虽能通过公式生成任意长度的位置编码（无需预训练），但公式固定的位置信息无法根据任务动态调整（如语义任务需侧重局部语序，摘要任务需侧重长距离逻辑），导致对复杂任务的适配性差。
    - 绝对编码仅关注 “单个位置的序号”，忽略了**位置之间的语义关联性**—— 而自然语言 / 时序数据中，“相对距离” 往往比 “绝对序号” 更重要：“相同相对距离下的相似语义依赖”
    - 绝对编码将 “位置序号” 与 “语义” 强绑定，导致**轻微语序调整可能破坏语义理解**—— 但实际语言中，部分语序变化不影响核心语义：
- 绝对的位置编码外，用**相对的位置编码！**

  - 核心思想是将绝对位置嵌入 p 替换为其正弦编码的相对对应项 $\hat{p}_{m - n}$，同时用两个与查询位置无关的可训练向量 u 和 v 固定第三和第四项。此外，$W_k$ 用于区分基于内容和基于位置的键向量 $x_n$ 和 $p_n$，分别表示为 $W_k$ 和 $\hat{W}_k$，从而得到：

    $\boldsymbol{q}_m^\top \boldsymbol{k}_n = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \widetilde{\boldsymbol{W}}_k \tilde{\boldsymbol{p}}_{m-n} + \mathbf{u}^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + \mathbf{v}^\top \boldsymbol{W}_q^\top \widetilde{\boldsymbol{W}}_k \tilde{\boldsymbol{p}}_{m-n}$

#### Key Design（Blue）

- 设计旋转位置嵌入（RoPE），用乘法旋转替代加法注入；
- ​#TODO#​

#### Key Algorithm（Blue）

|符号|含义与工程定义|
| ------| -------------------------------------------------------------------------------------------|
|$d_{\text{model}}$|模型隐藏层维度（RoFormer 默认与 WoBERT 一致，设为 768），需为偶数（便于拆分为 2D 子空间）|
|$x$|词嵌入矩阵，形状为`[batch_size, seq_len, d_model]`​（batch\_size：批次大小，seq\_len：序列长度）|
|$q, k$|注意力层的查询（Query）、键（Key）矩阵，由$x$经线性投影得到：$q = x \cdot W_q$，$k = x \cdot W_k$，形状均为`[batch_size, seq_len, d_model]`​|
|$pos$|位置索引数组，形状为`[seq_len]`​，取值为$0,1,2,...,seq\_len-1$（对应序列中每个 token 的绝对位置）|
|$\theta$|旋转参数数组，形状为`[d_model//2]`​，用于定义每个 2D 子空间的旋转尺度|

1. 与计算旋转参数 $\theta$
2. 计算每个位置旋转角度
3. 将高维q/k拆分成2D空间并旋转
4. 融入自注意力计算

我的实现如下，正确性不知，仅仅 LLM as a judge结论为可行

```python
class CausalSelfAttention(nn.Module):
    """
    Simple Multi Headed attention. query heads = key heads = value heads
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_query_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_query_head
        self.n_embd = config.n_embd
        
        ################################################     TODO     ################################################
        self.rope = config.rope
        if self.rope:
            # 初始化RoPE
            self.rope_emb = RotaryPositionalEmbeddings(d=config.n_embd / config.n_query_head)
        #############################################################################################################

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)

        ################################################     TODO     ################################################
        if self.rope:
            # 进行RoPE操作
            q = self.rope_emb(q)
            k = self.rope_emb(k)
        #############################################################################################################

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, d) -> (B, nh, T, d)
        end_memory = torch.cuda.memory_allocated()
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, end_memory-start_memory

```

#### Other

1. **英文任务验证缺失**：论文仅在中文任务（WoBERT 底座）上验证，未测试英文基准（如 GLUE、WikiText-103）—— 无法确定 RoPE 在不同语言（尤其是字母语言）中的通用性；
2. **跨模态与更长序列验证不足**：论文提到 “未来探索跨模态”，但未做初步验证；且最大序列长度仅测试到 1536，未验证更长序列（如 4096、8192）下的性能与效率（是否仍能保持$O(d)$复杂度，是否存在梯度问题）；
3. **与其他高效注意力的兼容性未验证**：仅验证了 “线性自注意力”，未测试其他高效注意力（如稀疏注意力、局部注意力）—— 无法确定 RoPE 的兼容性边界。

#### 评估：

- 摘要和引言中没有清晰总结你的方法和成果，引入时候至少应该先阐明研究目标和约束条件。若能结合具体应用场景详细说明则更佳，同时还需解释现有方案为何失效——简而言之，要讲清楚研究动机
- 核心想法是什么，以及这些想法如何突破你具体实现的局限（泛化）
- 有没有简明、清晰的术语
- 大量计算机科学研究都是先有解决方案，再反向推导问题（我自己也常犯这种错）。这个文章是不是需要人为构造一个虚假问题
- 图表表达性如何？理想情况下，图表应当自成体系：说明文字既要概括图表内容，也要阐明所呈现数据的意义

---

‍

‍

‍

‍

‍

# Longformer

[Longforme](https://arxiv.org/pdf/2004.05150)

#### Key Insight(Green)

‍

#### Analysis（Red）

‍

#### Key Design（Blue）

‍

#### System Overview（Blue）

‍

#### Key Algorithm（Blue）

‍

#### Experiment（Purple）

‍

#### Other

‍

#### 评估：

- 摘要和引言中没有清晰总结你的方法和成果，引入时候至少应该先阐明研究目标和约束条件。若能结合具体应用场景详细说明则更佳，同时还需解释现有方案为何失效——简而言之，要讲清楚研究动机
- 核心想法是什么，以及这些想法如何突破你具体实现的局限（泛化）
- 有没有简明、清晰的术语
- 大量计算机科学研究都是先有解决方案，再反向推导问题（我自己也常犯这种错）。这个文章是不是需要人为构造一个虚假问题
- 图表表达性如何？理想情况下，图表应当自成体系：说明文字既要概括图表内容，也要阐明所呈现数据的意义

---

‍

‍

‍

‍

---

‍
