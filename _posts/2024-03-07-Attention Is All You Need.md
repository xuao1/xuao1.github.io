---
title: Paper Reading Attention Is All You Need
author: xuao
date: 2024-03-07 16:29:00 +800
categories: [Paper Reading]
tags: [AI, Machine Learning, Transformer, Attention]
---

# Attention Is All You Need

## 1. 概括

### 1.1 是什么

Transformer

摒弃了循环性质、完全依赖于注意力机制来绘制输入和输出之间全局依赖关系的模型架构

> a model architecture **eschewing recurrence** and instead **relying entirely on an attention mechanism** to draw global dependencies between input and output

### 1.2 为什么

sequential nature **precludes parallelization** within training

减少了每层的计算复杂度；可并行化；长依赖中的 path length 短。

> One is the total computational complexity per layer. 
>
> Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
>
> The third is the path length between long-range dependencies in the network.

可解释性：Self-attention could yield more interpretable models. 

+ individual attention heads clearly learn to perform different tasks
+ exhibit behavior related to the syntactic and semantic structure of the sentences

### 1.3 怎么做

> based solely on **attention mechanisms**

![Transformer-architecture]({{ site.url }}/my_img/Transformer-architecture.png)

+ **Encoder 和 Decoder** 都包含 N 层 layers，都引入了残差连接（a residual connection）
  + Encoder 每层 layer 包含一个 multi-head self-attention 和一个 feed-forward network
  + Decoder 每层 layer 包含一个 ，masked multi-head self-attention，一个  multi-head self-attention,  和一个 feed-forward network
  + mask 是为了在训练时，避免看到后续 token，具体实现就是设置为负无穷
+ **Attention:**
  + **mapping a query and a set of key-value pairs to an output**
  + The output is computed as **a weighted sum of the values**, the weight is computed by a compatibility function of the **query** with the corresponding **key**
  + ![Transformer-image1]({{ site.url }}/my_img/Transformer-image1.png)
  + 在 encoder-decoder attention 层中，queries 来自 decoder，keys 和 values 来自 encoder
+ **Multi-head Attention:**
  + 将 queries, keys, values 通过 h 个学习到的线性映射，先进行映射，然后可以并行地分别计算 Attention，然后连接到一起，再进行一次映射
  + ![Transformer-image2]({{ site.url }}/my_img/Transformer-image2.png)
  + 学习不同的表示：Multi-head attention allows the model to jointly attend to information **from different representation**
    **subspaces at different positions**
+ **Position-wise Feed-Forward Networks**：一种比较直观的解释是，attention 的作用在于观察其他 token 并且收集其信息，而 FFN 的作用在于思考并且处理这些信息。
+ **Positional Encoding**

## 2. 有价值的内容

+ Attention mechanisms allowing modeling of dependencies **without regard to their distance** in the input or output sequences
+ Self-attention,  is an attention mechanism relating different positions of a single sequence **in order to compute a representation of the sequence**
+ 残差连接（residual connection）：深层网络可以在保持底层输入信息的同时学习新的特征，有效地解决了梯度消失和网络退化问题。
+ Decoder 的输入有两部分：一部分是来自编码器最后一层的输出，这部分提供了关于输入序列的上下文信息；另一部分是解码器自己的前一次的输出，用于生成下一个输出元素。在训练过程中，为了提高效率和避免信息泄露，通常使用「教师强制」（teacher forcing）策略，即直接将目标序列（向右偏移一个位置）作为输入而不是使用模型的预测作为下一时间步的输入。
+ Decoder 的输出是序列中的下一个元素的预测
+ 尽管 Decoder 的输入和输出看起来都是「output」，但它们在训练和推理（解码）过程中的使用方式有所不同：
  - 在**训练过程**中，Decoder 的输入通常是目标序列本身（教师强制），而输出是模型预测的序列。目标序列和预测序列之间通常存在一个时间步的偏移，预测序列试图匹配这个向右偏移的目标序列。
  - 在**推理过程**中（比如生成文本），Decoder 的输入是到目前为止模型生成的序列（可能仅包含一个起始符号），而输出是下一个时间步的预测。然后，这个预测被添加到序列中，并作为下一时间步的输入，这个过程一直重复，直到生成结束符号或达到某个长度限制为止。

## 3. 文中提到的其他内容

### 3.1 End-to-end memory networks（端到端记忆网络）

End-to-end memory networks（端到端记忆网络）是一种深度学习模型，设计用于处理需要一定记忆能力的序列化任务，比如问答系统、语言模型、机器翻译等。这种模型通过在神经网络中引入可读写的记忆组件（memory components），使网络能够在处理输入数据时存储和访问重要信息。

End-to-end memory networks 的核心是一个记忆矩阵，其中每一行都代表一个记忆单元，用于存储过去的信息。当模型接收到新的输入时，它会使用一个注意力机制（attention mechanism）来决定哪些记忆单元是相关的，并根据这些相关的记忆来生成输出。这种机制使得模型能够利用之前存储的信息来做出更加准确的判断或预测。

这种模型的"端到端"（end-to-end）特性指的是它可以直接从输入到输出进行训练，而不需要人为地设计特征提取或其他预处理步骤。这种端到端的训练方式使得模型可以在训练过程中自动学习到如何最有效地使用其记忆。

End-to-end memory networks在自然语言处理（NLP）等领域有广泛的应用，因为它们能够处理复杂的依赖关系和长距离的上下文信息，这在处理语言数据时是非常重要的。

### 3.2 Dropout

Dropout 是一种正则化技术，用于防止神经网络过拟合。在训练过程中，Dropout 通过随机将网络中的一部分输出单元归零，来减少单元间复杂的共适应关系。这意味着在每次训练迭代中，网络的每个单元都有一定概率被“丢弃”，从而每次都训练一个略有不同的网络。

### 3.3 Label Smoothing

标签平滑（Label Smoothing）是一种在训练深度学习模型时使用的正则化技术，特别是在分类任务中。

标签平滑通过将硬标签修改为更加“柔和”的分布来解决这个问题。具体来说，对于每个样本的真实标签，我们保留大部分的权重在正确的类别上，但也给其他类别分配一个小的、非零的权重。这样做的效果是鼓励模型在做出预测时变得更加谨慎，即使对于其非常确定的预测也不会给出 100% 的自信度。这种做法可以增加模型的泛化能力，因为它防止了模型在训练数据上的过拟合。
