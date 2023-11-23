---
title: Paper Reading UGACHE
author: xuao
date: 2023-11-23 17:27:00 +800
categories: [Paper Reading]
tags: [EmbDL,GPU,Multi-GPU,Cache,Deep Learning,MILP]
---

# UGACHE: A Unified GPU Cache for Embedding-based Deep Learning

## * 精简概括

### *.1 What

### *.2 Why

### *.3 How

## 0 Abstract

是一种用于 EmbDL 的统一多 GPU 缓存系统。

动机是 EmbDL 应用的独特特性，即：**只读（read-only）、批处理（batched）、偏斜（skewed）和可预测的嵌入访问（predictable embedding accesses）**.

UGACHE 引入了一种新颖的**因式提取机制**，避免了带宽拥塞，充分利用高速跨 GPU 互连（例如 NVLink 和 NVSwitch）.

基于一种**新的热度度量**，UGACHE 还提供了一种接近最优的**缓存策略**，平衡本地和远程访问，以最小化提取时间.

> Keywords: GPU cache, Embedding, GPU interconnect

## 1 Introduction

EmbDL 可以通过将稀疏输入（如用户 ID 和图节点）映射到嵌入表中的嵌入条目来高效处理它们.

与 DLR 和 GNN 应用相关的**嵌入表大小**通常可以达到高达 400 GB，超过了 GPU 内存的有限容量。因此，通过 PCIe 带宽从主机内存获取嵌入条目已经成为系统瓶颈，因为它的速度比 GPU 的高带宽内存（HBM）慢了一个数量级.

在实际工作负载中，访问嵌入条目通常呈现出一种**偏斜模式**，即某些条目比其他条目更频繁地被访问.

具有高速互连（例如，NVLink 和 NVSwitch）的多 GPU 平台已经成为现代数据中心的主流。在这种设置中，每个 GPU 可以直接访问其他 GPU 的内存，带宽比访问主机内存高一个数量级。但是，部署单 GPU 缓存系统在多 GPU 平台上是低效的，因为每个 GPU 会独立地缓存相同的 frequently accessed embeddings.

UGACHE，一个嵌入式缓存系统（embedding cache system），解决了与**提取机制**和**缓存策略**相关的挑战，有效地统一了多个 GPU 的内存.

> 引入了一种新颖的因子提取机制，用于实现跨 GPU 的嵌入式提取，**静态地**分配不同的 GPU 核心来从不同的源获取嵌入式条目。这种分配将以前的随机、不受限制的并行性变成有组织的并行性，从而避免带宽拥塞和由此引发的核心停滞。
>
> 缓存策略，关键是在「缓存更多不同的条目以提高**全局命中率**」和「缓存更多副本以提高**本地命中率**」之间找到平衡。UGACHE 引入了一个称为“热度”的新指标，用于衡量嵌入式条目的访问频率，并将多 GPU 平台上的提取时间建模为混合整数线性规划（MILP）问题

UGACHE 可以找到一个接近最优的解决方案，以最小化提取时间

UGACHE 充当了一个嵌入层，可以无缝集成到嵌入式深度学习应用程序工作流中

## 2 Characterizing Embedding-based DL

传统的深度学习模型无法直接处理稀疏输入；因此，引入了一个嵌入层来将其转换为密集输入。

嵌入表（EMT）表示从稀疏输入到密集值的映射。嵌入层从嵌入表中为每个键提取相应的条目，到一个连续的输出区域。这些提取的嵌入条目可以与密集层和密集输入一起用于进一步的计算。

在 EmbDL 中，端到端时间通常由嵌入层决定。

> 典型的嵌入层的大小可以达到数百 G 字节，远远超过现有 GPU 内存的容量，并且通常存储在主机内存中。
>
> 由于 PCIe 带宽限制，从主机传输嵌入到 GPU 所花费的时间成为系统瓶颈

总结了 embedding access 的几个特点：

+ 只读访问（Read-only access）
+ 批处理、子集访问（Batched, subset access）
+ 倾斜访问（Skewed access）
+ 稳定、可预测的访问（Stable, predictable access）：访问嵌入的倾斜性是可预测和稳定的

## 3 Challenges of Multi-GPU Embedding Cache

单 GPU 的嵌入缓存，嵌入层的时间仍然主导着端到端的时间。

在跨多个 GPU 创建嵌入式缓存面临两个基本挑战，即缓存策略（cache policy）和提取机制（extraction mechanism）。缓存策略必须合理地将嵌入式条目分布在多个 GPU 上，而提取机制必须高效地利用快速连接在多个 GPU 之间获取嵌入式。

### 3.1 缓存策略 Cache Policy

现有的多 GPU 嵌入式缓存的缓存策略可以分为**复制（replication）**和**分区（partition）**两类。

复制缓存：每个 GPU 独立地缓存最热门的条目，问题是每个 GPU 的缓存涵盖相似的请求

分区缓存：平均分区最热门的条目在每个 GPU 之间，尽可能地缓存许多单独的条目，并通过快速 GPU 互连服务大多数访问，面临边际效果和低本地命中率的问题。

### 3.2 提取机制 Extraction Mechanism

现有多 GPU 平台的**拓扑结构和带宽的多样性和限制**为开发高效的提取机制带来了挑战。

为在多 GPU 平台上执行提取过程，现有系统可分为两类：基于消息的（message-based）和基于对等的（peer-based）方法.

基于对等的访问，意味着 GPU 可以直接加载或存储非本地地址，消除了基于消息的方法中的冗余数据移动，并实现了零拷贝的嵌入式提取。

## 4 Overview of UGACHE

UGache，一个用于 EmbDL 应用的统一多 GPU 嵌入式缓存系统.

UGache 通过在多个 GPU 之间缓存嵌入式条目，同时隐藏平台细节，无缝地增强了现有的嵌入式层。

在内部，UGache 包括两个主要组件：**提取器（Extractor）**和**求解器（Solver）**，分别解决提取机制和缓存策略中的两个基本挑战。

![UGACHE-image1]({{ site.url }}/my_img/UGACHE-image1.png)

Extractor 提供了一种分解的提取机制，用于从多个源中提取嵌入式条目。其核心思想是静态地**将 GPU 核心专用于访问不同的源**。这种静态专用限制了同时访问同一链接的 GPU 核心数，避免了带宽拥塞和随之而来的核心停滞。

Extractor 还引入了本地提取填充以容忍由专用引起的潜在负载不平衡。

Solver 确定将嵌入式条目放置在多个 GPU 上的缓存策略，并为提取器提供在哪里找到这些条目的指导。

为了获得最佳的缓存策略，Solver 在「缓存更多不同的条目以提高全局命中率」和「缓存更多副本以提高本地命中率」之间找到平衡。为了实现这种平衡，Solver 定义了一个热度度量来衡量每个条目的访问频率，并对硬件平台的信息进行剖析以估计嵌入式提取时间。然后，它利用混合整数线性规划（MILP）来解决一个缓存策略，以最小化提取时间。

在前台，Extractor 处理 lookup 请求。

在后台，Solver 解决新的缓存策略，并在必要时调用 Filler 来填充（或重新填充）缓存内容。

Extractor 和 Solver 之间的协调通过 per-GPU 哈希表实现。每个缓存的嵌入条目都与其位置关联，通过 `<GPU_i, Offset>`。该格式指导 Extractor 获取该嵌入条目，Solver 和 Filler 则更新哈希表和缓存内容。

## 5 Extraction Mechanism

> the major challenge is handling the conflict between high parallelism of GPUs and limited bandwidth to non-local memory.

**GPU 的高并行性**和**非本地内存有限带宽**之间的冲突

### 5.1 提取过程的特征 Characteristics of Extarction Procedure

非本地内存只能容忍一部分核心占用其带宽

### 5.2 性能问题：链路拥塞 Performance Issues: Link Congestion

现有系统采用基于对等的方法来提取条目，

当前批次中的输入键被随机分派到 GPU 核心。随机分派的方式在并发核心从相同源提取时很容易导致链路拥塞，使得朝向源位置的链路带宽耗尽。随机分派方式会导致瞬时的负载不平衡。

慢速链路容忍较少的核心，迫使超额分配的核心停滞，导致核心利用率低

### 5.3 分解提取机制 Factorized Extraction Mechanism (FEM)

UGache 提出了一种分解提取机制（FEM）。其思想是专门**为每个源位置分配核心**，限制链接容忍性内的并发，防止链接拥塞和核心停滞。

![UGACHE-image2]({{ site.url }}/my_img/UGACHE-image2.png)

> 疑惑：
> 首先，关于 cores 和链路容量的对应关系，在这篇论文里，链路容量应该是最起码能够满足当前一台 GPU 全部 cores 均去使用这个链路。
>
> 或者它是以一台 GPU 全部 cores 使用当前链路为上限、基准。

访问 embedding 的位置，包括：

**local GPU memory, remote GPU memory with varying bandwidths, and host memory**

访问过程，优先访问非本地，也即先将所有 cores 用于访问非本地内存。

对于非本地的内存访问，静态分配一小部分 cores 用于访问 HOST 端，剩下的 cores 根据链路带宽容量静态分配给不同的其他 GPU。

这样做，保证了访问 GPU 不会导致超过基准（一台 GPU 全部 cores 使用当前链路）。

当从某个远程的访问完成后，分配给这部分的 cores 就去处理本地的内存访问。

> 个人点评：
>
> 优点如文章所说，防止链接拥塞和核心停滞。
>
> 但是静态分配感觉利用效率底下，尤其 GPU 的操作不只有访存，虽然论文中也论证了，嵌入层的时间主导着端到端的时间。
>
> 缺乏灵活性（即根据链路负载动态调整的能力），或者说，更适用于高负载。
>
> 论文中的原话也是：
>
> > The static dedication allows UGache to fully utilize all GPU cores **for concurrent embedding extraction**

## 6 Cache Policy

UGACHE 的 Solver 提供了缓存策略，一方面是指定 embedding entries 在多个 GPU 上的放置位置，另一方面是指导 Extractor 去定位 embedding entries.

Solver 定义了一个 hotness metric 来建模访问频率。通过 hotness 和 profiled 信息，Solver 将缓存策略建模为一个 MILP 问题，提供能够最小化预估的 extraction 时间的解决方案。

### 6.1 热度度量 Hotness Metric

UGACHE 允许应用直接提供热度信息，以增加灵活性。

> 模型训练的第一轮就可以预估访问频率，因为每轮都会访问相同的 dataset
>
> GNN 中，一个结点的度可以近似访问频率

### 6.2 建模提取时间 Modelling of Extraction Time

> Using hotness and platform information, UGache’s Solver builds a model to estimate the time spent on embedding extraction with the arrangement variation of **embedding storage** and **embedding access**

storage arrangement 描述了放置 embedding entry 的位置

access arrangement 描述了当一个 embedding entry 可以从多个 GPU 读取时，应该选择哪个 source location

Solver 通过 access arrangement 建模和最小化提取时间，同时限制 storage arrangement.

####  6.2.1  Storage Arrangement

![UGACHE-image3]({{ site.url }}/my_img/UGACHE-image3.png)

第一个不等式表示：若 GPUi 想去位置 j 读取 embedding e，那么 j 需要存储了 e

第二个不等式表示：存储在位置 j 的总 embedding，应该小于 j 的容量

#### 6.2.2 Time Estimation

![UGACHE-image4]({{ site.url }}/my_img/UGACHE-image4.png)

$T_{i\leftarrow j}$ 表示 GPUi 从位置 j 读取一个 embedding entry 的时间，包含了带宽和拓扑信息

![UGACHE-image5]({{ site.url }}/my_img/UGACHE-image5.png)

$R_{i\leftarrow j}$ 代表了 GPUi 上分配到从 j 读取 embedding 的 cores 的比例

最终要由优化的目标为：
$$
minimize \ z, \ \ \ \ z \ge t_i,\forall i ∈ G
$$
即最小化所有 GPUi 的访问时间

### 6.3 复杂度、近似和优化 Complexity, Approximation and Optimizations

MILP 问题是 NP-complete，求解时间与 E 和 G 呈指数，E 是 embedding entries 的数目，G 是 GPU 的数目

UGACHE 提出一个近似解法，思想就是**其思想是将相似的 entries 分组（into blocks），一起决定它们的放置和访问策略**。

UGACHE 采用了两种优化策略：

+ 将 hotness 对数化，再判断相似程度
+ controls the size of a single block by dividing it into smaller ones

结果就是降低了解决 MILP 的时间，同时近似最优结果与理论最优结果的平均差异不到 2%

> 但是具体到代码，Solver 函数，看上去并不是严格按照论文中说的那样实现的，出现过三次，在三个类内出现过：
>
> `void OptimalSolver::Solve`
>
> ![UGACHE-image6]({{ site.url }}/my_img/UGACHE-image6.png)
>
> 只有判断一个 block 是否在 host，是否在某个远程 GPU 上，而且居然还可以设置为连续变量。
>
> `void CollFineGrainSolver::Solve`:
>
> ![UGACHE-image7]({{ site.url }}/my_img/UGACHE-image7.png)
>
> 判断一个 block 是否在 host，是否在某个远程 GPU 上，链路选择

## 7 Implementation

### 7.1 系统集成 System Integration

### 7.2 缓存刷新 Cache Refresh

在后台，UGache 的刷新器收集统计信息并定期使用新的热度重新评估 Solver 的模型。当估计的提取时间显著增加时，将触发刷新过程。

### 7.3 硬件要求 Hardware Requirements

**Core dedication**：UGache 利用 MP S的 API 创建多个 GPU 上下文，并限制 GPU 核心占用，并在这些上下文上启动内核以实现 GPU 核心的专用。

**Kernel priority**：为了处理潜在的负载不平衡，必须早期启动本地提取，但在非本地位置的提取内核之后进行调度。UGache 利用 CUDA 中的流优先级（stream priority），使本地提取在具有较低优先级的流上启动

## 8 Evaluation

> static cache design with no online eviction

### 8.3 性能细分

> 当缓存比例较小时，UGache 产生了类似于分区的缓存策略，对分区的改进主要归因于 UGache 的提取机制，提高了 1.72 倍。 
>
> 随着缓存比例的增加，UGache 的缓存策略通过利用足够的缓存容量来平衡本地和全局命中率。分歧点和缓存策略带来的改进取决于数据集的偏斜。最终，在高缓存比例下，缓存策略始终主导 UGache 的性能改进。

## # 概念学习

### #.1 Embedding-based Deep Learning (EmbDL)

基于嵌入的深度学习。

嵌入是将高维离散数据（如单词、图像像素、用户 ID 等）转换为低维连续向量的过程，从而使学习算法更加有效。这些向量捕捉了原始数据的重要特征和关系。

### #.2 GPU's high bandwidth memory (HBM)

GPU 的高性能内存技术。提供更高的内存带宽和更小的物理尺寸。

### #.3 power-law distribution

幂律分布。它描述了一个变量的概率与其取值呈幂函数关系的情况

具体而言，一个变量的取值 x 的概率密度函数按照如下形式变化：
$$
P(X \ge x) ∝ x^{-\alpha}
$$
这些系统中，少数的事件或数值非常大，而大多数事件或数值相对较小。

在网络科学中，例如在社交网络或互联网拓扑中，节点的度（连接数）分布经常符合幂律分布。这表示有一些节点（称为“超级节点”）具有非常高的度，而大多数节点具有较低的度

### #.4 Long-tail effect

长尾效应。

指在一组数据中，少数现象具有非常高的频率或出现概率，而大多数现象却相对较低的现象。

### #.5 GPU topology

GPU 互连的拓扑结构可以分为两种类型：硬连线（hard-wired）和交换机基础（switch-based）

在硬连线平台中，单个 GPU 的总出站带宽在所有连接的远程 GPU 之间被物理分配，形成一个均匀的全连接图

在交换机基础平台中，所有 GPU 直接连接到 NVSwitch，其中不同 GPU 对之间的带宽是动态分配的。虽然 NVSwitch 的带宽能够支持所有 GPU 的全出站带宽，但当一个 GPU 同时被多个 GPU 访问时，仍然可能发生带宽冲突

### #.6 MILP

混合整数线性规划问题。

这些变量中同时包含了连续变量和离散变量，那么这类问题被称为混合整数规划问题
