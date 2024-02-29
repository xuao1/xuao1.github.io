---
title: Paper Reading Sia
author: xuao
date: 2023-11-25 21:31:00 +800
categories: [Paper Reading]
tags: [GPU,Cluster,Scheduling,MILP,ILP,Heterogeneity]
---

# Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling

## * 精简概括

### *.1 What

### *.2 Why

### *.3 How

## 0 Abstract

Sia 调度器高效地将异构深度学习集群资源分配给弹性的资源自适应作业

> efficiently assigns heterogeneous deep learning (DL) cluster resources to elastic resource-adaptive jobs

Sia 引入了一种新的调度表达方式，可以适应搜索空间的大小，有意地匹配作业及其配置与 GPU 类型和数量，同时适应变化的集群负载和作业混合（job mix）

Sia 还引入了一种 low-profiling-overhead 的方法，用于引导（对于每个新作业）用于评估可能的资源分配的吞吐量模型（throughput models），并且它是第一个支持混合并行作业弹性扩展的集群调度器

> Key words: cluster scheduling, resource allocation, deep learning training

## 1 Introduction

现有的 DL cluster 的调度器，要么只考虑了集群中的 GPU 异构性，要么只考虑了 adaptive jobs.

**Sia is a new scheduler designed for「resource-adaptive DL training jobs」and「heterogeneous resources」**

在每个调度轮中，Sia 考虑了将 GPU（数量和类型）分配给当前作业的每种可能方式，选择下一个时间段的最佳集群资源分配。

但是有两个挑战：

+ 搜索空间巨大
+ 不同的 DL jobs 会经历不同的性能变化，当比较不同的 GPU 的类型和扩展能力时

Sia 使用：

+ 一个新的 solver formulation，来解决规模挑战
+ 一个新的方法，来在线学习 per-job per-GPU-type **throughput models**

Sia 的 ILP formulation 和搜索空间缩减，使其可以高效找到最佳分配，甚至当工作负载和集群大小增加时。

Sia 的吞吐量建模（throughput models）方法，避免了大量的 profiling.

> Sia 通过每个 GPU 类型的一个最小配置的 profiles 来引导每个新作业的吞吐量模型，最初假定在尚未知配置的情况下进行简单的缩放/投影，并在作业使用不同配置时动态细化模型

Sia 被实现为开源的 AdaptDL 框架中的插件兼容调度器替代品

发现 Sia 的优势随着集群负载/竞争的增加而增加

在异构集群上，对于自适应作业来说，动态调整作业资源分配（GPU 类型和数量）是至关重要的。这样是 Sia 性能优异的原因。

Sia 还在公平性指标上超越了其他调度器。

## 2 DL cluster scheduling and related work

训练任务并行，大多采用 synchronous data parallelism，梯度计算阶段在 GPU 之间划分，然后将它们同步，更新每个 GPU 上的模型参数。

**Elastic and resource-adaptive DL jobs**：数据并行的深度学习任务可以随时间弹性地调整大小，通过进行检查点操作，然后在不同数量的 GPU 上重新启动。

### 2.1 Related work in DL cluster scheduling

Gavel 是最先进的异构 DL 集群调度器，采用快速的线性规划形式，适用于大型集群。将 Gavel 扩展为处理作业自适应性并非易事。

Pollux 是一种先进的 DL 集群调度器，用于同质集群的弹性资源自适应作业。Pollux 的无预先配置的吞吐量建模方法阻碍了对 GPU 异构性的考虑。集群大小的扩展性也非常差。

> 上一段第二句没懂

## 3 Sia Design and Implementation

Sia 是一种抢占式、基于轮次的调度器，目标是最大化整个集群的吞吐量。

在每个轮次中，作业接收资源捆绑包（包括 CPU、GPU 和网络）

Sia 使用检查点还原抢占（checkpoint-restore preemption）来优化作业的适应性

一个 Job 在 Sia 中的生命周期如下：

![Sia-image1]({{ site.url }}/my_img/Sia-image1.png)

+ Job 提交后，它在每种 GPU 类型上对几种批处理大小进行测试。Good Estimator 在每种 GPU 类型的测试中，引导 throughput model. Job 在不同资源配置下的吞吐量估计（goodput estimates）会提供给 Policy 优化器。
+ 在获得资源分配后，该作业开始了在集群中的持续优化循环（步骤 5-8）。Policy 不断优化作业的资源分配，Goodput Estimator 提供最新的性能和梯度统计信息，以协助策略进行决策。

### 3.1 Sia components and job life cycle

在循环（步骤 5-8）中，Job 的适应性（adaptivity）按以下操作持续优化。

#### 3.1.1 持续优化的作业适应性 Continuously optimized job adaptivity

Policy 使用 Goodput Estimator 提供的 goodput estimates，为 Job 提供最佳的资源分配。

Placer 根据分配数量，确定具体的 GPU，并尝试减少 Job 迁移。

Sia 在 Adaptive Executors 上运行 Job，这些执行器支持：

1. 透明的 checkpoint-restore for **low-overhead job preemption** and **resource scaling**
2. 批处理大小的适应性（batchsize adaptivity）
3. 对梯度和吞吐量信息的频繁报告

Goodput Estimator 使用 3 的信息更新 Job 的 goodput model.

在下一轮调度中，Policy 查询 Job 在所有 GPU 类型上的 goodput extimates，为 Job 提供最佳的资源分配。

#### 3.1.2 异构执行 Heterogeneous Execution

Sia 透明地处理 GPU 异构：将 GPU 内存容量、互连速度和吞吐量在 Goodput Estimator 中进行建模。

如果统计效率要求比 GPU 内存限制支持的更大的批处理大小，将使用梯度累积（Gradient accumulation）

#### 3.1.3 作业扩展策略 Job Scaling policy

每个作业从 1 个 GPU 开始，并在每个调度轮次最多扩展 2 倍.

#### 3.1.4 解耦分配和放置 Decoupled allocation and placement

一组异构资源，要被分配给一组 Jobs. 分配过程分为两个阶段：

+ Allocation：确定每个 Job 要分配资源的数量和类型
+ Placement：确定确切的物理分配

> 疑惑：
>
> 文中提到「这种解耦允许我们限制分配的放置空间」，不明白为什么

在 Placer 中遵循 3 个规则：

1. patial 节点的分配不能跨越两个节点（partial 节点，指的是分配的 GPU 小于其要求的最大 GPU 数）
2. whole node allocations must take whole nodes
3. 如果前两条得不到满足，驱逐一些 Jobs 并重试

> 对于第二点，我的理解是：当一个 Job 在 Allocation 阶段请求整个节点的计算资源时，那么在 Place 阶段，就应该给它物理上的整个节点

将 allocations 限制为特定集合，能够保证 Sia 的的 allocations 都得到有效的 placement（将在 3.3 节中讨论）

### 3.2 吞吐量模型的引导 Bootstrapping of throughput models

吞吐量模型是作为 GPU 数量和批处理大小（batchsize）的函数。

Sia 从最小化的分析信息开始，并根据观察到的分配进行调整（refine based on observed allocations）。

对于每个 Job，Sia：

+ 为每种 GPU 类型学习一个 throughput model
+ 学习一个统计效率模型 statistical effciency model

> 根据论文，这个阶段不会考虑 GPU 数量，因为 Sia 是使用数据并行结合 all-reduce 来进行扩展的，即利用多个 GPU
>
> 这个阶段只会考虑 GPU 类型和 batchsize

Sia 使用每种 GPU 类型的 1-GPU 分析文件初始化 Job 的吞吐量模型，最初也是将 Job 放在某个类型的 1-GPU 上运行。

在线学习：

+ 学习统计效率模型
+ 调整在当前类型 1-GPU 上的吞吐量模型

此时 Sia 无法考虑通信时间（目前只在一个 GPU 运行），假设两个数据并行副本的吞吐量是单个副本的两倍。

在之后如果 Job 用到了多 GPU，那么在线分析就可以使用测量的通信时间来优化在当前 GPU 类型上的吞吐量模型。

此外，Sia将 Job 对于 A GPU 的学习到的吞吐量模型与最初对 A 和 B 单 GPU 吞吐量进行的分析文件结合起来，得到 B GPU的一个粗略的引导吞吐量模型：

![Sia-image2]({{ site.url }}/my_img/Sia-image2.png)

### 3.3 配置 Configurations

配置代表一组资源（CPU、GPU、网络等），用 (n, r, t) 表示，节点数量，资源数量，资源类型

Policy 提供的 allocation 涵盖一个小的配置集合，用来简化 Placer 中的放置逻辑。

配置集合为：

![Sia-image3]({{ site.url }}/my_img/Sia-image3.png)

单节点集将分配限制为节点内的 2 的幂，并且最多为 R，即节点内 GPU 的数量.

多节点要求对每个结点，配置其全部 GPU。

依靠 Submesh Shape Covering theorem 确保了没有两个分布式作业共享任何节点。

限制配置集合，一方面并没有降低性能，另一方面降低了问题的复杂性，增强了 Sia 的扩展性。

### 3.4 调度器目标 Scheduler objective

#### 3.4.1 有效配置 Valid configurations

#### 3.4.2 吞吐量估计 Goodput estimation

Sia 对每个（Job，GPU 类型）使用一个吞吐量模型

使用 pre-Job 统计效率模型来推导吞吐量估计，每个（Job，GPU 类型）一个。

定义一个大小为 $|J|\times |C|$ 的吞吐量矩阵，$G_{ij}$ 表示第 i 个 Job 在配置 j 上的吞吐量估计。

将矩阵 G 标准化一下，使其可以：

+ 每行最大值，反应对当前 Job 的最佳配置
+ 每列最大值，反应当前配置在哪个 Job 上被最好地使用

#### 3.4.3 调度器目标 Scheduler objective

Sia 选择矩阵 G 中的（Job，配置）对，最大化标准吞吐量之和。

定义一个与 G 相同大小的矩阵 A，$A_{ij} = 1$ 表示 Job $J_i$ 选择配置 $c_j$.

优化问题变为：

![Sia-image4]({{ site.url }}/my_img/Sia-image4.png) 

将 Sia 调度器的目标变成了一个（二进制）整数线性规划问题，二元矩阵 A 作为优化变量，约束为：

1. 每个 Job 最多选择一个配置：$||A_i||_1 \le 1$
2. 分配的 GPU 数量不超过其可用数量

求解出的 A 给出了下一个调度轮次的分配。

#### 3.4.4 重启因子 Restart Factor

本质考虑了重启开销，折扣 Job 在非当前配置上的其他配置的标准吞吐量，以避免改变配置以及额外的检查点恢复开销。

#### 3.4.5 平衡吞吐量和公平性 Balancing goodput with fairness

引入了因子 p，p 取负可以翻转原本的优化目标

![Sia-image5]({{ site.url }}/my_img/Sia-image5.png)

> 没看懂是要做什么

#### 3.4.6 有限的适应性的支持 Support for limited adaptivity

Sia 支持执行某个 Job with some adaptations disabled (batch size, GPU count and/or type)

对方程和约束进行一些修改，Sia 的调度方案可以支持调度定制的资源请求和具有用户定义的并行性的作业

#### 3.4.6 抢占和保留 Preemption and reservation

Sia 假设所有作业都是可抢占的，但也可以支持集群中的少量非抢占性作业

#### 3.4.7 支持其他并行化技术 Support for other parallelization techniques

Sia 只需要 Job 提供一个在有效配置上可评估的 goodput 估算器。

这种设计使得 Sia 能够支持使用更先进的并行化策略的作业

#### 3.4.8 调度其他工作负载类型 Scheduling other workload types

论文认为 Sia 也可以通过使用定制的 goodput 估算器来处理其他批处理工作负载类型.

例如，对于对延迟敏感的推断作业，如果一个配置可以在承诺的延迟约束内支持推断，则 goodput=1，否则为 0

### 3.5 Implementation

使用开源的 AdaptDL 框架实现了 Sia，用自己的调度程序和数据加载程序替换了其实现。

Sia 自适应执行器不断对小批量运行时间和梯度统计进行分析，定期（默认 30 秒）使用这些分析优化 goodput 模型参数

将 3.4.5 中的公式制定为混合整数线性规划，使用 CVXPY 包中的 GLPK_MI 求解器，并使用输出解决方案确定作业分配。

使用检查点恢复的预抢占。如果 DL 训练作业的分配发生变化，Sia 仅在当前小批量处理完成后抢占作业，以确保没有正在进行的通信。

Sia 还使用检查点-恢复机制来从工作进程故障中恢复

## 4 Experimental Setup

## 5 Evaluation

## 6 Conclusion

Sia 在异构资源上高效地调度自适应的深度学习作业，同时协同调整每个作业的 GPU 数量、GPU 类型和批大小，从而提高了深度学习集群的性能。

## # 概念学习

### #.1 JCT

Job Completion Time. 作业完成时间

### #.2 ILP

整数线性规划（Integer Linear Programming）

用于解决线性规划问题，其中目标是最大化或最小化线性方程的值，同时约束条件为线性不等式和等式。整数线性规划在标准线性规划的基础上增加了一个约束条件，即决策变量必须是整数

### #.3 checkpoint-restore preemption

「Checkpoint-restore preemption」是一种调度机制，其核心思想是在作业的执行过程中进行检查点（checkpoint）以保存当前作业的状态，以便在需要抢占时能够迅速恢复到该状态。这个机制可以在需要释放资源以分配给其他作业时，或者在系统需要回收资源以提高整体效率时发挥作用。

具体而言，当一个作业被抢占时，它的当前状态（通常是内存中的数据和执行上下文）被保存到一个检查点。这个检查点可以存储在磁盘或其他持久性存储中。之后，系统可以使用这个检查点来快速还原作业的状态，使其能够在之前中断的地方继续执行，而无需从头开始重新计算。
