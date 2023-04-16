---
title: Learning PowerLyra
author: xuao
date: 2023-04-16 20:19:00 +800
categories: [Graph Computing]
tags: [Graph,IPADS,Distribute,Hybrid-cut,Skewed Graph]
---

## 一、是什么

PowerLyra: Differentiated Graph Computation and Partitioning on Skewed Graphs

是一个分布式图处理系统，融合了现有图并行系统的两个最佳方案。PowerLyra 使用**集中计算**来处理低度顶点，避免频繁通信，并将度数高的顶点的计算**分布**以平衡工作负载。

> Specifically, PowerLyra uses centralized computation for low-degree vertices to avoid frequent communications and distributes the computation for high-degree vertices to balance workloads. 

PowerLyra 还提供了一种高效的混合图分区算法（即 **hybrid-cut** ），将边割（用于低度顶点）和顶点割（用于高度顶点）与启发式算法相结合。为了提高节点间图访问的缓存局部性，PowerLyra 还提供了一种局部感知的数据布局优化。

## 二、为什么

自然图具有偏斜分布，给分布式图计算和分区带来独特的挑战。现有的图并行系统通常使用“一刀切”的设计，均匀处理所有顶点，这会导致高度不平衡的负载和高度竞争高度顶点，或者即使对于低度顶点也会产生高通信成本和高内存消耗.

Pregel 中和 GraphLab 中采用的方法被称为 edge-cut 的方法，它的优点就是有更加好的局部性。可以试图将可能地降低跨越了分区的边的数量，缺点就是由于顶点的度数在很多自然的数据中是发布不平衡的，少数的顶点拥有大量的边，而其它的部分的顶点的边的数量比较少，这种情况下就会导致负载不均衡的现象。而 PowerGraph、GrpahX 之类的系统采用的 vertex-cut 的方法好处就是解决了一些数据分布不均衡的问题，能获得更高的并行性。缺点就是会造成比较大的复制因子(the average number of replicas for a vertex)，另外就是增加通信量，对于一些低度数的点这样的分区更多是无用的。所以，既然不同的分区方式有不同的优缺点，PowerLyra 就应用一种混合的方法。

## 三、怎么做

在图划分的时候，和 PowerGraph 一样，PowerLyra 将一个顶点的副本中使用一种 hash 算法随机选择一个作为 Master，其余的作为 Mirrors，也同样适用 GAS 的编程模型。对于顶点计算的处理，PowerLyra 采用了 Hybrid 的方法，处理 high-degree 的顶点时，为了获取更高的并行性，PowerLyra 基本上采用的就是 PowerGraph 的方法，做了一些的修改。

处理 low-degree 的顶点时，为了获取更好的局部性，PowerLyra 又采用了类似 GraphLab 的方式，但是并没有想 GrapgLab 那些双向边都考虑，而是注意到实际的算法中，这个局部性往往只体现在一个方向，所以 PowerLyra 提出了一种单向的局部性。还有 PowerLyra 根据一个自适应的策略来处理不同算法在 gathering 和 scattering 数据的时候在应用在不同类型的边的问题(出边，入边).

在大规模图计算中，图的划分方法对于计算性能和通信开销具有重要影响。传统的划分方法包括 Vertex-cut 和 Edge-cut，它们在处理不同类型的图时具有不同的优势。Vertex-cut 适用于处理高度偏斜的图，因为它可以将高度连接的节点划分到不同的分区以平衡计算和通信负载；而 Edge-cut 适用于处理均匀分布的图，因为它可以将边沿着节点进行划分，从而最小化通信开销。

Hybrid-cut 是一种结合了 Vertex-cut 和 Edge-cut 优势的划分方法。它根据节点的度数（与节点连接的边数）来判断如何划分图。具体来说，Hybrid-cut 使用以下两种策略：

1. 对于低度数节点，使用 Edge-cut：当节点的度数低于某个阈值时，将其边划分到不同的分区。这样可以在处理低度数节点时最小化通信开销。
2. 对于高度数节点，使用 Vertex-cut：当节点的度数超过某个阈值时，将节点本身划分到不同的分区，以便在多个分区之间共享计算和通信负载。

## * 概念补充

### 1. Skewed Graph

指具有高度不平衡度分布的图数据。在这种类型的图中，有一些节点的度数（即连接的边数）远远高于其他节点，这导致图数据的处理和分析变得更加困难。
